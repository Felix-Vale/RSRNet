from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        k: int = 3,
        s: int = 1,
        p: Optional[int] = None,
        groups: int = 1,
        act: bool = True,
    ):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=k,
            stride=s,
            padding=p,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


# ============================================================
# Schedule / Stage Control
# ============================================================

@dataclass
class CandidateStageConfig:
    epoch: int
    warmup_epochs: int = 10
    gate_start_epoch: int = 30
    total_epochs: int = 100
    tau_start: float = 0.30
    tau_end: float = 0.55
    temp_start: float = 1.0
    temp_end: float = 0.25
    detach_until_gate: bool = True
    direction_start_epoch: int = 8

    def stage_mode(self) -> str:
        if self.epoch < self.warmup_epochs:
            return "warmup"
        if self.epoch < self.gate_start_epoch:
            return "soft"
        return "gate"

    def tau(self) -> float:
        if self.epoch < self.gate_start_epoch:
            return self.tau_start
        progress = (self.epoch - self.gate_start_epoch) / max(1, self.total_epochs - self.gate_start_epoch)
        progress = min(max(progress, 0.0), 1.0)
        return self.tau_start + (self.tau_end - self.tau_start) * progress

    def temp(self) -> float:
        if self.epoch < self.gate_start_epoch:
            return self.temp_start
        progress = (self.epoch - self.gate_start_epoch) / max(1, self.total_epochs - self.gate_start_epoch)
        progress = min(max(progress, 0.0), 1.0)
        return self.temp_start + (self.temp_end - self.temp_start) * progress

    def detach_pred(self) -> bool:
        return self.detach_until_gate and (self.epoch < self.gate_start_epoch)

    def enable_direction(self) -> bool:
        return self.epoch >= self.direction_start_epoch


# ============================================================
# Evidence Builder A: Transition Evidence Tp
# ============================================================

class TransitionEvidence(nn.Module):
    def forward(self, prob: torch.Tensor) -> torch.Tensor:
        return 4.0 * prob * (1.0 - prob)


class MultiScaleTransitionEvidence(nn.Module):
    def __init__(self, pool_scales: Sequence[int] = (1, 2, 4), fuse_ch: int = 8):
        super().__init__()
        self.pool_scales = list(pool_scales)
        self.base = TransitionEvidence()
        self.fuse = nn.Sequential(
            ConvBNAct(len(self.pool_scales), fuse_ch, 3),
            nn.Conv2d(fuse_ch, 1, kernel_size=1, bias=True),
        )

    def _resize_to(self, x: torch.Tensor, ref_hw: Tuple[int, int]) -> torch.Tensor:
        if x.shape[-2:] == ref_hw:
            return x
        return F.interpolate(x, size=ref_hw, mode="bilinear", align_corners=False)

    def forward(self, prob: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        h, w = prob.shape[-2:]
        outs = []
        raw = []
        for s in self.pool_scales:
            if s == 1:
                p = prob
            else:
                p = F.avg_pool2d(prob, kernel_size=s, stride=s, ceil_mode=False)
            tp_s = self.base(p)
            raw.append(tp_s)
            outs.append(self._resize_to(tp_s, (h, w)))
        ms_tp = torch.sigmoid(self.fuse(torch.cat(outs, dim=1)))
        return ms_tp, outs


# ============================================================
# Evidence Builder B: Structural Variation Evidence Ts
# ============================================================

class StructuralVariationEvidence(nn.Module):
    def __init__(self, mode: str = "dw_diff", dilation: int = 1):
        super().__init__()
        self.mode = mode
        self.dilation = dilation
        if mode == "dw_diff":
            self.dw = nn.Conv2d(
                1,
                1,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                groups=1,
                bias=False,
            )
        elif mode == "conv_block":
            self.block = nn.Sequential(
                ConvBNAct(1, 8, 3, p=dilation, groups=1),
                ConvBNAct(8, 8, 3, p=dilation, groups=1),
                nn.Conv2d(8, 1, kernel_size=1, bias=True),
            )
        else:
            raise ValueError(f"Unsupported structural variation mode: {mode}")

    def forward(self, prob: torch.Tensor) -> torch.Tensor:
        if self.mode == "dw_diff":
            return torch.abs(self.dw(prob) - prob)
        return torch.abs(self.block(prob))


class MultiScaleStructuralVariationEvidence(nn.Module):
    def __init__(
        self,
        mode: str = "dw_diff",
        dilations: Sequence[int] = (1, 2, 3),
        fuse_ch: int = 8,
    ):
        super().__init__()
        self.branches = nn.ModuleList([
            StructuralVariationEvidence(mode=mode, dilation=d) for d in dilations
        ])
        self.fuse = nn.Sequential(
            ConvBNAct(len(dilations), fuse_ch, 3),
            nn.Conv2d(fuse_ch, 1, kernel_size=1, bias=True),
        )

    def forward(self, prob: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        ts_list = [branch(prob) for branch in self.branches]
        ms_ts = torch.sigmoid(self.fuse(torch.cat(ts_list, dim=1)))
        return ms_ts, ts_list


# ============================================================
# Evidence Builder C: Feature Conflict Evidence Tf
# ============================================================

class FeatureConflictEvidence(nn.Module):
    def __init__(self, in_ch: int, out_ch: int = 8, kernel_size: int = 3):
        super().__init__()
        self.kernel_size = kernel_size
        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        pad = self.kernel_size // 2
        local_mean = F.avg_pool2d(feat, kernel_size=self.kernel_size, stride=1, padding=pad)
        diff = torch.abs(feat - local_mean)
        return self.proj(diff)


class MultiScaleFeatureConflictEvidence(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int = 8,
        kernel_sizes: Sequence[int] = (3, 5, 7),
        fuse_ch: int = 16,
    ):
        super().__init__()
        self.branches = nn.ModuleList([
            FeatureConflictEvidence(in_ch=in_ch, out_ch=out_ch, kernel_size=k) for k in kernel_sizes
        ])
        self.fuse = nn.Sequential(
            ConvBNAct(out_ch * len(kernel_sizes), fuse_ch, 3),
            nn.Conv2d(fuse_ch, out_ch, kernel_size=1, bias=True),
        )

    def forward(self, feat: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        tf_list = [branch(feat) for branch in self.branches]
        ms_tf = self.fuse(torch.cat(tf_list, dim=1))
        return ms_tf, tf_list


# ============================================================
# Vessel Direction Prior
# ============================================================

def _build_line_kernel(kernel_size: int, angle: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    k = torch.zeros((kernel_size, kernel_size), device=device, dtype=dtype)
    c = kernel_size // 2

    if angle == 0:
        k[c, :] = 1.0
    elif angle == 90:
        k[:, c] = 1.0
    elif angle == 45:
        for i in range(kernel_size):
            k[i, kernel_size - 1 - i] = 1.0
    elif angle == 135:
        for i in range(kernel_size):
            k[i, i] = 1.0
    else:
        raise ValueError(f"Unsupported angle: {angle}")

    k = k / (k.sum() + 1e-6)
    return k


class VesselDirectionalPrior(nn.Module):
    def __init__(
        self,
        feat_ch: int,
        mid_ch: int = 8,
        kernel_size: int = 7,
        angles: Sequence[float] = (0, 45, 90, 135),
        fuse_ch: int = 8,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.angles = list(angles)
        self.feat_reduce = nn.Sequential(
            nn.Conv2d(feat_ch, mid_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, 1, kernel_size=1, bias=True),
        )
        self.pre_fuse = nn.Sequential(
            ConvBNAct(2, fuse_ch, 3),
            nn.Conv2d(fuse_ch, 1, kernel_size=1, bias=True),
        )
        self.out_fuse = nn.Sequential(
            ConvBNAct(2, fuse_ch, 3),
            nn.Conv2d(fuse_ch, 1, kernel_size=1, bias=True),
        )

    def _directional_response_bank(self, x: torch.Tensor) -> torch.Tensor:
        kernels = []
        for angle in self.angles:
            k = _build_line_kernel(self.kernel_size, angle, x.device, x.dtype)
            kernels.append(k)
        weight = torch.stack(kernels, dim=0).unsqueeze(1)  # [K,1,kh,kw]
        pad = self.kernel_size // 2
        resp = F.conv2d(x, weight, bias=None, stride=1, padding=pad)  # [B,K,H,W]
        return resp

    def forward(self, feat: torch.Tensor, prob: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        feat_map = self.feat_reduce(feat)
        base_map = torch.sigmoid(self.pre_fuse(torch.cat([feat_map, prob], dim=1)))

        resp_bank = self._directional_response_bank(base_map)
        max_resp, _ = resp_bank.max(dim=1, keepdim=True)
        mean_resp = resp_bank.mean(dim=1, keepdim=True)
        coherence = (max_resp - mean_resp).clamp(min=0.0)

        dir_prior = torch.sigmoid(self.out_fuse(torch.cat([max_resp, coherence], dim=1)))

        aux = {
            "dir_base_map": base_map,
            "dir_resp_bank": resp_bank,
            "dir_max_resp": max_resp,
            "dir_coherence": coherence,
            "dir_prior": dir_prior,
        }
        return dir_prior, aux


# ============================================================
# Soft Gate Builder
# ============================================================

class ProgressiveGateMap(nn.Module):
    def forward(self, cand_map: torch.Tensor, tau: float, temp: float) -> torch.Tensor:
        temp = max(float(temp), 1e-6)
        return torch.sigmoid((cand_map - tau) / temp)


# ============================================================
# Main Plugin Module
# ============================================================

class CandidateBoundaryRegionPluginV2(nn.Module):
    def __init__(
        self,
        feat_ch: int,
        tf_ch: int = 8,
        fuse_ch: int = 32,
        ts_mode: str = "dw_diff",
        prob_pool_scales: Sequence[int] = (1, 2, 4),
        ts_dilations: Sequence[int] = (1, 2, 3),
        tf_kernel_sizes: Sequence[int] = (3, 5, 7),
        use_direction_prior: bool = True,
        direction_kernel_size: int = 7,
        direction_start_epoch: int = 8,
    ):
        super().__init__()
        self.use_direction_prior = use_direction_prior
        self.direction_start_epoch = direction_start_epoch

        self.ms_transition = MultiScaleTransitionEvidence(
            pool_scales=prob_pool_scales,
            fuse_ch=8,
        )
        self.ms_structural_variation = MultiScaleStructuralVariationEvidence(
            mode=ts_mode,
            dilations=ts_dilations,
            fuse_ch=8,
        )
        self.ms_feature_conflict = MultiScaleFeatureConflictEvidence(
            in_ch=feat_ch,
            out_ch=tf_ch,
            kernel_sizes=tf_kernel_sizes,
            fuse_ch=max(16, tf_ch * 2),
        )
        if self.use_direction_prior:
            self.direction_prior = VesselDirectionalPrior(
                feat_ch=feat_ch,
                mid_ch=max(8, tf_ch),
                kernel_size=direction_kernel_size,
                angles=(0, 45, 90, 135),
                fuse_ch=8,
            )
        self.gate_builder = ProgressiveGateMap()

        in_evidence_ch = 1 + 1 + tf_ch + (1 if self.use_direction_prior else 0)
        self.fuse = nn.Sequential(
            ConvBNAct(in_evidence_ch, fuse_ch, 3),
            ConvBNAct(fuse_ch, fuse_ch, 3),
            nn.Conv2d(fuse_ch, 1, kernel_size=1, bias=True),
        )

    def forward(
        self,
        feat: torch.Tensor,
        logit: torch.Tensor,
        stage_mode: str = "warmup",
        detach_pred: bool = True,
        tau: float = 0.4,
        temp: float = 1.0,
        return_aux: bool = True,
        enable_direction_prior: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        if feat.dim() != 4:
            raise ValueError(f"feat must be 4D tensor [B,C,H,W], got shape={feat.shape}")
        if logit.dim() != 4:
            raise ValueError(f"logit must be 4D tensor [B,1,H,W], got shape={logit.shape}")
        if logit.shape[1] != 1:
            raise ValueError(f"logit channel must be 1 for binary segmentation candidate map, got {logit.shape[1]}")
        if feat.shape[0] != logit.shape[0]:
            raise ValueError("feat and logit batch size mismatch")
        if feat.shape[-2:] != logit.shape[-2:]:
            raise ValueError(
                f"feat spatial size {feat.shape[-2:]} and logit spatial size {logit.shape[-2:]} must match. "
                "Please align them outside the plugin first."
            )

        prob = torch.sigmoid(logit)
        prob_in = prob.detach() if detach_pred else prob

        # 1) 多尺度概率证据
        tp_ms, tp_scale_list = self.ms_transition(prob_in)
        ts_ms, ts_scale_list = self.ms_structural_variation(prob_in)

        # 2) 多尺度特征冲突
        tf_ms, tf_scale_list = self.ms_feature_conflict(feat)

        evidence_list = [tp_ms, ts_ms, tf_ms]
        aux: Dict[str, torch.Tensor] = {
            "prob": prob,
            "tp_ms": tp_ms,
            "ts_ms": ts_ms,
            "tf_ms": tf_ms,
        }
        for i, x in enumerate(tp_scale_list):
            aux[f"tp_scale_{i}"] = x
        for i, x in enumerate(ts_scale_list):
            aux[f"ts_scale_{i}"] = x
        for i, x in enumerate(tf_scale_list):
            aux[f"tf_scale_{i}"] = x

        # 3) 血管方向先验
        direction_flag = self.use_direction_prior if enable_direction_prior is None else enable_direction_prior
        if direction_flag:
            dir_prior, dir_aux = self.direction_prior(feat=feat, prob=prob_in)
            evidence_list.append(dir_prior)
            aux.update(dir_aux)
        else:
            dir_prior = torch.zeros_like(tp_ms)
            aux["dir_prior"] = dir_prior

        evidence = torch.cat(evidence_list, dim=1)
        cand_map = torch.sigmoid(self.fuse(evidence))

        if stage_mode in ["warmup", "soft"]:
            gate_map = torch.ones_like(cand_map)
        elif stage_mode == "gate":
            gate_map = self.gate_builder(cand_map, tau=tau, temp=temp)
        else:
            raise ValueError(f"Unsupported stage_mode: {stage_mode}")

        aux["cand_map"] = cand_map
        aux["gate_map"] = gate_map

        if not return_aux:
            aux = {}
        return cand_map, gate_map, aux


# ============================================================
# Optional Weak Supervision Utilities
# ============================================================

@torch.no_grad()
def mask_to_boundary_band(mask: torch.Tensor, radius: int = 2) -> torch.Tensor:
    mask = (mask > 0.5).float()
    k = 2 * radius + 1
    dilated = F.max_pool2d(mask, kernel_size=k, stride=1, padding=radius)
    eroded = 1.0 - F.max_pool2d(1.0 - mask, kernel_size=k, stride=1, padding=radius)
    band = (dilated - eroded).clamp(0.0, 1.0)
    return band


def candidate_boundary_bce_loss(cand_map: torch.Tensor, gt_mask: torch.Tensor, radius: int = 2) -> torch.Tensor:
    target_band = mask_to_boundary_band(gt_mask, radius=radius)
    return F.binary_cross_entropy(cand_map, target_band)


def candidate_sparse_ratio_loss(cand_map: torch.Tensor, target_ratio: float = 0.20) -> torch.Tensor:
    return torch.abs(cand_map.mean() - cand_map.new_tensor(target_ratio))


def vessel_direction_consistency_loss(
    cand_map: torch.Tensor,
    dir_prior: torch.Tensor,
    weight_map: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if weight_map is None:
        weight_map = dir_prior.detach()
    return ((cand_map - dir_prior.detach()).abs() * weight_map).mean()


# ============================================================
# Example Usage
# ============================================================

def example_usage():
    B, C, H, W = 2, 64, 64, 64
    feat = torch.randn(B, C, H, W)
    logit = torch.randn(B, 1, H, W)

    config = CandidateStageConfig(
        epoch=12,
        warmup_epochs=10,
        gate_start_epoch=30,
        total_epochs=100,
        direction_start_epoch=8,
    )

    plugin = CandidateBoundaryRegionPluginV2(
        feat_ch=C,
        tf_ch=8,
        fuse_ch=32,
        ts_mode="dw_diff",
        prob_pool_scales=(1, 2, 4),
        ts_dilations=(1, 2, 3),
        tf_kernel_sizes=(3, 5, 7),
        use_direction_prior=True,
        direction_kernel_size=7,
    )

    cand_map, gate_map, aux = plugin(
        feat=feat,
        logit=logit,
        stage_mode=config.stage_mode(),
        detach_pred=config.detach_pred(),
        tau=config.tau(),
        temp=config.temp(),
        return_aux=True,
        enable_direction_prior=config.enable_direction(),
    )

    print("cand_map:", cand_map.shape)
    print("gate_map:", gate_map.shape)
    print("tp_ms:", aux["tp_ms"].shape)
    print("ts_ms:", aux["ts_ms"].shape)
    print("tf_ms:", aux["tf_ms"].shape)
    print("dir_prior:", aux["dir_prior"].shape)


if __name__ == "__main__":
    example_usage()
