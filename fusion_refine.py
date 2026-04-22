from dataclasses import dataclass
from typing import Dict, Optional, Tuple

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
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


@dataclass
class FusionRefineStageConfig:
    epoch: int
    warmup_epochs: int = 10
    gate_start_epoch: int = 30

    def stage_mode(self) -> str:
        if self.epoch < self.warmup_epochs:
            return "warmup"
        if self.epoch < self.gate_start_epoch:
            return "soft"
        return "gate"


def _check_4d(name: str, x: torch.Tensor):
    if x.dim() != 4:
        raise ValueError(f"{name} must be a 4D tensor [B,C,H,W], got shape={tuple(x.shape)}")


# ============================================================
# Weight / Mask Builders
# ============================================================


class FusionWeightHead(nn.Module):
    def __init__(self, in_ch: int, hidden_ch: int = 64):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNAct(in_ch, hidden_ch, 3),
            nn.Conv2d(hidden_ch, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualRefineBlock(nn.Module):
    def __init__(self, in_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNAct(in_ch, in_ch, 3),
            ConvBNAct(in_ch, in_ch, 3, act=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

class BoundaryGuidedFusionRefinementPlugin(nn.Module):
    def __init__(
        self,
        in_ch: int,
        num_classes: int = 1,
        hidden_ch: int = 64,
        use_relation_branch: bool = True,
        use_adaptive_branch: bool = True,
        use_candidate_for_fusion: bool = True,
        use_gate_for_fusion: bool = True,
        detach_candidate_in_fusion: bool = True,
        use_residual: bool = True,
        residual_alpha: float = 1.0,
        soft_stage_scale: float = 0.5,
        gate_stage_scale: float = 1.0,
        use_stage_scaled_injection: bool = True,
        refine_with_main_only_when_missing: bool = True,
        return_intermediate_features: bool = True,
    ):
        super().__init__()
        self.in_ch = in_ch
        self.num_classes = num_classes
        self.use_relation_branch = use_relation_branch
        self.use_adaptive_branch = use_adaptive_branch
        self.use_candidate_for_fusion = use_candidate_for_fusion
        self.use_gate_for_fusion = use_gate_for_fusion
        self.detach_candidate_in_fusion = detach_candidate_in_fusion
        self.use_residual = use_residual
        self.residual_alpha = residual_alpha
        self.soft_stage_scale = soft_stage_scale
        self.gate_stage_scale = gate_stage_scale
        self.use_stage_scaled_injection = use_stage_scaled_injection
        self.refine_with_main_only_when_missing = refine_with_main_only_when_missing
        self.return_intermediate_features = return_intermediate_features

        # 关系分支 / 自适应卷积分支投影头
        self.rel_proj = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
        )
        self.adc_proj = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
        )

        self.rel_weight_head = FusionWeightHead(in_ch * 3 + 2, hidden_ch)
        self.adc_weight_head = FusionWeightHead(in_ch * 3 + 2, hidden_ch)

        self.post_mix = nn.Sequential(
            ConvBNAct(in_ch, in_ch, 3),
            ConvBNAct(in_ch, in_ch, 3, act=False),
        )

        self.refine_block = ResidualRefineBlock(in_ch)

        self.seg_head = nn.Conv2d(in_ch, num_classes, kernel_size=1, bias=True)

    # ------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------
    def _validate_inputs(
        self,
        feat_main: torch.Tensor,
        feat_rel: Optional[torch.Tensor],
        feat_adc: Optional[torch.Tensor],
        cand_map: Optional[torch.Tensor],
        gate_map: Optional[torch.Tensor],
    ):
        _check_4d("feat_main", feat_main)
        b, c, h, w = feat_main.shape
        if c != self.in_ch:
            raise ValueError(f"feat_main channel must equal in_ch={self.in_ch}, got {c}")

        for name, x in [("feat_rel", feat_rel), ("feat_adc", feat_adc)]:
            if x is None:
                continue
            _check_4d(name, x)
            if x.shape != feat_main.shape:
                raise ValueError(
                    f"{name} must align with feat_main. feat_main={tuple(feat_main.shape)}, {name}={tuple(x.shape)}"
                )

        for name, x in [("cand_map", cand_map), ("gate_map", gate_map)]:
            if x is None:
                continue
            _check_4d(name, x)
            if x.shape[0] != b or x.shape[1] != 1 or x.shape[-2:] != (h, w):
                raise ValueError(
                    f"{name} must be [B,1,H,W] and aligned with feat_main. "
                    f"feat_main={tuple(feat_main.shape)}, {name}={tuple(x.shape)}"
                )

    def _resolve_map(self, cand_map: Optional[torch.Tensor], gate_map: Optional[torch.Tensor], feat_main: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, _, h, w = feat_main.shape
        zero_map = feat_main.new_zeros((b, 1, h, w))

        cand_in = cand_map if cand_map is not None else zero_map
        gate_in = gate_map if gate_map is not None else zero_map

        if self.detach_candidate_in_fusion:
            cand_in = cand_in.detach()
            gate_in = gate_in.detach()

        return cand_in, gate_in

    def _stage_scale(self, stage_mode: str) -> float:
        if not self.use_stage_scaled_injection:
            return 1.0
        if stage_mode == "warmup":
            return 0.0
        if stage_mode == "soft":
            return float(self.soft_stage_scale)
        if stage_mode == "gate":
            return float(self.gate_stage_scale)
        raise ValueError(f"Unsupported stage_mode: {stage_mode}")

    def _build_refine_mask(
        self,
        feat_main: torch.Tensor,
        cand_map: Optional[torch.Tensor],
        gate_map: Optional[torch.Tensor],
        stage_mode: str,
    ) -> torch.Tensor:
        b, _, h, w = feat_main.shape
        zero_map = feat_main.new_zeros((b, 1, h, w))

        if stage_mode == "warmup":
            return zero_map
        if stage_mode == "soft":
            return cand_map if cand_map is not None else zero_map
        if stage_mode == "gate":
            if gate_map is not None:
                return gate_map
            if cand_map is not None:
                return cand_map
            return zero_map
        raise ValueError(f"Unsupported stage_mode: {stage_mode}")

    # ------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------
    def forward(
        self,
        feat_main: torch.Tensor,
        feat_rel: Optional[torch.Tensor] = None,
        feat_adc: Optional[torch.Tensor] = None,
        cand_map: Optional[torch.Tensor] = None,
        gate_map: Optional[torch.Tensor] = None,
        stage_mode: str = "warmup",
        return_aux: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:

        self._validate_inputs(feat_main, feat_rel, feat_adc, cand_map, gate_map)
        b, c, h, w = feat_main.shape

        aux: Dict[str, torch.Tensor] = {}
        stage_scale = self._stage_scale(stage_mode)
        cand_in, gate_in = self._resolve_map(cand_map, gate_map, feat_main)

        relation_available = feat_main.new_zeros((b,))
        adaptive_available = feat_main.new_zeros((b,))
        if feat_rel is not None and self.use_relation_branch:
            relation_available = feat_main.new_ones((b,))
        if feat_adc is not None and self.use_adaptive_branch:
            adaptive_available = feat_main.new_ones((b,))

        # ------------------------------
        # Stage 1: warmup
        # ------------------------------
        if stage_mode == "warmup":
            refined_feat = feat_main
            refine_logit = self.seg_head(refined_feat)
            if return_aux:
                aux = {
                    "rel_weight": feat_main.new_zeros((b, 1, h, w)),
                    "adc_weight": feat_main.new_zeros((b, 1, h, w)),
                    "refine_mask": feat_main.new_zeros((b, 1, h, w)),
                    "fused_feat": feat_main if self.return_intermediate_features else feat_main.new_zeros((b, c, h, w)),
                    "delta_map": feat_main.new_zeros((b, c, h, w)),
                    "relation_available": relation_available,
                    "adaptive_available": adaptive_available,
                    "stage_flag": feat_main.new_zeros((b,)),
                }
            return refined_feat, refine_logit, aux


        if feat_rel is not None and self.use_relation_branch:
            feat_rel_p = self.rel_proj(feat_rel)
        else:
            feat_rel_p = feat_main.new_zeros((b, c, h, w))

        if feat_adc is not None and self.use_adaptive_branch:
            feat_adc_p = self.adc_proj(feat_adc)
        else:
            feat_adc_p = feat_main.new_zeros((b, c, h, w))


        fuse_in = torch.cat([feat_main, feat_rel_p, feat_adc_p, cand_in, gate_in], dim=1)
        rel_weight = self.rel_weight_head(fuse_in)
        adc_weight = self.adc_weight_head(fuse_in)

        if feat_rel is None or (not self.use_relation_branch):
            rel_weight = rel_weight * 0.0
        if feat_adc is None or (not self.use_adaptive_branch):
            adc_weight = adc_weight * 0.0

        rel_weight = rel_weight * stage_scale
        adc_weight = adc_weight * stage_scale

        feat_mix = feat_main + rel_weight * feat_rel_p + adc_weight * feat_adc_p
        feat_mix = feat_mix + self.post_mix(feat_mix)

        refine_mask = self._build_refine_mask(feat_main, cand_map, gate_map, stage_mode).clamp(0.0, 1.0)

        no_extra_branch = (feat_rel is None or not self.use_relation_branch) and (feat_adc is None or not self.use_adaptive_branch)
        if no_extra_branch and (not self.refine_with_main_only_when_missing):
            delta_map = feat_main.new_zeros((b, c, h, w))
        else:
            delta_map = self.refine_block(feat_mix)

        delta_map = delta_map * refine_mask

        if self.use_residual:
            refined_feat = feat_main + self.residual_alpha * delta_map
        else:
            refined_feat = delta_map

        refine_logit = self.seg_head(refined_feat)

        if return_aux:
            if stage_mode == "soft":
                stage_flag = feat_main.new_ones((b,))
            elif stage_mode == "gate":
                stage_flag = feat_main.new_full((b,), 2.0)
            else:
                stage_flag = feat_main.new_zeros((b,))

            aux = {
                "rel_weight": rel_weight,
                "adc_weight": adc_weight,
                "refine_mask": refine_mask,
                "fused_feat": feat_mix if self.return_intermediate_features else feat_main.new_zeros((b, c, h, w)),
                "delta_map": delta_map,
                "relation_available": relation_available,
                "adaptive_available": adaptive_available,
                "stage_flag": stage_flag,
            }

        return refined_feat, refine_logit, aux


def refinement_mask_sparsity_loss(refine_mask: torch.Tensor, target_ratio: float = 0.20) -> torch.Tensor:
    return torch.abs(refine_mask.mean() - refine_mask.new_tensor(target_ratio))


def example_usage():
    torch.manual_seed(42)

    B, C, H, W = 2, 64, 64, 64
    feat_main = torch.randn(B, C, H, W)
    feat_rel = torch.randn(B, C, H, W)
    feat_adc = torch.randn(B, C, H, W)
    cand_map = torch.sigmoid(torch.randn(B, 1, H, W))
    gate_map = torch.sigmoid((cand_map - 0.45) / 0.30)

    plugin = BoundaryGuidedFusionRefinementPlugin(
        in_ch=C,
        num_classes=1,
        hidden_ch=64,
        use_relation_branch=True,
        use_adaptive_branch=True,
        use_candidate_for_fusion=True,
        use_gate_for_fusion=True,
        detach_candidate_in_fusion=True,
        use_residual=True,
        residual_alpha=1.0,
        soft_stage_scale=0.5,
        gate_stage_scale=1.0,
        use_stage_scaled_injection=True,
        refine_with_main_only_when_missing=True,
        return_intermediate_features=True,
    )

    # warmup
    refined_warmup, logit_warmup, aux_warmup = plugin(
        feat_main=feat_main,
        feat_rel=feat_rel,
        feat_adc=feat_adc,
        cand_map=cand_map,
        gate_map=gate_map,
        stage_mode="warmup",
        return_aux=True,
    )
    print("warmup refined_feat:", refined_warmup.shape, "logit:", logit_warmup.shape)

    # soft
    refined_soft, logit_soft, aux_soft = plugin(
        feat_main=feat_main,
        feat_rel=feat_rel,
        feat_adc=feat_adc,
        cand_map=cand_map,
        gate_map=gate_map,
        stage_mode="soft",
        return_aux=True,
    )
    print("soft refined_feat:", refined_soft.shape, "logit:", logit_soft.shape)
    print("soft refine_mask:", aux_soft["refine_mask"].shape)

    # gate
    refined_gate, logit_gate, aux_gate = plugin(
        feat_main=feat_main,
        feat_rel=feat_rel,
        feat_adc=feat_adc,
        cand_map=cand_map,
        gate_map=gate_map,
        stage_mode="gate",
        return_aux=True,
    )
    print("gate refined_feat:", refined_gate.shape, "logit:", logit_gate.shape)
    print("rel_weight:", aux_gate["rel_weight"].shape)
    print("adc_weight:", aux_gate["adc_weight"].shape)
    print("delta_map:", aux_gate["delta_map"].shape)


if __name__ == "__main__":
    example_usage()
