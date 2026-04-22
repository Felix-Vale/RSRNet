
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

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
        d: int = 1,
        groups: int = 1,
        act: bool = True,
    ):
        super().__init__()
        if p is None:
            p = ((k - 1) // 2) * d
        self.conv = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=k,
            stride=s,
            padding=p,
            dilation=d,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class IdentityOrConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        if in_ch == out_ch:
            self.block = nn.Identity()
        else:
            self.block = ConvBNAct(in_ch, out_ch, k=1, p=0, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


@dataclass
class AdaptiveConvStageConfig:
    epoch: int
    warmup_epochs: int = 10
    gate_start_epoch: int = 30

    def stage_mode(self) -> str:
        if self.epoch < self.warmup_epochs:
            return "warmup"
        if self.epoch < self.gate_start_epoch:
            return "soft"
        return "gate"


def _check_4d(name: str, x: Optional[torch.Tensor]):
    if x is None:
        return
    if x.dim() != 4:
        raise ValueError(f"{name} must be 4D tensor [B,C,H,W], got shape={tuple(x.shape)}")


# ============================================================
# Direction Utilities
# ============================================================


class DirectionFieldEstimator(nn.Module):
    def __init__(self, in_ch: int, hidden_ch: int = 16):
        super().__init__()
        self.structure_proj = nn.Sequential(
            nn.Conv2d(in_ch, hidden_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_ch, 1, kernel_size=1, bias=True),
        )

        # 使用固定差分核估计局部梯度，再转成切线方向
        gx = torch.tensor([[1, 0, -1],
                           [2, 0, -2],
                           [1, 0, -1]], dtype=torch.float32).view(1, 1, 3, 3)
        gy = torch.tensor([[1, 2, 1],
                           [0, 0, 0],
                           [-1, -2, -1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("gx_kernel", gx, persistent=False)
        self.register_buffer("gy_kernel", gy, persistent=False)

    def forward(self, feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        s = self.structure_proj(feat)
        gx = F.conv2d(s, self.gx_kernel, padding=1)
        gy = F.conv2d(s, self.gy_kernel, padding=1)

        tx = -gy
        ty = gx
        norm = torch.sqrt(tx * tx + ty * ty + 1e-6)
        dir_x = tx / norm
        dir_y = ty / norm
        return dir_x, dir_y, s


# ============================================================
# Noise / Update Utilities
# ============================================================


class FeatureNoiseSuppressor(nn.Module):
    def __init__(self, in_ch: int, hidden_ch: int = 16):
        super().__init__()
        self.conflict_proj = nn.Sequential(
            nn.Conv2d(in_ch, hidden_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_ch),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            ConvBNAct(hidden_ch + 1, hidden_ch, k=3),
            nn.Conv2d(hidden_ch, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, feat: torch.Tensor, structure_map: torch.Tensor) -> torch.Tensor:
        local_mean = F.avg_pool2d(feat, kernel_size=3, stride=1, padding=1)
        conflict = torch.abs(feat - local_mean)
        conflict = self.conflict_proj(conflict)
        structure_strength = structure_map.abs()
        noise_score = self.head(torch.cat([conflict, structure_strength], dim=1))
        # 返回 suppress map：值越大表示越可信，越适合动态更新
        return 1.0 - noise_score


class BoundaryUpdateMaskBuilder(nn.Module):
    def __init__(
        self,
        soft_stage_scale: float = 0.5,
        gate_stage_scale: float = 1.0,
        warmup_scale: float = 0.0,
    ):
        super().__init__()
        self.soft_stage_scale = soft_stage_scale
        self.gate_stage_scale = gate_stage_scale
        self.warmup_scale = warmup_scale

    def forward(
        self,
        feat: torch.Tensor,
        cand_map: Optional[torch.Tensor],
        gate_map: Optional[torch.Tensor],
        stage_mode: str,
    ) -> torch.Tensor:
        b, _, h, w = feat.shape
        if stage_mode == "warmup":
            return feat.new_full((b, 1, h, w), float(self.warmup_scale))
        if stage_mode == "soft":
            if cand_map is None:
                return feat.new_full((b, 1, h, w), float(self.soft_stage_scale))
            return cand_map.clamp(0.0, 1.0) * float(self.soft_stage_scale)
        if stage_mode == "gate":
            if gate_map is None:
                if cand_map is None:
                    return feat.new_full((b, 1, h, w), float(self.gate_stage_scale))
                return cand_map.clamp(0.0, 1.0) * float(self.gate_stage_scale)
            return gate_map.clamp(0.0, 1.0) * float(self.gate_stage_scale)
        raise ValueError(f"Unsupported stage_mode: {stage_mode}")


# ============================================================
# Context Banks
# ============================================================


class MultiScaleContextBank(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        branch_dilations: Tuple[int, ...] = (1, 2, 3),
        depthwise_separable: bool = False,
    ):
        super().__init__()
        self.branch_dilations = tuple(branch_dilations)
        self.num_branches = len(self.branch_dilations)

        branches: List[nn.Module] = []
        for d in self.branch_dilations:
            if depthwise_separable:
                branch = nn.Sequential(
                    ConvBNAct(in_ch, in_ch, k=3, d=d, groups=in_ch),
                    ConvBNAct(in_ch, out_ch, k=1, p=0, act=False),
                )
            else:
                branch = ConvBNAct(in_ch, out_ch, k=3, d=d, act=False)
            branches.append(branch)
        self.branches = nn.ModuleList(branches)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        return [branch(x) for branch in self.branches]


class DirectionAwareContextBank(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        direction_dilations: Tuple[int, ...] = (1, 2),
        direction_kernel_size: int = 5,
    ):
        super().__init__()
        if direction_kernel_size not in (3, 5, 7):
            raise ValueError("direction_kernel_size must be 3, 5 or 7 for this implementation")

        self.in_ch = in_ch
        self.out_ch = out_ch
        self.direction_dilations = tuple(direction_dilations)
        self.direction_kernel_size = int(direction_kernel_size)

        self.orientations = ("h", "v", "d1", "d2")
        self.num_branches = len(self.orientations) * len(self.direction_dilations)

        self.depthwise_convs = nn.ModuleList()
        self.pointwise_convs = nn.ModuleList()

        for d in self.direction_dilations:
            for ori in self.orientations:
                dw = nn.Conv2d(
                    in_ch,
                    in_ch,
                    kernel_size=self.direction_kernel_size,
                    padding=((self.direction_kernel_size - 1) // 2) * d,
                    dilation=d,
                    groups=in_ch,
                    bias=False,
                )
                dw.weight.data.copy_(self._build_direction_kernel(in_ch, self.direction_kernel_size, ori))
                dw.weight.requires_grad = True  # 允许在方向模板基础上微调，更灵活但仍稳定
                pw = ConvBNAct(in_ch, out_ch, k=1, p=0, act=False)
                self.depthwise_convs.append(dw)
                self.pointwise_convs.append(pw)

    def _build_direction_kernel(self, channels: int, k: int, ori: str) -> torch.Tensor:
        kernel = torch.zeros((channels, 1, k, k), dtype=torch.float32)
        center = k // 2
        if ori == "h":
            kernel[:, 0, center, :] = 1.0 / float(k)
        elif ori == "v":
            kernel[:, 0, :, center] = 1.0 / float(k)
        elif ori == "d1":
            for i in range(k):
                kernel[:, 0, i, i] = 1.0 / float(k)
        elif ori == "d2":
            for i in range(k):
                kernel[:, 0, i, k - 1 - i] = 1.0 / float(k)
        else:
            raise ValueError(f"Unsupported orientation: {ori}")
        return kernel

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        outs = []
        for dw, pw in zip(self.depthwise_convs, self.pointwise_convs):
            y = dw(x)
            y = pw(y)
            outs.append(y)
        return outs


# ============================================================
# Branch Gating
# ============================================================


class CandidateGuidedHybridBranchGating(nn.Module):
    def __init__(self, in_ch: int, hidden_ch: int, num_scale_branches: int, num_direction_branches: int):
        super().__init__()
        self.num_scale_branches = num_scale_branches
        self.num_direction_branches = num_direction_branches

        self.fuse = nn.Sequential(
            ConvBNAct(in_ch + 4, hidden_ch, k=3),
            ConvBNAct(hidden_ch, hidden_ch, k=3),
        )
        self.scale_head = nn.Conv2d(hidden_ch, num_scale_branches, kernel_size=1, bias=True)
        self.direction_head = nn.Conv2d(hidden_ch, num_direction_branches, kernel_size=1, bias=True)
        self.mix_head = nn.Sequential(
            nn.Conv2d(hidden_ch, hidden_ch // 2 if hidden_ch >= 16 else hidden_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_ch // 2 if hidden_ch >= 16 else hidden_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_ch // 2 if hidden_ch >= 16 else hidden_ch, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(
        self,
        feat: torch.Tensor,
        cand_map: Optional[torch.Tensor],
        gate_map: Optional[torch.Tensor],
        dir_x: torch.Tensor,
        dir_y: torch.Tensor,
        stage_mode: str,
        detach_candidate: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, _, h, w = feat.shape

        if cand_map is None:
            cand_map = feat.new_zeros((b, 1, h, w))
        if gate_map is None:
            gate_map = feat.new_zeros((b, 1, h, w))

        if detach_candidate:
            cand_in = cand_map.detach()
            gate_in = gate_map.detach()
        else:
            cand_in = cand_map
            gate_in = gate_map

        if stage_mode == "warmup":
            guide_a = cand_in.new_zeros(cand_in.shape)
            guide_b = gate_in.new_zeros(gate_in.shape)
        elif stage_mode == "soft":
            guide_a = cand_in
            guide_b = cand_in
        elif stage_mode == "gate":
            guide_a = cand_in
            guide_b = gate_in
        else:
            raise ValueError(f"Unsupported stage_mode: {stage_mode}")

        x = torch.cat([feat, guide_a, guide_b, dir_x, dir_y], dim=1)
        x = self.fuse(x)
        scale_weights = F.softmax(self.scale_head(x), dim=1)
        direction_weights = F.softmax(self.direction_head(x), dim=1)
        direction_mix = self.mix_head(x)  # 越大越依赖方向分支
        return scale_weights, direction_weights, direction_mix


# ============================================================
# Main Module
# ============================================================


class DirectionAwareAdaptiveContextConv(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: Optional[int] = None,
        hidden_ch: int = 64,
        branch_dilations: Tuple[int, ...] = (1, 2, 3),
        direction_dilations: Tuple[int, ...] = (1, 2),
        direction_kernel_size: int = 5,
        use_base_conv: bool = True,
        use_residual: bool = True,
        residual_alpha: float = 1.0,
        soft_stage_scale: float = 0.5,
        gate_stage_scale: float = 1.0,
        warmup_scale: float = 0.0,
        detach_candidate_in_gating: bool = True,
        refine_with_base_when_no_candidate: bool = True,
        depthwise_separable: bool = False,
        noise_suppress_alpha: float = 0.5,
        return_intermediate_features: bool = True,
    ):
        super().__init__()
        if out_ch is None:
            out_ch = in_ch

        self.in_ch = in_ch
        self.out_ch = out_ch
        self.hidden_ch = hidden_ch
        self.branch_dilations = tuple(branch_dilations)
        self.direction_dilations = tuple(direction_dilations)
        self.use_base_conv = use_base_conv
        self.use_residual = use_residual
        self.residual_alpha = float(residual_alpha)
        self.detach_candidate_in_gating = detach_candidate_in_gating
        self.refine_with_base_when_no_candidate = refine_with_base_when_no_candidate
        self.noise_suppress_alpha = float(noise_suppress_alpha)
        self.return_intermediate_features = return_intermediate_features

        self.base_conv = ConvBNAct(in_ch, out_ch, k=3) if use_base_conv else nn.Identity()

        self.direction_estimator = DirectionFieldEstimator(in_ch=in_ch, hidden_ch=max(in_ch // 4, 16))
        self.scale_bank = MultiScaleContextBank(
            in_ch=in_ch,
            out_ch=out_ch,
            branch_dilations=self.branch_dilations,
            depthwise_separable=depthwise_separable,
        )
        self.direction_bank = DirectionAwareContextBank(
            in_ch=in_ch,
            out_ch=out_ch,
            direction_dilations=self.direction_dilations,
            direction_kernel_size=direction_kernel_size,
        )

        self.branch_gating = CandidateGuidedHybridBranchGating(
            in_ch=in_ch,
            hidden_ch=hidden_ch,
            num_scale_branches=len(self.branch_dilations),
            num_direction_branches=self.direction_bank.num_branches,
        )

        self.update_mask_builder = BoundaryUpdateMaskBuilder(
            soft_stage_scale=soft_stage_scale,
            gate_stage_scale=gate_stage_scale,
            warmup_scale=warmup_scale,
        )

        self.noise_suppressor = FeatureNoiseSuppressor(in_ch=in_ch, hidden_ch=max(in_ch // 4, 16))

        self.context_refine = nn.Sequential(
            ConvBNAct(out_ch, out_ch, k=3),
            ConvBNAct(out_ch, out_ch, k=3, act=False),
        )

        self.residual_proj = IdentityOrConv(in_ch, out_ch)
        self.out_act = nn.ReLU(inplace=True)

    def _validate_inputs(
        self,
        feat: torch.Tensor,
        cand_map: Optional[torch.Tensor],
        gate_map: Optional[torch.Tensor],
    ):
        _check_4d("feat", feat)
        _check_4d("cand_map", cand_map)
        _check_4d("gate_map", gate_map)

        b, _, h, w = feat.shape
        if cand_map is not None:
            if cand_map.shape[0] != b or cand_map.shape[-2:] != (h, w) or cand_map.shape[1] != 1:
                raise ValueError("cand_map must be [B,1,H,W] and aligned with feat")
        if gate_map is not None:
            if gate_map.shape[0] != b or gate_map.shape[-2:] != (h, w) or gate_map.shape[1] != 1:
                raise ValueError("gate_map must be [B,1,H,W] and aligned with feat")

    def forward(
        self,
        feat: torch.Tensor,
        cand_map: Optional[torch.Tensor] = None,
        gate_map: Optional[torch.Tensor] = None,
        stage_mode: str = "warmup",
        return_aux: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        self._validate_inputs(feat, cand_map, gate_map)

        base_feat = self.base_conv(feat) if self.use_base_conv else feat

        dir_x, dir_y, structure_map = self.direction_estimator(feat)

        scale_feats = self.scale_bank(feat)
        scale_stack = torch.stack(scale_feats, dim=1)  # [B,Ks,C,H,W]

        direction_feats = self.direction_bank(feat)
        direction_stack = torch.stack(direction_feats, dim=1)  # [B,Kd,C,H,W]

        scale_weights, direction_weights, direction_mix = self.branch_gating(
            feat=feat,
            cand_map=cand_map,
            gate_map=gate_map,
            dir_x=dir_x,
            dir_y=dir_y,
            stage_mode=stage_mode,
            detach_candidate=self.detach_candidate_in_gating,
        )

        isotropic_context = (scale_stack * scale_weights.unsqueeze(2)).sum(dim=1)
        directional_context = (direction_stack * direction_weights.unsqueeze(2)).sum(dim=1)

        hybrid_context = (1.0 - direction_mix) * isotropic_context + direction_mix * directional_context
        dynamic_context = self.context_refine(hybrid_context)

        update_mask = self.update_mask_builder(
            feat=feat,
            cand_map=cand_map,
            gate_map=gate_map,
            stage_mode=stage_mode,
        )
        noise_suppress_map = self.noise_suppressor(feat, structure_map)
        effective_mask = update_mask * ((1.0 - self.noise_suppress_alpha) + self.noise_suppress_alpha * noise_suppress_map)

        delta_context = effective_mask * dynamic_context

        if self.use_base_conv:
            fused_feat = base_feat + delta_context
        else:
            fused_feat = feat if self.refine_with_base_when_no_candidate else dynamic_context
            fused_feat = fused_feat + delta_context

        if self.use_residual:
            residual = self.residual_proj(feat)
            refined_feat = residual + self.residual_alpha * fused_feat
        else:
            refined_feat = fused_feat

        refined_feat = self.out_act(refined_feat)

        if return_aux:
            aux: Dict[str, torch.Tensor] = {
                "base_feat": base_feat if isinstance(base_feat, torch.Tensor) else feat,
                "dir_x": dir_x,
                "dir_y": dir_y,
                "structure_map": structure_map,
                "scale_weights": scale_weights,
                "direction_weights": direction_weights,
                "direction_mix": direction_mix,
                "update_mask": update_mask,
                "noise_suppress_map": noise_suppress_map,
                "effective_mask": effective_mask,
                "dynamic_context": dynamic_context,
                "delta_context": delta_context,
                "refined_feat": refined_feat,
            }
            if self.return_intermediate_features:
                for idx, feat_i in enumerate(scale_feats):
                    aux[f"scale_branch_feat_d{self.branch_dilations[idx]}"] = feat_i
                for idx, feat_i in enumerate(direction_feats):
                    aux[f"direction_branch_feat_{idx}"] = feat_i
        else:
            aux = {}

        return refined_feat, aux


# ============================================================
# Optional Loss
# ============================================================


def adaptive_update_sparsity_loss(update_mask: torch.Tensor, target_ratio: float = 0.20) -> torch.Tensor:
    return torch.abs(update_mask.mean() - update_mask.new_tensor(target_ratio))


def directional_consistency_regularization(dir_x: torch.Tensor, dir_y: torch.Tensor, cand_map: Optional[torch.Tensor] = None) -> torch.Tensor:

    dx1 = torch.abs(dir_x[:, :, :, 1:] - dir_x[:, :, :, :-1])
    dy1 = torch.abs(dir_y[:, :, :, 1:] - dir_y[:, :, :, :-1])
    dx2 = torch.abs(dir_x[:, :, 1:, :] - dir_x[:, :, :-1, :])
    dy2 = torch.abs(dir_y[:, :, 1:, :] - dir_y[:, :, :-1, :])

    loss = dx1.mean() + dy1.mean() + dx2.mean() + dy2.mean()
    if cand_map is not None:
        w1 = cand_map[:, :, :, 1:]
        w2 = cand_map[:, :, 1:, :]
        loss = (dx1 * w1).mean() + (dy1 * w1).mean() + (dx2 * w2).mean() + (dy2 * w2).mean()
    return loss


# ============================================================
# Backward-compatible Alias
# ============================================================

CandidateGuidedAdaptiveContextConv = DirectionAwareAdaptiveContextConv

#
# def example_usage():
#     B, C, H, W = 2, 64, 64, 64
#     feat = torch.randn(B, C, H, W)
#     cand_map = torch.sigmoid(torch.randn(B, 1, H, W))
#     gate_map = torch.sigmoid(torch.randn(B, 1, H, W))
#
#     plugin = DirectionAwareAdaptiveContextConv(
#         in_ch=C,
#         out_ch=C,
#         hidden_ch=32,
#         branch_dilations=(1, 2, 3),
#         direction_dilations=(1, 2),
#         direction_kernel_size=5,
#         use_base_conv=True,
#         use_residual=True,
#         residual_alpha=1.0,
#         soft_stage_scale=0.5,
#         gate_stage_scale=1.0,
#         warmup_scale=0.0,
#         detach_candidate_in_gating=True,
#         noise_suppress_alpha=0.5,
#         return_intermediate_features=True,
#     )
#
#     out, aux = plugin(feat, cand_map=cand_map, gate_map=gate_map, stage_mode="gate", return_aux=True)
#     print("____out:", out.shape)
#     print("scale_weights:", aux["scale_weights"].shape)
#     print("direction_weights:", aux["direction_weights"].shape)
#     print("direction_mix:", aux["direction_mix"].shape)
#     print("noise_suppress_map:", aux["noise_suppress_map"].shape)
#
#
# if __name__ == "__main__":
#     example_usage()
