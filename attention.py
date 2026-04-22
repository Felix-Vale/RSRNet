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
class UPPRAttentionStageConfig:
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
        raise ValueError(f"{name} must be a 4D tensor [B,C,H,W], got shape={tuple(x.shape)}")


def _flatten_hw(x: torch.Tensor) -> torch.Tensor:
    b, c, h, w = x.shape
    return x.flatten(2).transpose(1, 2).contiguous()


def _indices_to_xy(indices: torch.Tensor, w: int) -> Tuple[torch.Tensor, torch.Tensor]:
    y = torch.div(indices, w, rounding_mode='floor')
    x = indices % w
    return y, x


class StructuralUncertaintyScoreBuilder(nn.Module):
    def __init__(self, in_ch: int, hidden_ch: int = 16):
        super().__init__()
        self.feat_conflict_proj = nn.Sequential(
            nn.Conv2d(in_ch, hidden_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_ch),
            nn.ReLU(inplace=True),
        )
        self.fuse = nn.Sequential(
            ConvBNAct(hidden_ch + 3, hidden_ch, 3),
            nn.Conv2d(hidden_ch, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(
        self,
        feat: torch.Tensor,
        cand_map: Optional[torch.Tensor],
        gate_map: Optional[torch.Tensor],
        prob_map: Optional[torch.Tensor],
    ) -> torch.Tensor:
        b, _, h, w = feat.shape
        local_mean = F.avg_pool2d(feat, kernel_size=3, stride=1, padding=1)
        feat_conflict = torch.abs(feat - local_mean)
        feat_conflict = self.feat_conflict_proj(feat_conflict)

        if cand_map is None:
            cand_map = feat.new_zeros((b, 1, h, w))
        if gate_map is None:
            gate_map = feat.new_zeros((b, 1, h, w))
        if prob_map is None:
            trans = feat.new_zeros((b, 1, h, w))
        else:
            prob = prob_map.clamp(0.0, 1.0)
            trans = 4.0 * prob * (1.0 - prob)

        fuse_in = torch.cat([feat_conflict, cand_map, gate_map, trans], dim=1)
        return self.fuse(fuse_in)


class VesselTopologyBranchPrior(nn.Module):
    def __init__(
        self,
        in_ch: int,
        direction_kernel_size: int = 7,
        base_prob_weight: float = 0.65,
        base_cand_weight: float = 0.35,
    ):
        super().__init__()
        if direction_kernel_size not in (5, 7, 9):
            raise ValueError("direction_kernel_size must be 5, 7, or 9")
        self.direction_kernel_size = direction_kernel_size
        self.base_prob_weight = base_prob_weight
        self.base_cand_weight = base_cand_weight

        hidden = max(in_ch // 4, 16)
        self.feat_to_structure = nn.Sequential(
            nn.Conv2d(in_ch, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        self.fuse = nn.Sequential(
            ConvBNAct(4, hidden, 3),
            nn.Conv2d(hidden, 2, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        kernels, dirs = self._build_direction_kernels(direction_kernel_size)
        self.register_buffer("dir_kernels", kernels, persistent=False)   # [4,1,K,K]
        self.register_buffer("dir_vectors", dirs, persistent=False)      # [4,2]

    def _build_direction_kernels(self, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        kernels = []

        # 0 deg (horizontal)
        ker = torch.zeros((k, k), dtype=torch.float32)
        ker[k // 2, :] = 1.0
        kernels.append(ker)

        # 45 deg
        ker = torch.zeros((k, k), dtype=torch.float32)
        for i in range(k):
            ker[k - 1 - i, i] = 1.0
        kernels.append(ker)

        # 90 deg (vertical)
        ker = torch.zeros((k, k), dtype=torch.float32)
        ker[:, k // 2] = 1.0
        kernels.append(ker)

        # 135 deg
        ker = torch.zeros((k, k), dtype=torch.float32)
        for i in range(k):
            ker[i, i] = 1.0
        kernels.append(ker)

        kernels = torch.stack(kernels, dim=0).unsqueeze(1)  # [4,1,K,K]
        kernels = kernels / kernels.sum(dim=(2, 3), keepdim=True).clamp_min(1e-6)

        dirs = torch.tensor([
            [1.0, 0.0],
            [0.7071, -0.7071],
            [0.0, 1.0],
            [0.7071, 0.7071],
        ], dtype=torch.float32)
        dirs = F.normalize(dirs, dim=1)
        return kernels, dirs

    def _directional_response(self, x: torch.Tensor) -> torch.Tensor:
        pad = self.direction_kernel_size // 2
        weight = self.dir_kernels.to(dtype=x.dtype, device=x.device)
        # [B,4,H,W]
        return F.conv2d(x, weight, bias=None, stride=1, padding=pad)

    def forward(
        self,
        feat: torch.Tensor,
        cand_map: Optional[torch.Tensor],
        prob_map: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        b, _, h, w = feat.shape
        feat_struct = self.feat_to_structure(feat)

        if prob_map is None and cand_map is None:
            base_map = feat_struct
        elif prob_map is None:
            base_map = 0.5 * cand_map + 0.5 * feat_struct
        elif cand_map is None:
            base_map = self.base_prob_weight * prob_map.clamp(0.0, 1.0) + (1.0 - self.base_prob_weight) * feat_struct
        else:
            base_map = (
                self.base_prob_weight * prob_map.clamp(0.0, 1.0)
                + self.base_cand_weight * cand_map.clamp(0.0, 1.0)
                + max(0.0, 1.0 - self.base_prob_weight - self.base_cand_weight) * feat_struct
            )

        # [B,4,H,W]
        dir_resp = self._directional_response(base_map)
        dir_prob = torch.softmax(dir_resp, dim=1)

        dir_vec = self.dir_vectors.to(dtype=feat.dtype, device=feat.device)                     # [4,2]
        direction_field = torch.einsum('bdhw,dc->bchw', dir_prob, dir_vec)                      # [B,2,H,W]
        direction_field = F.normalize(direction_field, dim=1, eps=1e-6)

        strongest = dir_resp.max(dim=1, keepdim=True).values                                    # [B,1,H,W]
        second = torch.topk(dir_resp, k=2, dim=1).values[:, 1:2]                                # [B,1,H,W]
        dominant_gap = (strongest - second).clamp_min(0.0)
        local_density = F.avg_pool2d(base_map, kernel_size=3, stride=1, padding=1)

        topology_in = torch.cat([base_map, strongest, dominant_gap, local_density], dim=1)
        topo_branch = self.fuse(topology_in)
        topology_map = topo_branch[:, 0:1]

        branch_raw = 0.45 * strongest + 0.35 * dominant_gap + 0.20 * (1.0 - local_density).clamp(0.0, 1.0)
        branch_map = torch.sigmoid(4.0 * (branch_raw - 0.35))
        branch_map = torch.max(branch_map, topo_branch[:, 1:2])

        return topology_map, branch_map, direction_field, base_map


class TopologyAwareTokenSelector(nn.Module):
    def __init__(
        self,
        selection_mode: str = "topk_thresh_hybrid",
        score_threshold: float = 0.5,
        topk_ratio: float = 0.15,
        min_keep_tokens: int = 32,
        max_keep_tokens: int = 256,
        use_gate_for_selection: bool = True,
        detach_score_in_selection: bool = True,
        allow_dense_fallback: bool = True,
        dense_fallback_ratio: float = 0.25,
        uncertainty_weight: float = 0.35,
        cand_weight: float = 0.20,
        gate_weight: float = 0.15,
        topology_weight: float = 0.20,
        branch_weight: float = 0.10,
    ):
        super().__init__()
        self.selection_mode = selection_mode
        self.score_threshold = score_threshold
        self.topk_ratio = topk_ratio
        self.min_keep_tokens = min_keep_tokens
        self.max_keep_tokens = max_keep_tokens
        self.use_gate_for_selection = use_gate_for_selection
        self.detach_score_in_selection = detach_score_in_selection
        self.allow_dense_fallback = allow_dense_fallback
        self.dense_fallback_ratio = dense_fallback_ratio
        self.uncertainty_weight = uncertainty_weight
        self.cand_weight = cand_weight
        self.gate_weight = gate_weight
        self.topology_weight = topology_weight
        self.branch_weight = branch_weight

    def build_score_map(
        self,
        uncertainty_map: torch.Tensor,
        cand_map: Optional[torch.Tensor],
        gate_map: Optional[torch.Tensor],
        topology_map: Optional[torch.Tensor],
        branch_map: Optional[torch.Tensor],
    ) -> torch.Tensor:
        score = self.uncertainty_weight * uncertainty_map
        if cand_map is not None:
            score = score + self.cand_weight * cand_map
        if self.use_gate_for_selection and gate_map is not None:
            score = score + self.gate_weight * gate_map
        if topology_map is not None:
            score = score + self.topology_weight * topology_map
        if branch_map is not None:
            score = score + self.branch_weight * branch_map
        score = score.clamp(0.0, 1.0)
        if self.detach_score_in_selection:
            score = score.detach()
        return score

    def forward(
        self,
        feat: torch.Tensor,
        uncertainty_map: torch.Tensor,
        cand_map: Optional[torch.Tensor],
        gate_map: Optional[torch.Tensor],
        topology_map: Optional[torch.Tensor],
        branch_map: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        _check_4d("feat", feat)
        _check_4d("uncertainty_map", uncertainty_map)
        b, _, h, w = feat.shape
        n = h * w

        score_map = self.build_score_map(uncertainty_map, cand_map, gate_map, topology_map, branch_map)
        score_flat = score_map.flatten(1)

        selection_mask = torch.zeros((b, 1, h, w), device=feat.device, dtype=feat.dtype)
        indices_list: List[torch.Tensor] = []
        selected_count = []
        selected_ratio = []

        for i in range(b):
            s = score_flat[i]
            topk_by_ratio = int(round(float(self.topk_ratio) * n))
            k_max = min(max(self.min_keep_tokens, topk_by_ratio), self.max_keep_tokens, n)
            k_dense = min(max(int(round(float(self.dense_fallback_ratio) * n)), self.min_keep_tokens), n)

            if self.selection_mode == "topk":
                keep_idx = torch.topk(s, k=k_max, largest=True).indices
            elif self.selection_mode == "threshold":
                keep_idx = torch.nonzero(s >= self.score_threshold, as_tuple=False).squeeze(1)
                if keep_idx.numel() < self.min_keep_tokens:
                    keep_idx = torch.topk(s, k=min(self.min_keep_tokens, n), largest=True).indices
                elif keep_idx.numel() > self.max_keep_tokens:
                    keep_idx = torch.topk(s, k=self.max_keep_tokens, largest=True).indices
            elif self.selection_mode == "topk_thresh_hybrid":
                idx_thresh = torch.nonzero(s >= self.score_threshold, as_tuple=False).squeeze(1)
                if idx_thresh.numel() == 0:
                    keep_idx = torch.topk(s, k=min(self.min_keep_tokens, n), largest=True).indices
                elif idx_thresh.numel() < self.min_keep_tokens:
                    keep_idx = torch.topk(s, k=min(self.min_keep_tokens, n), largest=True).indices
                elif idx_thresh.numel() > self.max_keep_tokens:
                    keep_idx = torch.topk(s, k=self.max_keep_tokens, largest=True).indices
                else:
                    keep_idx = idx_thresh

                if keep_idx.numel() > k_max:
                    keep_idx = torch.topk(s, k=k_max, largest=True).indices
            else:
                raise ValueError(f"Unsupported selection_mode: {self.selection_mode}")

            if keep_idx.numel() < 2 and self.allow_dense_fallback:
                keep_idx = torch.topk(s, k=min(max(2, k_dense), n), largest=True).indices

            keep_idx = torch.unique(keep_idx, sorted=False)
            indices_list.append(keep_idx)

            mask_flat = torch.zeros(n, device=feat.device, dtype=feat.dtype)
            mask_flat[keep_idx] = 1.0
            selection_mask[i, 0] = mask_flat.view(h, w)

            selected_count.append(float(keep_idx.numel()))
            selected_ratio.append(float(keep_idx.numel()) / float(n))

        selected_count_t = feat.new_tensor(selected_count)
        selected_ratio_t = feat.new_tensor(selected_ratio)
        return selection_mask, indices_list, selected_count_t, selected_ratio_t, score_map


class ForegroundBackgroundPrototypeBuilder(nn.Module):
    def __init__(self, fg_threshold: float = 0.75, bg_threshold: float = 0.25, detach_masks: bool = True):
        super().__init__()
        self.fg_threshold = fg_threshold
        self.bg_threshold = bg_threshold
        self.detach_masks = detach_masks

    def _weighted_proto(self, feat: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        weight_sum = weight.sum(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        return (feat * weight).sum(dim=(2, 3), keepdim=True) / weight_sum

    def forward(
        self,
        feat: torch.Tensor,
        cand_map: Optional[torch.Tensor],
        prob_map: Optional[torch.Tensor],
        uncertainty_map: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        b, c, h, w = feat.shape
        if prob_map is not None:
            prob = prob_map.clamp(0.0, 1.0)
            fg_mask = (prob >= self.fg_threshold).float()
            bg_mask = (prob <= self.bg_threshold).float()
        else:
            if cand_map is None:
                cand_map = feat.new_zeros((b, 1, h, w))
            if uncertainty_map is None:
                uncertainty_map = feat.new_zeros((b, 1, h, w))
            stable_mask = (1.0 - cand_map).clamp(0.0, 1.0) * (1.0 - uncertainty_map).clamp(0.0, 1.0)
            fg_mask = stable_mask
            bg_mask = 1.0 - stable_mask

        if self.detach_masks:
            fg_mask = fg_mask.detach()
            bg_mask = bg_mask.detach()

        fg_proto = self._weighted_proto(feat, fg_mask).view(b, c)
        bg_proto = self._weighted_proto(feat, bg_mask).view(b, c)
        return fg_proto, bg_proto, fg_mask, bg_mask


class PrototypeAffinityGuidance(nn.Module):
    def __init__(self, feat_ch: int):
        super().__init__()
        self.norm = nn.LayerNorm(feat_ch)
        self.fuse = nn.Linear(feat_ch * 3, feat_ch, bias=True)

    def forward(self, token_feat: torch.Tensor, fg_proto: torch.Tensor, bg_proto: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if token_feat.numel() == 0:
            return token_feat, token_feat.new_zeros((0, 2))

        x = self.norm(token_feat)
        fg = F.normalize(fg_proto.unsqueeze(0), dim=1)
        bg = F.normalize(bg_proto.unsqueeze(0), dim=1)
        x_n = F.normalize(x, dim=1)

        aff_fg = torch.sum(x_n * fg, dim=1, keepdim=True)
        aff_bg = torch.sum(x_n * bg, dim=1, keepdim=True)
        proto_affinity = torch.cat([aff_fg, aff_bg], dim=1)
        proto_weight = torch.softmax(proto_affinity, dim=1)

        proto_mix = proto_weight[:, :1] * fg + proto_weight[:, 1:] * bg
        guided_feat = self.fuse(torch.cat([token_feat, proto_mix, token_feat - proto_mix], dim=1))
        return guided_feat, proto_affinity


class TopologyConstrainedSparseAttention(nn.Module):
    def __init__(
        self,
        in_ch: int,
        project_dim: int = 64,
        num_heads: int = 4,
        uncertainty_bias_scale: float = 1.0,
        topology_bias_scale: float = 1.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        if project_dim % num_heads != 0:
            raise ValueError("project_dim must be divisible by num_heads")
        self.in_ch = in_ch
        self.project_dim = project_dim
        self.num_heads = num_heads
        self.head_dim = project_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.uncertainty_bias_scale = uncertainty_bias_scale
        self.topology_bias_scale = topology_bias_scale

        self.pre_norm = nn.LayerNorm(in_ch)
        self.q_proj = nn.Linear(in_ch, project_dim, bias=True)
        self.k_proj = nn.Linear(in_ch, project_dim, bias=True)
        self.v_proj = nn.Linear(in_ch, project_dim, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.out_proj = nn.Linear(project_dim, in_ch, bias=True)
        self.out_drop = nn.Dropout(proj_drop)

    def build_topology_bias(
        self,
        token_indices: torch.Tensor,
        topology_score: torch.Tensor,
        direction_vec: torch.Tensor,
        h: int,
        w: int,
    ) -> torch.Tensor:
        # token_indices: [K]
        # topology_score: [N]
        # direction_vec: [N,2]
        y, x = _indices_to_xy(token_indices, w)
        dy = y[:, None].float() - y[None, :].float()
        dx = x[:, None].float() - x[None, :].float()
        dist = torch.sqrt(dx ** 2 + dy ** 2 + 1e-6)
        dir_to_pair = torch.stack([dx, dy], dim=-1)
        dir_to_pair = F.normalize(dir_to_pair, dim=-1, eps=1e-6)  # [K,K,2]

        topo_sel = topology_score[token_indices]  # [K]
        vec_sel = direction_vec[token_indices]    # [K,2]
        vec_sel = F.normalize(vec_sel, dim=1, eps=1e-6)

        align_i = torch.abs((vec_sel[:, None, :] * dir_to_pair).sum(dim=-1))
        align_j = torch.abs((vec_sel[None, :, :] * dir_to_pair).sum(dim=-1))
        align = 0.5 * (align_i + align_j)

        dist_bias = torch.exp(-dist / max(float(min(h, w)) * 0.12, 1.0))
        topo_pair = 0.5 * (topo_sel[:, None] + topo_sel[None, :])
        bias = topo_pair * align * dist_bias
        return bias

    def forward(
        self,
        tokens: torch.Tensor,
        uncertainty_score: torch.Tensor,
        token_indices: torch.Tensor,
        topology_score_all: torch.Tensor,
        direction_vec_all: torch.Tensor,
        h: int,
        w: int,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        if tokens.dim() != 2:
            raise ValueError(f"tokens must be [K,C], got shape={tuple(tokens.shape)}")
        k_num, _ = tokens.shape
        if k_num == 0:
            return tokens, None, None

        x = self.pre_norm(tokens)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(k_num, self.num_heads, self.head_dim).transpose(0, 1)
        k = k.view(k_num, self.num_heads, self.head_dim).transpose(0, 1)
        v = v.view(k_num, self.num_heads, self.head_dim).transpose(0, 1)

        sim = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        u_pair = 0.5 * (uncertainty_score[:, None] + uncertainty_score[None, :])
        topo_bias = self.build_topology_bias(token_indices, topology_score_all, direction_vec_all, h, w)

        attn = sim + self.uncertainty_bias_scale * u_pair.unsqueeze(0) + self.topology_bias_scale * topo_bias.unsqueeze(0)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(0, 1).contiguous().view(k_num, self.project_dim)
        out = self.out_drop(self.out_proj(out))
        out = out * uncertainty_score.view(k_num, 1).clamp(0.0, 1.0)
        attn_map_mean = attn.mean(dim=0)
        return out, attn_map_mean, topo_bias


class SoftTopologyBranchRefiner(nn.Module):
    def __init__(self, in_ch: int):
        super().__init__()
        hidden = max(in_ch // 2, 16)
        self.local_refine = nn.Sequential(
            ConvBNAct(in_ch, hidden, 3),
            ConvBNAct(hidden, in_ch, 3, act=False),
        )
        self.proto_gate = nn.Sequential(
            nn.Conv2d(in_ch, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(
        self,
        feat: torch.Tensor,
        weight_map: torch.Tensor,
        fg_proto: torch.Tensor,
        bg_proto: torch.Tensor,
        topology_map: torch.Tensor,
        branch_map: torch.Tensor,
    ) -> torch.Tensor:
        b, c, h, w = feat.shape
        local_delta = self.local_refine(feat)
        fg_map = fg_proto.view(b, c, 1, 1).expand_as(feat)
        bg_map = bg_proto.view(b, c, 1, 1).expand_as(feat)
        proto_ref = self.proto_gate(torch.abs(feat - fg_map) + torch.abs(feat - bg_map))
        topo_branch_weight = (0.6 * topology_map + 0.4 * branch_map).clamp(0.0, 1.0)
        return local_delta * weight_map * proto_ref * topo_branch_weight


class UncertaintyGuidedTopologyBranchAttention(nn.Module):
    def __init__(
        self,
        in_ch: int,
        project_dim: int = 64,
        num_heads: int = 4,
        enable_relation: bool = True,
        selection_mode: str = "topk_thresh_hybrid",
        score_threshold: float = 0.5,
        topk_ratio: float = 0.15,
        min_keep_tokens: int = 32,
        max_keep_tokens: int = 256,
        allow_dense_fallback: bool = True,
        use_gate_for_selection: bool = True,
        detach_score_in_selection: bool = True,
        uncertainty_weight: float = 0.35,
        cand_weight: float = 0.20,
        gate_weight: float = 0.15,
        topology_weight: float = 0.20,
        branch_weight: float = 0.10,
        uncertainty_bias_scale: float = 1.0,
        topology_bias_scale: float = 1.0,
        branch_token_alpha: float = 0.50,
        topology_token_alpha: float = 0.35,
        uncertainty_token_alpha: float = 0.15,
        soft_stage_scale: float = 0.5,
        gate_stage_scale: float = 1.0,
        warmup_stage_scale: float = 0.0,
        use_gate_for_injection: bool = True,
        use_residual: bool = True,
        residual_alpha: float = 1.0,
        direction_kernel_size: int = 7,
        return_intermediate_features: bool = True,
    ):
        super().__init__()
        self.in_ch = in_ch
        self.enable_relation = enable_relation
        self.soft_stage_scale = soft_stage_scale
        self.gate_stage_scale = gate_stage_scale
        self.warmup_stage_scale = warmup_stage_scale
        self.use_gate_for_injection = use_gate_for_injection
        self.use_residual = use_residual
        self.residual_alpha = residual_alpha
        self.return_intermediate_features = return_intermediate_features
        self.branch_token_alpha = branch_token_alpha
        self.topology_token_alpha = topology_token_alpha
        self.uncertainty_token_alpha = uncertainty_token_alpha

        self.uncertainty_builder = StructuralUncertaintyScoreBuilder(in_ch=in_ch, hidden_ch=max(in_ch // 4, 16))
        self.topology_prior = VesselTopologyBranchPrior(
            in_ch=in_ch,
            direction_kernel_size=direction_kernel_size,
        )
        self.token_selector = TopologyAwareTokenSelector(
            selection_mode=selection_mode,
            score_threshold=score_threshold,
            topk_ratio=topk_ratio,
            min_keep_tokens=min_keep_tokens,
            max_keep_tokens=max_keep_tokens,
            use_gate_for_selection=use_gate_for_selection,
            detach_score_in_selection=detach_score_in_selection,
            allow_dense_fallback=allow_dense_fallback,
            uncertainty_weight=uncertainty_weight,
            cand_weight=cand_weight,
            gate_weight=gate_weight,
            topology_weight=topology_weight,
            branch_weight=branch_weight,
        )
        self.prototype_builder = ForegroundBackgroundPrototypeBuilder()
        self.prototype_guidance = PrototypeAffinityGuidance(feat_ch=in_ch)
        self.sparse_attention = TopologyConstrainedSparseAttention(
            in_ch=in_ch,
            project_dim=project_dim,
            num_heads=num_heads,
            uncertainty_bias_scale=uncertainty_bias_scale,
            topology_bias_scale=topology_bias_scale,
        )
        self.soft_refiner = SoftTopologyBranchRefiner(in_ch=in_ch)
        self.out_fuse = nn.Sequential(
            ConvBNAct(in_ch, in_ch, 3),
            ConvBNAct(in_ch, in_ch, 3, act=False),
        )

    def _build_injection_map(
        self,
        feat: torch.Tensor,
        cand_map: Optional[torch.Tensor],
        gate_map: Optional[torch.Tensor],
        uncertainty_map: torch.Tensor,
        topology_map: torch.Tensor,
        branch_map: torch.Tensor,
        stage_mode: str,
    ) -> torch.Tensor:
        b, _, h, w = feat.shape
        topo_branch = (0.6 * topology_map + 0.4 * branch_map).clamp(0.0, 1.0)
        if stage_mode == "warmup":
            return feat.new_full((b, 1, h, w), float(self.warmup_stage_scale))
        if stage_mode == "soft":
            base = cand_map.clamp(0.0, 1.0) if cand_map is not None else uncertainty_map.clamp(0.0, 1.0)
            return (0.6 * base + 0.4 * topo_branch).clamp(0.0, 1.0) * float(self.soft_stage_scale)
        if stage_mode == "gate":
            if gate_map is not None:
                base = gate_map.clamp(0.0, 1.0)
            elif cand_map is not None:
                base = cand_map.clamp(0.0, 1.0)
            else:
                base = uncertainty_map.clamp(0.0, 1.0)
            return (0.5 * base + 0.5 * topo_branch).clamp(0.0, 1.0) * float(self.gate_stage_scale)
        raise ValueError(f"Unsupported stage_mode: {stage_mode}")

    def forward(
        self,
        feat: torch.Tensor,
        cand_map: Optional[torch.Tensor] = None,
        gate_map: Optional[torch.Tensor] = None,
        prob_map: Optional[torch.Tensor] = None,
        stage_mode: str = "warmup",
        return_aux: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        _check_4d("feat", feat)
        _check_4d("cand_map", cand_map)
        _check_4d("gate_map", gate_map)
        _check_4d("prob_map", prob_map)

        b, c, h, w = feat.shape
        if cand_map is not None and (cand_map.shape[0] != b or cand_map.shape[-2:] != (h, w) or cand_map.shape[1] != 1):
            raise ValueError("cand_map must be [B,1,H,W] and spatially aligned with feat")
        if gate_map is not None and (gate_map.shape[0] != b or gate_map.shape[-2:] != (h, w) or gate_map.shape[1] != 1):
            raise ValueError("gate_map must be [B,1,H,W] and spatially aligned with feat")
        if prob_map is not None and (prob_map.shape[0] != b or prob_map.shape[-2:] != (h, w) or prob_map.shape[1] != 1):
            raise ValueError("prob_map must be [B,1,H,W] and spatially aligned with feat")

        uncertainty_map = self.uncertainty_builder(feat, cand_map, gate_map, prob_map)
        topology_map, branch_map, direction_field, topology_base_map = self.topology_prior(feat, cand_map, prob_map)
        fg_proto, bg_proto, fg_mask, bg_mask = self.prototype_builder(
            feat=feat,
            cand_map=cand_map,
            prob_map=prob_map,
            uncertainty_map=uncertainty_map,
        )

        injection_map = self._build_injection_map(
            feat, cand_map, gate_map, uncertainty_map, topology_map, branch_map, stage_mode
        )

        if (not self.enable_relation) or stage_mode in ("warmup", "soft"):
            soft_delta = self.soft_refiner(feat, injection_map, fg_proto, bg_proto, topology_map, branch_map)
            update_map = self.out_fuse(soft_delta)
            refined_feat = feat + self.residual_alpha * update_map if self.use_residual else update_map
            if return_aux:
                aux = {
                    "selection_mask": feat.new_zeros((b, 1, h, w)),
                    "selected_ratio": feat.new_zeros((b,)),
                    "selected_count": feat.new_zeros((b,)),
                    "relation_enable_flag": feat.new_zeros((b,)),
                    "uncertainty_map": uncertainty_map,
                    "score_map": uncertainty_map,
                    "topology_map": topology_map,
                    "branch_map": branch_map,
                    "direction_field": direction_field,
                    "topology_base_map": topology_base_map,
                    "prototype_affinity_mean": feat.new_zeros((b, 2)),
                    "update_map": update_map,
                    "fg_mask": fg_mask,
                    "bg_mask": bg_mask,
                    "injection_map": injection_map,
                }
                return refined_feat, aux
            return refined_feat, {}

        selection_mask, indices_list, selected_count, selected_ratio, score_map = self.token_selector(
            feat=feat,
            uncertainty_map=uncertainty_map,
            cand_map=cand_map,
            gate_map=gate_map,
            topology_map=topology_map,
            branch_map=branch_map,
        )

        feat_flat = _flatten_hw(feat)
        uncertainty_flat = uncertainty_map.flatten(1)
        topology_flat = topology_map.flatten(1)
        branch_flat = branch_map.flatten(1)
        direction_flat = direction_field.flatten(2).transpose(1, 2).contiguous()  # [B,N,2]

        sparse_delta = feat.new_zeros((b, c, h, w))
        topo_bias_mean = []
        proto_affinity_mean = []
        relation_enable_flag = []
        branch_token_mean = []

        for i in range(b):
            idx = indices_list[i]
            if idx.numel() < 2:
                topo_bias_mean.append(0.0)
                proto_affinity_mean.append([0.0, 0.0])
                relation_enable_flag.append(0.0)
                branch_token_mean.append(0.0)
                continue

            tokens_i = feat_flat[i, idx]                  # [K,C]
            unc_i = uncertainty_flat[i]                   # [N]
            topo_i = topology_flat[i]                     # [N]
            branch_i = branch_flat[i]                     # [N]
            dir_i = direction_flat[i]                     # [N,2]

            attn_out_i, _, topo_bias_i = self.sparse_attention(
                tokens=tokens_i,
                uncertainty_score=unc_i[idx],
                token_indices=idx,
                topology_score_all=topo_i,
                direction_vec_all=dir_i,
                h=h,
                w=w,
            )

            proto_guided_i, proto_aff_i = self.prototype_guidance(
                token_feat=attn_out_i,
                fg_proto=fg_proto[i],
                bg_proto=bg_proto[i],
            )

            token_weight = (
                self.branch_token_alpha * branch_i[idx]
                + self.topology_token_alpha * topo_i[idx]
                + self.uncertainty_token_alpha * unc_i[idx]
            ).clamp(0.0, 1.0).unsqueeze(1)
            proto_guided_i = proto_guided_i * token_weight

            scatter_map = sparse_delta[i].flatten(1).transpose(0, 1).contiguous()
            scatter_map[idx] = proto_guided_i
            scatter_map = scatter_map.transpose(0, 1).reshape(c, h, w).contiguous()
            sparse_delta[i] = scatter_map

            topo_bias_mean.append(float(topo_bias_i.mean().item()) if topo_bias_i is not None else 0.0)
            proto_affinity_mean.append([
                float(proto_aff_i[:, 0].mean().item()),
                float(proto_aff_i[:, 1].mean().item()),
            ])
            relation_enable_flag.append(1.0)
            branch_token_mean.append(float(token_weight.mean().item()))

        if self.use_gate_for_injection:
            sparse_delta = sparse_delta * injection_map

        update_map = self.out_fuse(sparse_delta)
        refined_feat = feat + self.residual_alpha * update_map if self.use_residual else update_map

        if return_aux:
            aux = {
                "selection_mask": selection_mask,
                "selected_ratio": selected_ratio,
                "selected_count": selected_count,
                "relation_enable_flag": feat.new_tensor(relation_enable_flag),
                "uncertainty_map": uncertainty_map,
                "score_map": score_map,
                "topology_map": topology_map,
                "branch_map": branch_map,
                "direction_field": direction_field,
                "topology_base_map": topology_base_map,
                "prototype_affinity_mean": feat.new_tensor(proto_affinity_mean),
                "topology_bias_mean": feat.new_tensor(topo_bias_mean),
                "branch_token_mean": feat.new_tensor(branch_token_mean),
                "update_map": update_map,
                "fg_mask": fg_mask,
                "bg_mask": bg_mask,
                "injection_map": injection_map,
            }
            if self.return_intermediate_features:
                aux["sparse_delta"] = sparse_delta
                aux["refined_feat"] = refined_feat
            return refined_feat, aux
        return refined_feat, {}


UncertaintyGuidedHybridGeometricPrototypeAttention = UncertaintyGuidedTopologyBranchAttention


# ============================================================
# Optional Loss Utilities
# ============================================================


def relation_update_sparsity_loss(update_map: torch.Tensor, target_ratio: float = 0.20) -> torch.Tensor:
    return torch.abs(update_map.abs().mean() - update_map.new_tensor(target_ratio))


def vessel_topology_consistency_loss(
    update_map: torch.Tensor,
    topology_map: torch.Tensor,
    branch_map: torch.Tensor,
) -> torch.Tensor:
    update_energy = update_map.abs().mean(dim=1, keepdim=True)
    target = (0.7 * topology_map + 0.3 * branch_map).clamp(0.0, 1.0)
    return F.l1_loss(update_energy, target)


# if __name__ == "__main__":
#     torch.manual_seed(7)
#     B, C, H, W = 2, 64, 64, 64
#     feat = torch.randn(B, C, H, W)
#     cand_map = torch.sigmoid(torch.randn(B, 1, H, W))
#     gate_map = torch.sigmoid((cand_map - 0.45) / 0.3)
#     prob_map = torch.sigmoid(torch.randn(B, 1, H, W))
#
#     plugin = UncertaintyGuidedTopologyBranchAttention(
#         in_ch=C,
#         project_dim=64,
#         num_heads=4,
#         enable_relation=True,
#         selection_mode="topk_thresh_hybrid",
#         score_threshold=0.5,
#         topk_ratio=0.12,
#         min_keep_tokens=32,
#         max_keep_tokens=256,
#         allow_dense_fallback=True,
#         use_gate_for_selection=True,
#         detach_score_in_selection=True,
#         direction_kernel_size=7,
#         return_intermediate_features=True,
#     )
#
#     out_gate, aux_gate = plugin(
#         feat=feat,
#         cand_map=cand_map,
#         gate_map=gate_map,
#         prob_map=prob_map,
#         stage_mode="gate",
#         return_aux=True,
#     )
#     print("gate ____out:", out_gate.shape)
#     print("selected ratio:", aux_gate["selected_ratio"])
#     print("topology bias mean:", aux_gate["topology_bias_mean"])
#     print("branch token mean:", aux_gate["branch_token_mean"])
