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
class SparseRelationStageConfig:
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


def _safe_ratio(numer: torch.Tensor, denom: torch.Tensor) -> torch.Tensor:
    return numer / denom.clamp_min(1e-6)


def _flatten_hw(x: torch.Tensor) -> torch.Tensor:
    b, c, h, w = x.shape
    return x.flatten(2).transpose(1, 2).contiguous()


def _unflatten_hw(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    b, n, c = x.shape
    if n != h * w:
        raise ValueError(f"Token number mismatch: N={n}, expected H*W={h*w}")
    return x.transpose(1, 2).reshape(b, c, h, w).contiguous()


# ============================================================
# Candidate-driven Token Selector
# ============================================================


class CandidateTokenSelector(nn.Module):
    def __init__(
        self,
        use_gate_for_selection: bool = True,
        selection_mode: str = "topk_thresh_hybrid",
        score_threshold: float = 0.5,
        topk_ratio: float = 0.15,
        min_keep_tokens: int = 32,
        max_keep_tokens: int = 256,
        detach_gate_in_selection: bool = True,
        allow_dense_fallback: bool = True,
        dense_fallback_ratio: float = 0.25,
    ):
        super().__init__()
        self.use_gate_for_selection = use_gate_for_selection
        self.selection_mode = selection_mode
        self.score_threshold = score_threshold
        self.topk_ratio = topk_ratio
        self.min_keep_tokens = min_keep_tokens
        self.max_keep_tokens = max_keep_tokens
        self.detach_gate_in_selection = detach_gate_in_selection
        self.allow_dense_fallback = allow_dense_fallback
        self.dense_fallback_ratio = dense_fallback_ratio

    def _resolve_score_map(
        self,
        cand_map: Optional[torch.Tensor],
        gate_map: Optional[torch.Tensor],
    ) -> torch.Tensor:

        src = None
        if self.use_gate_for_selection and gate_map is not None:
            src = gate_map
        elif cand_map is not None:
            src = cand_map
        elif gate_map is not None:
            src = gate_map
        else:
            raise ValueError("At least one of cand_map or gate_map must be provided for token selection.")

        if self.detach_gate_in_selection:
            src = src.detach()
        return src

    def forward(
        self,
        feat: torch.Tensor,
        cand_map: Optional[torch.Tensor],
        gate_map: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        _check_4d("feat", feat)
        b, _, h, w = feat.shape
        n = h * w

        score_map = self._resolve_score_map(cand_map, gate_map)
        _check_4d("score_map", score_map)

        if score_map.shape[0] != b or score_map.shape[-2:] != (h, w):
            raise ValueError(
                f"score_map spatial size must match feat. feat={tuple(feat.shape)}, score_map={tuple(score_map.shape)}"
            )
        if score_map.shape[1] != 1:
            raise ValueError(f"score_map channel must be 1, got {score_map.shape[1]}")

        score_flat = score_map.flatten(1)  # [B,N]
        selection_mask = torch.zeros((b, 1, h, w), device=feat.device, dtype=feat.dtype)
        indices_list: List[torch.Tensor] = []
        selected_count = []
        selected_ratio = []

        for i in range(b):
            s = score_flat[i]  # [N]
            topk_by_ratio = int(round(n * float(self.topk_ratio)))
            k_max = min(max(self.min_keep_tokens, topk_by_ratio), self.max_keep_tokens, n)
            k_dense = min(max(int(round(n * float(self.dense_fallback_ratio))), self.min_keep_tokens), n)

            if self.selection_mode == "topk":
                keep_idx = torch.topk(s, k=k_max, dim=0, largest=True).indices

            elif self.selection_mode == "threshold":
                keep_idx = torch.nonzero(s >= self.score_threshold, as_tuple=False).squeeze(1)
                if keep_idx.numel() < self.min_keep_tokens:
                    keep_idx = torch.topk(s, k=min(self.min_keep_tokens, n), dim=0, largest=True).indices
                elif keep_idx.numel() > self.max_keep_tokens:
                    vals, idx_top = torch.topk(s, k=self.max_keep_tokens, dim=0, largest=True)
                    keep_idx = idx_top

            elif self.selection_mode == "topk_thresh_hybrid":
                idx_thresh = torch.nonzero(s >= self.score_threshold, as_tuple=False).squeeze(1)

                if idx_thresh.numel() == 0:
                    keep_idx = torch.topk(s, k=min(self.min_keep_tokens, n), dim=0, largest=True).indices
                elif idx_thresh.numel() < self.min_keep_tokens:
                    keep_idx = torch.topk(s, k=min(self.min_keep_tokens, n), dim=0, largest=True).indices
                elif idx_thresh.numel() > self.max_keep_tokens:
                    keep_idx = torch.topk(s, k=self.max_keep_tokens, dim=0, largest=True).indices
                else:
                    keep_idx = idx_thresh

                if keep_idx.numel() > k_max:
                    keep_idx = torch.topk(s, k=k_max, dim=0, largest=True).indices

            else:
                raise ValueError(f"Unsupported selection_mode: {self.selection_mode}")

            if keep_idx.numel() < 2 and self.allow_dense_fallback:
                keep_idx = torch.topk(s, k=min(max(2, k_dense), n), dim=0, largest=True).indices

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





class SparseTokenSelfAttention(nn.Module):
    def __init__(
        self,
        in_ch: int,
        project_dim: int = 64,
        num_heads: int = 4,
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

        self.q_proj = nn.Linear(in_ch, project_dim, bias=True)
        self.k_proj = nn.Linear(in_ch, project_dim, bias=True)
        self.v_proj = nn.Linear(in_ch, project_dim, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.out_proj = nn.Linear(project_dim, in_ch, bias=True)
        self.out_drop = nn.Dropout(proj_drop)

        self.pre_norm = nn.LayerNorm(in_ch)

    def forward(
        self,
        tokens: torch.Tensor,
        token_score: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if tokens.dim() != 2:
            raise ValueError(f"tokens must be [K,C], got shape={tuple(tokens.shape)}")
        k_num, c = tokens.shape
        if k_num == 0:
            return tokens, None

        x = self.pre_norm(tokens)
        q = self.q_proj(x)  # [K,D]
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(k_num, self.num_heads, self.head_dim).transpose(0, 1)  # [H,K,Dh]
        k = k.view(k_num, self.num_heads, self.head_dim).transpose(0, 1)
        v = v.view(k_num, self.num_heads, self.head_dim).transpose(0, 1)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [H,K,K]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, v)  # [H,K,Dh]
        out = out.transpose(0, 1).contiguous().view(k_num, self.project_dim)  # [K,D]
        out = self.out_drop(self.out_proj(out))  # [K,C]
        if token_score is not None:
            token_score = token_score.view(k_num, 1).clamp(0.0, 1.0)
            out = out * token_score

        attn_map_mean = attn.mean(dim=0) if k_num > 0 else None
        return out, attn_map_mean


class SoftContextRefiner(nn.Module):
    def __init__(self, in_ch: int, hidden_ch: Optional[int] = None):
        super().__init__()
        if hidden_ch is None:
            hidden_ch = max(in_ch // 2, 16)

        self.local_context = nn.Sequential(
            ConvBNAct(in_ch, hidden_ch, 3),
            ConvBNAct(hidden_ch, in_ch, 3, act=False),
        )

    def forward(self, feat: torch.Tensor, weight_map: torch.Tensor) -> torch.Tensor:
        local_delta = self.local_context(feat)
        return local_delta * weight_map


class GatedSparseRelationPlugin(nn.Module):
    def __init__(
        self,
        in_ch: int,
        project_dim: int = 64,
        num_heads: int = 4,
        enable_relation: bool = True,
        use_candidate_gate: bool = True,
        selection_mode: str = "topk_thresh_hybrid",
        score_threshold: float = 0.5,
        topk_ratio: float = 0.15,
        min_keep_tokens: int = 32,
        max_keep_tokens: int = 256,
        allow_dense_fallback: bool = True,
        dense_fallback_ratio: float = 0.25,
        detach_gate_in_selection: bool = True,
        use_residual: bool = True,
        residual_alpha: float = 1.0,
        use_soft_rescale: bool = True,
        soft_stage_scale: float = 0.5,
        use_gate_for_injection: bool = True,
        normalize_selected_update: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()

        self.enable_relation = enable_relation
        self.use_candidate_gate = use_candidate_gate
        self.use_residual = use_residual
        self.residual_alpha = residual_alpha
        self.use_soft_rescale = use_soft_rescale
        self.soft_stage_scale = soft_stage_scale
        self.use_gate_for_injection = use_gate_for_injection
        self.normalize_selected_update = normalize_selected_update

        self.selector = CandidateTokenSelector(
            use_gate_for_selection=use_candidate_gate,
            selection_mode=selection_mode,
            score_threshold=score_threshold,
            topk_ratio=topk_ratio,
            min_keep_tokens=min_keep_tokens,
            max_keep_tokens=max_keep_tokens,
            detach_gate_in_selection=detach_gate_in_selection,
            allow_dense_fallback=allow_dense_fallback,
            dense_fallback_ratio=dense_fallback_ratio,
        )

        self.relation = SparseTokenSelfAttention(
            in_ch=in_ch,
            project_dim=project_dim,
            num_heads=num_heads,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )

        self.soft_refiner = SoftContextRefiner(in_ch=in_ch)

        self.out_fuse = nn.Sequential(
            ConvBNAct(in_ch, in_ch, 3),
            nn.Conv2d(in_ch, in_ch, kernel_size=1, bias=True),
        )

    def _resolve_weight_map(
        self,
        cand_map: Optional[torch.Tensor],
        gate_map: Optional[torch.Tensor],
        feat: torch.Tensor,
    ) -> torch.Tensor:
        b, _, h, w = feat.shape
        if gate_map is not None:
            return gate_map
        if cand_map is not None:
            return cand_map
        return feat.new_ones((b, 1, h, w))

    def _validate_inputs(
        self,
        feat: torch.Tensor,
        cand_map: Optional[torch.Tensor],
        gate_map: Optional[torch.Tensor],
    ):
        _check_4d("feat", feat)
        b, _, h, w = feat.shape

        if cand_map is not None:
            _check_4d("cand_map", cand_map)
            if cand_map.shape[0] != b or cand_map.shape[1] != 1 or cand_map.shape[-2:] != (h, w):
                raise ValueError(
                    f"cand_map must be [B,1,H,W] and aligned with feat. "
                    f"feat={tuple(feat.shape)}, cand_map={tuple(cand_map.shape)}"
                )

        if gate_map is not None:
            _check_4d("gate_map", gate_map)
            if gate_map.shape[0] != b or gate_map.shape[1] != 1 or gate_map.shape[-2:] != (h, w):
                raise ValueError(
                    f"gate_map must be [B,1,H,W] and aligned with feat. "
                    f"feat={tuple(feat.shape)}, gate_map={tuple(gate_map.shape)}"
                )

    def _scatter_updates(
        self,
        feat: torch.Tensor,
        indices_list: List[torch.Tensor],
        updated_tokens_list: List[torch.Tensor],
    ) -> torch.Tensor:
        b, c, h, w = feat.shape
        n = h * w
        feat_tokens = _flatten_hw(feat)  # [B,N,C]
        delta_tokens = feat.new_zeros((b, n, c))

        for i in range(b):
            keep_idx = indices_list[i]
            if keep_idx.numel() == 0:
                continue
            delta_tokens[i, keep_idx] = updated_tokens_list[i]

        delta_map = _unflatten_hw(delta_tokens, h, w)
        return delta_map

    def forward(
        self,
        feat: torch.Tensor,
        cand_map: Optional[torch.Tensor],
        gate_map: Optional[torch.Tensor],
        stage_mode: str = "warmup",
        return_aux: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        self._validate_inputs(feat, cand_map, gate_map)
        b, c, h, w = feat.shape
        n = h * w

        aux: Dict[str, torch.Tensor] = {}
        relation_enable_flag = feat.new_zeros((b,))


        if (not self.enable_relation) or stage_mode == "warmup":
            refined_feat = feat
            if return_aux:
                selection_mask = feat.new_zeros((b, 1, h, w))
                aux = {
                    "selection_mask": selection_mask,
                    "selected_ratio": feat.new_zeros((b,)),
                    "selected_count": feat.new_zeros((b,)),
                    "relation_enable_flag": relation_enable_flag,
                    "score_map": self._resolve_weight_map(cand_map, gate_map, feat),
                    "update_map": feat.new_zeros((b, c, h, w)),
                }
            return refined_feat, aux

        weight_map = self._resolve_weight_map(cand_map, gate_map, feat).clamp(0.0, 1.0)

        if stage_mode == "soft":
            soft_delta = self.soft_refiner(feat, weight_map)

            if self.use_soft_rescale:
                soft_delta = soft_delta * float(self.soft_stage_scale)

            update_map = self.out_fuse(soft_delta)

            if self.use_residual:
                refined_feat = feat + self.residual_alpha * update_map
            else:
                refined_feat = update_map

            relation_enable_flag = feat.new_ones((b,))
            if return_aux:
                aux = {
                    "selection_mask": weight_map,
                    "selected_ratio": weight_map.flatten(1).mean(dim=1),
                    "selected_count": weight_map.flatten(1).sum(dim=1),
                    "relation_enable_flag": relation_enable_flag,
                    "score_map": weight_map,
                    "update_map": update_map,
                }
            return refined_feat, aux


        if stage_mode != "gate":
            raise ValueError(f"Unsupported stage_mode: {stage_mode}")

        selection_mask, indices_list, selected_count, selected_ratio, score_map = self.selector(
            feat=feat,
            cand_map=cand_map,
            gate_map=gate_map,
        )

        feat_tokens = _flatten_hw(feat)           # [B,N,C]
        score_flat = score_map.flatten(1)         # [B,N]

        updated_tokens_list: List[torch.Tensor] = []
        attn_summary_list = []
        valid_relation_flags = []

        for i in range(b):
            keep_idx = indices_list[i]

            if keep_idx.numel() < 2:
                updated_tokens_list.append(feat.new_zeros((keep_idx.numel(), c)))
                attn_summary_list.append(feat.new_zeros((1, 1)))
                valid_relation_flags.append(0.0)
                continue

            tokens_i = feat_tokens[i, keep_idx]  # [K,C]
            score_i = score_flat[i, keep_idx]    # [K]

            updated_i, attn_map_i = self.relation(tokens_i, token_score=score_i)
            updated_tokens_list.append(updated_i)

            if attn_map_i is None:
                attn_summary_list.append(feat.new_zeros((1, 1)))
            else:
                attn_summary_list.append(attn_map_i)

            valid_relation_flags.append(1.0)

        relation_enable_flag = feat.new_tensor(valid_relation_flags)

        sparse_delta = self._scatter_updates(feat, indices_list, updated_tokens_list)  # [B,C,H,W]

        if self.normalize_selected_update:
            ratio = selected_ratio.view(b, 1, 1, 1).clamp_min(1.0 / float(n))
            norm_scale = (0.1 / ratio).clamp(max=2.0)
            sparse_delta = sparse_delta * norm_scale

        if self.use_gate_for_injection:
            sparse_delta = sparse_delta * weight_map

        update_map = self.out_fuse(sparse_delta)

        if self.use_residual:
            refined_feat = feat + self.residual_alpha * update_map
        else:
            refined_feat = update_map

        if return_aux:
            attn_mean_scalar = []
            attn_token_num = []
            for m in attn_summary_list:
                attn_mean_scalar.append(float(m.mean().item()))
                attn_token_num.append(float(m.shape[-1]))

            aux = {
                "selection_mask": selection_mask,
                "selected_ratio": selected_ratio,
                "selected_count": selected_count,
                "relation_enable_flag": relation_enable_flag,
                "score_map": score_map,
                "update_map": update_map,
                "attn_mean_scalar": feat.new_tensor(attn_mean_scalar),
                "attn_token_num": feat.new_tensor(attn_token_num),
            }
        return refined_feat, aux


def example_usage():
    torch.manual_seed(7)

    B, C, H, W = 2, 64, 64, 64
    feat = torch.randn(B, C, H, W)
    cand_map = torch.sigmoid(torch.randn(B, 1, H, W))
    gate_map = torch.sigmoid((cand_map - 0.45) / 0.3)

    plugin = GatedSparseRelationPlugin(
        in_ch=C,
        project_dim=64,
        num_heads=4,
        enable_relation=True,
        use_candidate_gate=True,
        selection_mode="topk_thresh_hybrid",
        score_threshold=0.5,
        topk_ratio=0.12,
        min_keep_tokens=32,
        max_keep_tokens=256,
        allow_dense_fallback=True,
        dense_fallback_ratio=0.25,
        detach_gate_in_selection=True,
        use_residual=True,
        residual_alpha=1.0,
        use_soft_rescale=True,
        soft_stage_scale=0.5,
        use_gate_for_injection=True,
        normalize_selected_update=True,
    )

    refined_warmup, aux_warmup = plugin(
        feat=feat,
        cand_map=cand_map,
        gate_map=gate_map,
        stage_mode="warmup",
        return_aux=True,
    )
    print("warmup refined:", refined_warmup.shape, aux_warmup["selected_ratio"])

    refined_soft, aux_soft = plugin(
        feat=feat,
        cand_map=cand_map,
        gate_map=gate_map,
        stage_mode="soft",
        return_aux=True,
    )
    print("soft refined:", refined_soft.shape, aux_soft["selected_ratio"])

    refined_gate, aux_gate = plugin(
        feat=feat,
        cand_map=cand_map,
        gate_map=gate_map,
        stage_mode="gate",
        return_aux=True,
    )
    print("gate refined:", refined_gate.shape)
    print("selected_ratio:", aux_gate["selected_ratio"])
    print("selected_count:", aux_gate["selected_count"])


if __name__ == "__main__":
    example_usage()
