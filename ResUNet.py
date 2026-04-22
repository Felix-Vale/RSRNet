from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
from Candidate_boundary_area import CandidateBoundaryRegionPluginV2
from fusion_refine import BoundaryGuidedFusionRefinementPlugin
from gated import GatedSparseRelationPlugin
from attention import UncertaintyGuidedHybridGeometricPrototypeAttention
from adaptive_conv import CandidateGuidedAdaptiveContextConv



class ConvBNAct(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        k: int = 3,
        s: int = 1,
        p: Optional[int] = None,
        act: bool = True,
    ):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class ResidualConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNAct(in_ch, out_ch, 3),
            ConvBNAct(out_ch, out_ch, 3, act=False),
        )
        if in_ch == out_ch:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.block(x) + self.shortcut(x))


class UpSampleFuseBlock(nn.Module):
    def __init__(self, in_low_ch: int, in_skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBNAct(in_low_ch, out_ch, k=1, p=0),
        )
        self.fuse = ResidualConvBlock(out_ch + in_skip_ch, out_ch)

    def forward(self, x_low: torch.Tensor, x_skip: torch.Tensor) -> torch.Tensor:
        x_low = self.up(x_low)
        x = torch.cat([x_low, x_skip], dim=1)
        return self.fuse(x)


class DecoderReasoningStage(nn.Module):
    def __init__(
        self,
        in_ch: int,
        stage_name: str = "decoder_reasoning_stage",
        use_candidate_plugin: bool = True,
        use_attention_plugin: bool = True,
        use_gated_plugin: bool = True,
        use_adaptive_conv_plugin: bool = True,
        use_fusion_plugin: bool = True,
    ):
        super().__init__()
        self.in_ch = in_ch
        self.stage_name = stage_name

        self.use_candidate_plugin = use_candidate_plugin
        self.use_attention_plugin = use_attention_plugin
        self.use_gated_plugin = use_gated_plugin
        self.use_adaptive_conv_plugin = use_adaptive_conv_plugin
        self.use_fusion_plugin = use_fusion_plugin

        self.coarse_head = nn.Conv2d(in_ch, 1, kernel_size=1, bias=True)

        if self.use_candidate_plugin:
            self.candidate_plugin = CandidateBoundaryRegionPluginV2(
                feat_ch=in_ch,
                tf_ch=8,
                fuse_ch=32,
                ts_mode="dw_diff",
                prob_pool_scales=(1, 2, 4),
                ts_dilations=(1, 2, 3),
                tf_kernel_sizes=(3, 5, 7),
                use_direction_prior=True,
                direction_kernel_size=7,
                direction_start_epoch=8,
            )

        if self.use_attention_plugin:
            self.attention_plugin = UncertaintyGuidedHybridGeometricPrototypeAttention(
                in_ch=in_ch,
                project_dim=64,
                num_heads=4,
                enable_relation=True,
                selection_mode="topk_thresh_hybrid",
                score_threshold=0.5,
                topk_ratio=0.12,
                min_keep_tokens=32,
                max_keep_tokens=256,
                allow_dense_fallback=True,
                use_gate_for_selection=True,
                detach_score_in_selection=True,

                uncertainty_weight=0.35,
                cand_weight=0.20,
                gate_weight=0.15,
                topology_weight=0.20,
                branch_weight=0.10,

                uncertainty_bias_scale=1.0,
                topology_bias_scale=1.0,
                direction_kernel_size=7,

                branch_token_alpha=0.50,
                topology_token_alpha=0.35,
                uncertainty_token_alpha=0.15,

                soft_stage_scale=0.5,
                gate_stage_scale=1.0,
                warmup_stage_scale=0.0,
                use_gate_for_injection=True,

                use_residual=True,
                residual_alpha=1.0,
                return_intermediate_features=True,
            )

        if self.use_gated_plugin:
            self.gated_plugin = GatedSparseRelationPlugin(
                in_ch=in_ch,
                project_dim=64,
                num_heads=4,
                enable_relation=True,
                use_candidate_gate=True,
                selection_mode="topk_thresh_hybrid",
                topk_ratio=0.15,
                min_keep_tokens=32,
                max_keep_tokens=256,
                score_threshold=0.5,
                allow_dense_fallback=True,
                use_residual=True,
            )
        if self.use_adaptive_conv_plugin:
            self.adaptive_conv_plugin = CandidateGuidedAdaptiveContextConv(
                in_ch=in_ch,
                out_ch=in_ch,
                hidden_ch=max(in_ch // 2, 16),
                branch_dilations=(1, 2, 3),
                use_base_conv=True,
                use_residual=True,
                residual_alpha=1.0,
                soft_stage_scale=0.5,
                gate_stage_scale=1.0,
                warmup_scale=0.0,
                detach_candidate_in_gating=True,
                depthwise_separable=False,
                return_intermediate_features=True,
            )


        if self.use_fusion_plugin:
            self.fusion_plugin = BoundaryGuidedFusionRefinementPlugin(
                in_ch=in_ch,
                num_classes=1,
                hidden_ch=max(in_ch, 32),
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

    def forward(
        self,
        feat: torch.Tensor,
        stage_mode: str = "warmup",
        detach_pred: bool = True,
        tau: float = 0.4,
        temp: float = 1.0,
        return_aux: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        aux: Dict[str, torch.Tensor] = {}

        coarse_logit = self.coarse_head(feat)
        cand_map = None
        gate_map = None
        prob_map = None
        cand_aux = {}

        if self.use_candidate_plugin:
            cand_map, gate_map, cand_aux = self.candidate_plugin(
                feat=feat,
                logit=coarse_logit,
                stage_mode=stage_mode,
                detach_pred=detach_pred,
                tau=tau,
                temp=temp,
                return_aux=True,
            )
            prob_map = cand_aux.get("prob", None)

        feat_attn = feat
        attn_aux = {}

        if self.use_attention_plugin:
            feat_attn, attn_aux = self.attention_plugin(
                feat=feat,
                cand_map=cand_map,
                gate_map=gate_map,
                prob_map=prob_map,
                stage_mode=stage_mode,
                return_aux=True,
            )

        feat_rel = feat_attn
        gated_aux = {}

        if self.use_gated_plugin:
            feat_rel, gated_aux = self.gated_plugin(
                feat=feat_attn,
                cand_map=cand_map,
                gate_map=gate_map,
                stage_mode=stage_mode,
                return_aux=True,
            )

        feat_adc = feat
        adc_aux = {}

        if self.use_adaptive_conv_plugin:
            feat_adc, adc_aux = self.adaptive_conv_plugin(
                feat=feat,
                cand_map=cand_map,
                gate_map=gate_map,
                stage_mode=stage_mode,
                return_aux=True,
            )

        fusion_aux = {}
        refine_logit = None

        if self.use_fusion_plugin:
            feat_out, refine_logit, fusion_aux = self.fusion_plugin(
                feat_main=feat,
                feat_rel=feat_rel,
                feat_adc=feat_adc,
                cand_map=cand_map,
                gate_map=gate_map,
                stage_mode=stage_mode,
                return_aux=True,
            )
        else:
            if self.use_gated_plugin:
                feat_out = feat_rel
            elif self.use_adaptive_conv_plugin:
                feat_out = feat_adc
            elif self.use_attention_plugin:
                feat_out = feat_attn
            else:
                feat_out = feat

        if return_aux:
            aux["coarse_logit"] = coarse_logit
            if cand_map is not None:
                aux["cand_map"] = cand_map
            if gate_map is not None:
                aux["gate_map"] = gate_map
            if prob_map is not None:
                aux["prob_map"] = prob_map
            if refine_logit is not None:
                aux["refine_logit"] = refine_logit

            # 子模块结果
            aux["candidate_aux"] = cand_aux
            aux["attention_aux"] = attn_aux
            aux["gated_aux"] = gated_aux
            aux["adaptive_conv_aux"] = adc_aux
            aux["fusion_aux"] = fusion_aux

            # 关键特征
            aux["feat_in"] = feat
            aux["feat_attn"] = feat_attn
            aux["feat_rel"] = feat_rel
            aux["feat_adc"] = feat_adc
            aux["feat_out"] = feat_out

        return feat_out, aux


class ResUNetLikeBoundaryReasoningNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 1,
        base_ch: int = 32,
        use_reasoning_at_1_4: bool = True,
        use_reasoning_at_1_2: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.base_ch = base_ch

        self.use_reasoning_at_1_4 = use_reasoning_at_1_4
        self.use_reasoning_at_1_2 = use_reasoning_at_1_2

        # ------------------------------------------------------
        # Encoder
        # ------------------------------------------------------
        self.enc1 = ResidualConvBlock(in_channels, base_ch)           # 1x
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ResidualConvBlock(base_ch, base_ch * 2)           # 1/2
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ResidualConvBlock(base_ch * 2, base_ch * 4)       # 1/4
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = ResidualConvBlock(base_ch * 4, base_ch * 8)       # 1/8
        self.pool4 = nn.MaxPool2d(2)

        self.center = ResidualConvBlock(base_ch * 8, base_ch * 16)    # 1/16

        # ------------------------------------------------------
        # Decoder
        # ------------------------------------------------------
        self.up4 = UpSampleFuseBlock(base_ch * 16, base_ch * 8, base_ch * 8)   # 1/8
        self.up3 = UpSampleFuseBlock(base_ch * 8, base_ch * 4, base_ch * 4)    # 1/4
        self.up2 = UpSampleFuseBlock(base_ch * 4, base_ch * 2, base_ch * 2)    # 1/2
        self.up1 = UpSampleFuseBlock(base_ch * 2, base_ch, base_ch)             # 1x

        # ------------------------------------------------------
        # Reasoning stages
        # ------------------------------------------------------
        if self.use_reasoning_at_1_4:
            self.reasoning_1_4 = DecoderReasoningStage(
                in_ch=base_ch * 4,
                stage_name="reasoning_1_4",
                use_candidate_plugin=True,
                use_attention_plugin=True,
                use_gated_plugin=True,
                use_adaptive_conv_plugin=True,
                use_fusion_plugin=True,
            )

        if self.use_reasoning_at_1_2:
            self.reasoning_1_2 = DecoderReasoningStage(
                in_ch=base_ch * 2,
                stage_name="reasoning_1_2",
                use_candidate_plugin=True,
                use_attention_plugin=True,
                use_gated_plugin=True,
                use_adaptive_conv_plugin=True,
                use_fusion_plugin=True,
            )

        # ------------------------------------------------------
        # Final segmentation head
        # ------------------------------------------------------
        self.final_head = nn.Conv2d(base_ch, num_classes, kernel_size=1, bias=True)

    def forward(
        self,
        x: torch.Tensor,
        stage_mode: str = "warmup",
        detach_pred: bool = True,
        tau: float = 0.4,
        temp: float = 1.0,
        return_aux: bool = False,
    ):
        aux = {}

        # ------------------------------------------------------
        # Encoder
        # ------------------------------------------------------
        x1 = self.enc1(x)                     # 1x
        x2 = self.enc2(self.pool1(x1))       # 1/2
        x3 = self.enc3(self.pool2(x2))       # 1/4
        x4 = self.enc4(self.pool3(x3))       # 1/8
        x5 = self.center(self.pool4(x4))     # 1/16

        # ------------------------------------------------------
        # Decoder
        # ------------------------------------------------------
        d4 = self.up4(x5, x4)                # 1/8
        d3 = self.up3(d4, x3)                # 1/4

        # 1/4 分辨率增强
        if self.use_reasoning_at_1_4:
            d3_refined, aux_1_4 = self.reasoning_1_4(
                feat=d3,
                stage_mode=stage_mode,
                detach_pred=detach_pred,
                tau=tau,
                temp=temp,
                return_aux=True,
            )
        else:
            d3_refined = d3
            aux_1_4 = {}

        d2 = self.up2(d3_refined, x2)        # 1/2

        if self.use_reasoning_at_1_2:
            d2_refined, aux_1_2 = self.reasoning_1_2(
                feat=d2,
                stage_mode=stage_mode,
                detach_pred=detach_pred,
                tau=tau,
                temp=temp,
                return_aux=True,
            )
        else:
            d2_refined = d2
            aux_1_2 = {}

        d1 = self.up1(d2_refined, x1)        # 1x
        final_logit = self.final_head(d1)

        if return_aux:
            aux["x1"] = x1
            aux["x2"] = x2
            aux["x3"] = x3
            aux["x4"] = x4
            aux["x5"] = x5

            aux["d4"] = d4
            aux["d3_raw"] = d3
            aux["d3_refined"] = d3_refined
            aux["d2_raw"] = d2
            aux["d2_refined"] = d2_refined
            aux["d1"] = d1
            aux["final_logit"] = final_logit

            aux["reasoning_1_4"] = aux_1_4
            aux["reasoning_1_2"] = aux_1_2

            return final_logit, aux

        return final_logit


def example_usage():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResUNetLikeBoundaryReasoningNet(
        in_channels=1,
        num_classes=1,
        base_ch=32,
        use_reasoning_at_1_4=True,
        use_reasoning_at_1_2=True,
    ).to(device)

    x = torch.randn(2, 1, 256, 256, device=device)

    model.eval()
    with torch.no_grad():
        logit, aux = model(
            x,
            stage_mode="soft",
            detach_pred=True,
            tau=0.30,
            temp=1.00,
            return_aux=True,
        )

    print("final_logit:", tuple(logit.shape))
    print("d3_refined :", tuple(aux["d3_refined"].shape))
    print("d2_refined :", tuple(aux["d2_refined"].shape))
    if "cand_map" in aux["reasoning_1_4"]:
        print("1/4 cand_map:", tuple(aux["reasoning_1_4"]["cand_map"].shape))
    if "cand_map" in aux["reasoning_1_2"]:
        print("1/2 cand_map:", tuple(aux["reasoning_1_2"]["cand_map"].shape))


if __name__ == "__main__":
    example_usage()
