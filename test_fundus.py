import os
import csv
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from scipy import ndimage
from scipy.spatial import cKDTree
from ResUNet import ResUNetLikeBoundaryReasoningNet


VALID_IM_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
VALID_GT_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate fundus checkpoint exactly like training validate() for 3-channel masks."
    )
    parser.add_argument("--pt_path", type=str, default=r"", help="")
    parser.add_argument("--image_dir", type=str, default=r"", help="")
    parser.add_argument("--gt_dir", type=str, default=r"", help="")
    parser.add_argument("--img_w", type=int, default=320, help="")
    parser.add_argument("--img_h", type=int, default=320, help="")
    parser.add_argument("--batch_size", type=int, default=4, help="")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--surface_connectivity", type=int, default=1)
    parser.add_argument("--save_csv", type=str, default=None)
    return parser.parse_args()


class VesselDataset(Dataset):
    """
    严格对齐眼底训练代码：
    - 图像: RGB
    - mask: RGB
    - resize
    - 图像 normalize(mean=0.5,std=0.5)
    - mask -> [3,H,W], 数值范围 [0,1]
    """
    def __init__(self, image_dir, mask_dir, img_size=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.img_size = img_size

        self.image_files = [
            f for f in os.listdir(self.image_dir)
            if Path(f).suffix.lower() in VALID_IM_EXTS
        ]
        self.image_files.sort()

        if len(self.image_files) == 0:
            raise RuntimeError(f"No images found in {self.image_dir}")

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                              std=[0.5, 0.5, 0.5])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask not found for image {img_name}: {mask_path}")

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("RGB")

        if self.img_size is not None:
            img = img.resize(self.img_size, resample=Image.BILINEAR)
            mask = mask.resize(self.img_size, resample=Image.NEAREST)

        img_t = self.to_tensor(img)
        img_t = self.normalize(img_t)

        mask_np = np.array(mask, dtype=np.float32)
        mask_np = np.clip(mask_np / 255.0, 0.0, 1.0)
        mask_t = torch.from_numpy(mask_np).permute(2, 0, 1).float()

        return img_t, mask_t, img_name


def soft_erode(img: torch.Tensor) -> torch.Tensor:
    if img.dim() != 4:
        raise ValueError(f"soft_erode expects [B,C,H,W], got {tuple(img.shape)}")
    p1 = -F.max_pool2d(-img, kernel_size=(3, 1), stride=1, padding=(1, 0))
    p2 = -F.max_pool2d(-img, kernel_size=(1, 3), stride=1, padding=(0, 1))
    return torch.min(p1, p2)


def soft_dilate(img: torch.Tensor) -> torch.Tensor:
    return F.max_pool2d(img, kernel_size=3, stride=1, padding=1)


def soft_open(img: torch.Tensor) -> torch.Tensor:
    return soft_dilate(soft_erode(img))


def soft_skeletonize(img: torch.Tensor, iter_num: int = 10) -> torch.Tensor:
    img = img.clamp(0.0, 1.0)
    skel = F.relu(img - soft_open(img))
    for _ in range(iter_num):
        img = soft_erode(img)
        delta = F.relu(img - soft_open(img))
        skel = skel + F.relu(delta - skel * delta)
    return skel.clamp(0.0, 1.0)


def estimate_terminal_map_from_skeleton(skel: torch.Tensor, dilate_radius: int = 2) -> torch.Tensor:
    if skel.dim() != 4:
        raise ValueError(f"skel must be [B,C,H,W], got {tuple(skel.shape)}")

    device = skel.device
    dtype = skel.dtype
    channels = skel.shape[1]
    kernel = torch.ones((channels, 1, 3, 3), device=device, dtype=dtype)
    neigh_sum = F.conv2d(skel, kernel, padding=1, groups=channels) - skel
    terminal = skel * ((neigh_sum > 0.5) & (neigh_sum < 1.5)).float()

    if dilate_radius > 0:
        k = 2 * dilate_radius + 1
        terminal = F.max_pool2d(terminal, kernel_size=k, stride=1, padding=dilate_radius)

    return terminal.clamp(0.0, 1.0)


@torch.no_grad()
def dice_iou_from_logits(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6):
    if logits.shape[-2:] != targets.shape[-2:]:
        logits = F.interpolate(logits, size=targets.shape[-2:], mode="bilinear", align_corners=False)

    probs = torch.sigmoid(logits)
    pred = (probs >= threshold)
    tgt = (targets >= threshold)

    dims = (0, 2, 3)
    inter = (pred & tgt).float().sum(dim=dims)
    union = (pred | tgt).float().sum(dim=dims)
    pred_sum = pred.float().sum(dim=dims)
    tgt_sum = tgt.float().sum(dim=dims)

    dice = (2.0 * inter + eps) / (pred_sum + tgt_sum + eps)
    iou = (inter + eps) / (union + eps)
    return float(dice.mean().item()), float(iou.mean().item())


@torch.no_grad()
def cldice_from_logits(logits: torch.Tensor, targets: torch.Tensor,
                       threshold: float = 0.5, iter_num: int = 10, eps: float = 1e-6):
    if logits.shape[-2:] != targets.shape[-2:]:
        logits = F.interpolate(logits, size=targets.shape[-2:], mode="bilinear", align_corners=False)

    probs = (torch.sigmoid(logits) >= threshold).float()
    targets = (targets >= threshold).float()

    skel_pred = soft_skeletonize(probs, iter_num=iter_num)
    skel_gt = soft_skeletonize(targets, iter_num=iter_num)

    dims = (0, 2, 3)
    tprec = (torch.sum(skel_pred * targets, dims) + eps) / (torch.sum(skel_pred, dims) + eps)
    tsens = (torch.sum(skel_gt * probs, dims) + eps) / (torch.sum(skel_gt, dims) + eps)
    cl_dice = (2.0 * tprec * tsens + eps) / (tprec + tsens + eps)
    return float(cl_dice.mean().item()), float(tprec.mean().item()), float(tsens.mean().item())


@torch.no_grad()
def terminal_recall_from_logits(logits: torch.Tensor, targets: torch.Tensor,
                                threshold: float = 0.5,
                                iter_num: int = 10,
                                dilate_radius: int = 2,
                                eps: float = 1e-6):
    if logits.shape[-2:] != targets.shape[-2:]:
        logits = F.interpolate(logits, size=targets.shape[-2:], mode="bilinear", align_corners=False)

    probs = (torch.sigmoid(logits) >= threshold).float()
    targets = (targets >= threshold).float()

    skel_gt = soft_skeletonize(targets, iter_num=iter_num)
    terminal_map = estimate_terminal_map_from_skeleton(skel_gt, dilate_radius=dilate_radius)

    dims = (0, 2, 3)
    inter = torch.sum(terminal_map * probs * targets, dims)
    denom = torch.sum(terminal_map * targets, dims) + eps
    recall = (inter + eps) / denom
    return float(recall.mean().item())


@torch.no_grad()
def precision_sensitivity_from_logits(logits: torch.Tensor, targets: torch.Tensor,
                                      threshold: float = 0.5, eps: float = 1e-6):
    if logits.shape[-2:] != targets.shape[-2:]:
        logits = F.interpolate(logits, size=targets.shape[-2:], mode="bilinear", align_corners=False)

    probs = torch.sigmoid(logits)
    pred = (probs >= threshold)
    tgt = (targets >= threshold)

    dims = (0, 2, 3)
    tp = (pred & tgt).float().sum(dim=dims)
    fp = (pred & (~tgt)).float().sum(dim=dims)
    fn = ((~pred) & tgt).float().sum(dim=dims)

    precision = (tp + eps) / (tp + fp + eps)
    sensitivity = (tp + eps) / (tp + fn + eps)

    return (
        float(precision.mean().item()),
        float(sensitivity.mean().item()),
        pred,
        tgt,
    )


def mask_to_surface(mask, connectivity=1):
    mask = np.asarray(mask, dtype=bool)
    if mask.sum() == 0:
        return np.zeros_like(mask, dtype=bool)

    footprint = ndimage.generate_binary_structure(mask.ndim, connectivity)
    eroded = ndimage.binary_erosion(mask, structure=footprint, border_value=0)
    surface = np.logical_and(mask, np.logical_not(eroded))
    return surface


def surface_distances(mask_a, mask_b, connectivity=1):
    surface_a = mask_to_surface(mask_a, connectivity=connectivity)
    surface_b = mask_to_surface(mask_b, connectivity=connectivity)

    pts_a = np.argwhere(surface_a)
    pts_b = np.argwhere(surface_b)

    if len(pts_a) == 0 and len(pts_b) == 0:
        return np.array([0.0]), np.array([0.0])

    if len(pts_a) == 0 or len(pts_b) == 0:
        return None, None

    tree_a = cKDTree(pts_a.astype(np.float64))
    tree_b = cKDTree(pts_b.astype(np.float64))

    dists_ab, _ = tree_b.query(pts_a.astype(np.float64), k=1)
    dists_ba, _ = tree_a.query(pts_b.astype(np.float64), k=1)
    return dists_ab, dists_ba


def hd95_assd_single(pred_bin, gt_bin, connectivity=1):
    pred_bin = np.asarray(pred_bin, dtype=bool)
    gt_bin = np.asarray(gt_bin, dtype=bool)

    if pred_bin.sum() == 0 and gt_bin.sum() == 0:
        return 0.0, 0.0
    if pred_bin.sum() == 0 or gt_bin.sum() == 0:
        return float("inf"), float("inf")

    d_pg, d_gp = surface_distances(pred_bin, gt_bin, connectivity=connectivity)
    if d_pg is None or d_gp is None:
        return float("inf"), float("inf")

    all_dists = np.concatenate([d_pg, d_gp], axis=0)
    hd95 = float(np.percentile(all_dists, 95))
    assd = float((d_pg.mean() + d_gp.mean()) / 2.0)
    return hd95, assd


def multichannel_hd95_assd_from_binary(pred_bin: torch.Tensor, tgt_bin: torch.Tensor, connectivity=1):
    """
    pred_bin, tgt_bin: [B,C,H,W] bool tensor
    口径：
    - 每张图每个通道分别算 HD95/ASSD
    - 先对通道平均
    - 再对 batch 平均
    """
    pred_np = pred_bin.detach().cpu().numpy().astype(bool)
    tgt_np = tgt_bin.detach().cpu().numpy().astype(bool)

    sample_hd95 = []
    sample_assd = []

    bsz, ch, _, _ = pred_np.shape
    for b in range(bsz):
        ch_hd95 = []
        ch_assd = []
        for c in range(ch):
            hd95, assd = hd95_assd_single(
                pred_np[b, c],
                tgt_np[b, c],
                connectivity=connectivity,
            )
            ch_hd95.append(hd95)
            ch_assd.append(assd)

        sample_hd95.append(float(np.mean(ch_hd95)))
        sample_assd.append(float(np.mean(ch_assd)))

    batch_hd95 = float(np.mean(sample_hd95)) if len(sample_hd95) else 0.0
    batch_assd = float(np.mean(sample_assd)) if len(sample_assd) else 0.0
    return batch_hd95, batch_assd


def get_stage_params(
    epoch: int,
    warmup_epochs: int = 10,
    gate_start_epoch: int = 30,
    total_epochs: int = 350,
    tau_start: float = 0.30,
    tau_end: float = 0.55,
    temp_start: float = 1.0,
    temp_end: float = 0.25,
):
    if epoch < warmup_epochs:
        return "warmup", True, tau_start, temp_start
    if epoch < gate_start_epoch:
        return "soft", True, tau_start, temp_start

    progress = (epoch - gate_start_epoch) / max(1, total_epochs - gate_start_epoch)
    progress = min(max(progress, 0.0), 1.0)
    tau = tau_start + (tau_end - tau_start) * progress
    temp = temp_start + (temp_end - temp_start) * progress
    return "gate", False, tau, temp


def fmt_metric(x):
    if np.isinf(x):
        return "inf"
    if np.isnan(x):
        return "nan"
    return f"{x:.6f}"


def main():
    args = parse_args()

    device = torch.device(
        args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
    )
    print("Device:", device)

    if not os.path.exists(args.pt_path):
        raise FileNotFoundError(f"pt 文件不存在: {args.pt_path}")
    if not os.path.exists(args.image_dir):
        raise FileNotFoundError(f"测试图像目录不存在: {args.image_dir}")
    if not os.path.exists(args.gt_dir):
        raise FileNotFoundError(f"测试GT目录不存在: {args.gt_dir}")

    ckpt = torch.load(args.pt_path, map_location=device)
    cfg = ckpt.get("cfg", {})

    img_size = tuple(cfg.get("img_size", (args.img_w, args.img_h)))
    warmup_epochs = int(cfg.get("warmup_epochs", 40))
    gate_start_epoch = int(cfg.get("gate_start_epoch", 180))
    total_epochs = max(int(ckpt.get("epoch", 1)), gate_start_epoch + 1)

    model = ResUNetLikeBoundaryReasoningNet(
        in_channels=cfg.get("in_ch", 3),
        num_classes=cfg.get("out_ch", 3),
        base_ch=cfg.get("base_ch", 32),
        use_reasoning_at_1_4=cfg.get("use_reasoning_at_1_4", True),
        use_reasoning_at_1_2=cfg.get("use_reasoning_at_1_2", True),
    ).to(device)

    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    ds = VesselDataset(
        image_dir=args.image_dir,
        mask_dir=args.gt_dir,
        img_size=img_size,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    epoch = int(ckpt.get("epoch", 1))
    stage_mode, detach_pred, tau, temp = get_stage_params(
        epoch=epoch,
        warmup_epochs=warmup_epochs,
        gate_start_epoch=gate_start_epoch,
        total_epochs=total_epochs,
    )

    dice_list = []
    iou_list = []
    cldice_list = []
    tprec_list = []
    tsens_list = []
    termr_list = []
    precision_list = []
    sensitivity_list = []
    hd95_list = []
    assd_list = []
    batch_rows = []

    with torch.no_grad():
        for bidx, (images, masks, names) in enumerate(loader):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            logits, aux = model(
                images,
                stage_mode=stage_mode,
                detach_pred=detach_pred,
                tau=tau,
                temp=temp,
                return_aux=True,
            )

            d, j = dice_iou_from_logits(logits, masks, threshold=args.threshold)
            cd, tprec, tsens = cldice_from_logits(
                logits, masks, threshold=args.threshold, iter_num=10
            )
            tr = terminal_recall_from_logits(
                logits, masks, threshold=args.threshold, iter_num=10, dilate_radius=2
            )
            prec, sens, pred_bin, tgt_bin = precision_sensitivity_from_logits(
                logits, masks, threshold=args.threshold
            )
            batch_hd95, batch_assd = multichannel_hd95_assd_from_binary(
                pred_bin, tgt_bin, connectivity=args.surface_connectivity
            )

            dice_list.append(d)
            iou_list.append(j)
            cldice_list.append(cd)
            tprec_list.append(tprec)
            tsens_list.append(tsens)
            termr_list.append(tr)
            precision_list.append(prec)
            sensitivity_list.append(sens)
            hd95_list.append(batch_hd95)
            assd_list.append(batch_assd)

            batch_rows.append({
                "batch_idx": bidx,
                "num_samples": len(names),
                "files": " | ".join(list(names)),
                "dice": d,
                "iou": j,
                "cldice": cd,
                "tprec": tprec,
                "tsens": tsens,
                "terminal_recall": tr,
                "precision": prec,
                "sensitivity": sens,
                "hd95": batch_hd95,
                "assd": batch_assd,
            })

            print(
                f"[Batch {bidx+1}/{len(loader)}] "
                f"Dice={d:.6f} | IoU={j:.6f} | clDice={cd:.6f} | "
                f"termR={tr:.6f} | Precision={prec:.6f} | Sensitivity={sens:.6f} | "
                f"HD95={fmt_metric(batch_hd95)} | ASSD={fmt_metric(batch_assd)}"
            )

    mean_dice = float(np.mean(dice_list)) if len(dice_list) else 0.0
    mean_iou = float(np.mean(iou_list)) if len(iou_list) else 0.0
    mean_cldice = float(np.mean(cldice_list)) if len(cldice_list) else 0.0
    mean_tprec = float(np.mean(tprec_list)) if len(tprec_list) else 0.0
    mean_tsens = float(np.mean(tsens_list)) if len(tsens_list) else 0.0
    mean_termr = float(np.mean(termr_list)) if len(termr_list) else 0.0
    mean_precision = float(np.mean(precision_list)) if len(precision_list) else 0.0
    mean_sensitivity = float(np.mean(sensitivity_list)) if len(sensitivity_list) else 0.0
    mean_hd95 = float(np.mean(hd95_list)) if len(hd95_list) else 0.0
    mean_assd = float(np.mean(assd_list)) if len(assd_list) else 0.0

    print("=" * 100)
    print(f"PT file          : {args.pt_path}")
    print(f"Checkpoint epoch : {epoch}")
    print(f"Stage mode       : {stage_mode} | detach_pred={detach_pred} | tau={tau:.4f} | temp={temp:.4f}")
    print(f"Batch size       : {args.batch_size}")
    print(f"Resize size      : {img_size}")
    print(f"Mean Dice        : {mean_dice:.6f}")
    print(f"Mean IoU         : {mean_iou:.6f}")
    print(f"Mean clDice      : {mean_cldice:.6f}")
    print(f"Mean tprec       : {mean_tprec:.6f}")
    print(f"Mean tsens       : {mean_tsens:.6f}")
    print(f"Mean TerminalR   : {mean_termr:.6f}")
    print(f"Mean Precision   : {mean_precision:.6f}")
    print(f"Mean Sensitivity : {mean_sensitivity:.6f}")
    print(f"Mean HD95        : {fmt_metric(mean_hd95)}")
    print(f"Mean ASSD        : {fmt_metric(mean_assd)}")
    print("=" * 100)

    if args.save_csv:
        save_csv_path = Path(args.save_csv)
        save_csv_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "batch_idx", "num_samples", "files",
                    "dice", "iou", "cldice", "tprec", "tsens",
                    "terminal_recall", "precision", "sensitivity",
                    "hd95", "assd"
                ]
            )
            writer.writeheader()
            writer.writerows(batch_rows)
            writer.writerow({
                "batch_idx": "MEAN",
                "num_samples": "",
                "files": "",
                "dice": mean_dice,
                "iou": mean_iou,
                "cldice": mean_cldice,
                "tprec": mean_tprec,
                "tsens": mean_tsens,
                "terminal_recall": mean_termr,
                "precision": mean_precision,
                "sensitivity": mean_sensitivity,
                "hd95": mean_hd95,
                "assd": mean_assd,
            })

        print(f"[INFO] 已保存 CSV: {save_csv_path}")


if __name__ == "__main__":
    main()