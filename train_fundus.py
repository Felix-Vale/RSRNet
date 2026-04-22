import os
import random
import numpy as np
from PIL import Image, ImageEnhance
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from ResUNet import ResUNetLikeBoundaryReasoningNet


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def add_gaussian_noise_to_tensor(img_t: torch.Tensor, std_range=(0.01, 0.05), p: float = 0.35) -> torch.Tensor:
    if random.random() >= p:
        return img_t
    std = random.uniform(*std_range)
    noise = torch.randn_like(img_t) * std
    img_t = (img_t + noise).clamp(0.0, 1.0)
    return img_t


def random_brightness_contrast_pil(img: Image.Image,
                                   brightness_range=(0.85, 1.15),
                                   contrast_range=(0.85, 1.20),
                                   p: float = 0.6) -> Image.Image:
    if random.random() < p:
        b = random.uniform(*brightness_range)
        img = ImageEnhance.Brightness(img).enhance(b)
        c = random.uniform(*contrast_range)
        img = ImageEnhance.Contrast(img).enhance(c)
    return img


def random_local_region_enhance_pil(img: Image.Image,
                                    p: float = 0.45,
                                    patch_scale_range=(0.20, 0.45),
                                    brightness_range=(1.05, 1.25),
                                    contrast_range=(1.05, 1.35)) -> Image.Image:
    if random.random() >= p:
        return img

    img_np = np.array(img).astype(np.uint8)
    h, w = img_np.shape[:2]
    ph = max(16, int(h * random.uniform(*patch_scale_range)))
    pw = max(16, int(w * random.uniform(*patch_scale_range)))
    top = random.randint(0, max(0, h - ph))
    left = random.randint(0, max(0, w - pw))

    patch = Image.fromarray(img_np[top:top + ph, left:left + pw])
    patch = ImageEnhance.Brightness(patch).enhance(random.uniform(*brightness_range))
    patch = ImageEnhance.Contrast(patch).enhance(random.uniform(*contrast_range))
    img_np[top:top + ph, left:left + pw] = np.array(patch).astype(np.uint8)
    return Image.fromarray(img_np)


class VesselDataset(Dataset):
    def __init__(self,
                 image_dir,
                 mask_dir,
                 img_size=None,
                 augment: bool = False,
                 hflip_prob: float = 0.5,
                 vflip_prob: float = 0.0,
                 noise_prob: float = 0.35,
                 local_enhance_prob: float = 0.45,
                 global_bc_prob: float = 0.60):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.augment = bool(augment)

        self.hflip_prob = float(hflip_prob)
        self.vflip_prob = float(vflip_prob)
        self.noise_prob = float(noise_prob)
        self.local_enhance_prob = float(local_enhance_prob)
        self.global_bc_prob = float(global_bc_prob)

        exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
        self.image_files = [f for f in os.listdir(self.image_dir) if f.lower().endswith(exts)]
        self.image_files.sort()

        if len(self.image_files) == 0:
            raise RuntimeError(f"No images found in {self.image_dir}.")

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask not found for image {img_name}: {mask_path}")

        img = Image.open(img_path).convert("RGB")
        # 由原来的单通道掩码改为 3 通道掩码（眼底图像分割）
        mask = Image.open(mask_path).convert("RGB")

        if self.img_size is not None:
            img = img.resize(self.img_size, resample=Image.BILINEAR)
            mask = mask.resize(self.img_size, resample=Image.NEAREST)

        if self.augment:
            if random.random() < self.hflip_prob:
                img = TF.hflip(img)
                mask = TF.hflip(mask)
            if random.random() < self.vflip_prob:
                img = TF.vflip(img)
                mask = TF.vflip(mask)

            img = random_brightness_contrast_pil(
                img,
                brightness_range=(0.85, 1.15),
                contrast_range=(0.85, 1.20),
                p=self.global_bc_prob,
            )

            img = random_local_region_enhance_pil(
                img,
                p=self.local_enhance_prob,
                patch_scale_range=(0.20, 0.45),
                brightness_range=(1.05, 1.25),
                contrast_range=(1.05, 1.35),
            )

        img_t = self.to_tensor(img)

        if self.augment:
            img_t = add_gaussian_noise_to_tensor(img_t, std_range=(0.01, 0.05), p=self.noise_prob)

        img_t = self.normalize(img_t)

        mask_np = np.array(mask, dtype=np.float32)
        mask_np = np.clip(mask_np / 255.0, 0.0, 1.0)
        # [H, W, 3] -> [3, H, W]
        mask_t = torch.from_numpy(mask_np).permute(2, 0, 1).float()

        return img_t, mask_t


def soft_dice_loss_multichannel(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if logits.shape[-2:] != targets.shape[-2:]:
        logits = F.interpolate(logits, size=targets.shape[-2:], mode="bilinear", align_corners=False)

    probs = torch.sigmoid(logits)
    targets = targets.float()

    dims = (0, 2, 3)
    intersection = torch.sum(probs * targets, dims)
    union = torch.sum(probs, dims) + torch.sum(targets, dims)

    dice = (2.0 * intersection + eps) / (union + eps)
    return 1.0 - dice.mean()


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


def soft_cldice_loss(logits: torch.Tensor, targets: torch.Tensor, iter_num: int = 10, eps: float = 1e-6) -> torch.Tensor:
    if logits.shape[-2:] != targets.shape[-2:]:
        logits = F.interpolate(logits, size=targets.shape[-2:], mode="bilinear", align_corners=False)

    probs = torch.sigmoid(logits).clamp(0.0, 1.0)
    targets = targets.float().clamp(0.0, 1.0)

    skel_pred = soft_skeletonize(probs, iter_num=iter_num)
    skel_gt = soft_skeletonize(targets, iter_num=iter_num)

    dims = (0, 2, 3)
    tprec = (torch.sum(skel_pred * targets, dims) + eps) / (torch.sum(skel_pred, dims) + eps)
    tsens = (torch.sum(skel_gt * probs, dims) + eps) / (torch.sum(skel_gt, dims) + eps)

    cl_dice = (2.0 * tprec * tsens + eps) / (tprec + tsens + eps)
    return 1.0 - cl_dice.mean()


def topology_branch_recall_loss(logits: torch.Tensor,
                                targets: torch.Tensor,
                                iter_num: int = 10,
                                branch_focus_gamma: float = 2.0,
                                eps: float = 1e-6) -> torch.Tensor:
    if logits.shape[-2:] != targets.shape[-2:]:
        logits = F.interpolate(logits, size=targets.shape[-2:], mode="bilinear", align_corners=False)

    probs = torch.sigmoid(logits).clamp(0.0, 1.0)
    targets = targets.float().clamp(0.0, 1.0)

    skel_gt = soft_skeletonize(targets, iter_num=iter_num)
    branch_weight = 1.0 + branch_focus_gamma * skel_gt

    dims = (0, 2, 3)
    inter = torch.sum(branch_weight * probs * targets, dims)
    denom = torch.sum(branch_weight * targets, dims) + eps
    recall = (inter + eps) / denom
    return 1.0 - recall.mean()


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


def terminal_aware_loss(logits: torch.Tensor,
                        targets: torch.Tensor,
                        iter_num: int = 10,
                        terminal_gamma: float = 4.0,
                        dilate_radius: int = 2,
                        eps: float = 1e-6) -> torch.Tensor:
    if logits.shape[-2:] != targets.shape[-2:]:
        logits = F.interpolate(logits, size=targets.shape[-2:], mode="bilinear", align_corners=False)

    probs = torch.sigmoid(logits).clamp(0.0, 1.0)
    targets = targets.float().clamp(0.0, 1.0)

    skel_gt = soft_skeletonize(targets, iter_num=iter_num)
    terminal_map = estimate_terminal_map_from_skeleton(skel_gt, dilate_radius=dilate_radius)

    terminal_weight = 1.0 + terminal_gamma * terminal_map
    dims = (0, 2, 3)
    inter = torch.sum(terminal_weight * probs * targets, dims)
    denom = torch.sum(terminal_weight * targets, dims) + eps
    recall = (inter + eps) / denom
    return 1.0 - recall.mean()


class VesselHybridTopologyLossV2(nn.Module):
    def __init__(self,
                 dice_weight: float = 0.5,
                 cldice_weight: float = 0.20,
                 branch_weight: float = 0.15,
                 terminal_weight: float = 0.10,
                 skeleton_iter: int = 10,
                 branch_focus_gamma: float = 2.0,
                 terminal_gamma: float = 4.0,
                 terminal_dilate_radius: int = 2):
        super().__init__()
        self.dice_weight = float(dice_weight)
        self.cldice_weight = float(cldice_weight)
        self.branch_weight = float(branch_weight)
        self.terminal_weight = float(terminal_weight)

        self.skeleton_iter = int(skeleton_iter)
        self.branch_focus_gamma = float(branch_focus_gamma)
        self.terminal_gamma = float(terminal_gamma)
        self.terminal_dilate_radius = int(terminal_dilate_radius)

        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.shape[-2:] != targets.shape[-2:]:
            logits = F.interpolate(logits, size=targets.shape[-2:], mode="bilinear", align_corners=False)

        targets = targets.float()
        loss_bce = self.bce(logits, targets)
        loss_dice = soft_dice_loss_multichannel(logits, targets)
        loss_cldice = soft_cldice_loss(logits, targets, iter_num=self.skeleton_iter)
        loss_branch = topology_branch_recall_loss(
            logits, targets,
            iter_num=self.skeleton_iter,
            branch_focus_gamma=self.branch_focus_gamma,
        )
        loss_terminal = terminal_aware_loss(
            logits, targets,
            iter_num=self.skeleton_iter,
            terminal_gamma=self.terminal_gamma,
            dilate_radius=self.terminal_dilate_radius,
        )

        total = (
            loss_bce
            + self.dice_weight * loss_dice
            + self.cldice_weight * loss_cldice
            + self.branch_weight * loss_branch
            + self.terminal_weight * loss_terminal
        )
        return total


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
def cldice_from_logits(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, iter_num: int = 10, eps: float = 1e-6) -> float:
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
    return float(cl_dice.mean().item())


@torch.no_grad()
def terminal_recall_from_logits(logits: torch.Tensor,
                                targets: torch.Tensor,
                                threshold: float = 0.5,
                                iter_num: int = 10,
                                dilate_radius: int = 2,
                                eps: float = 1e-6) -> float:
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


def collect_aux_logits_from_aux(aux: dict):
    aux_logits = []
    if not isinstance(aux, dict):
        return aux_logits

    for key in ["reasoning_1_4", "reasoning_1_2"]:
        stage_aux = aux.get(key, None)
        if isinstance(stage_aux, dict):
            refine_logit = stage_aux.get("refine_logit", None)
            if torch.is_tensor(refine_logit):
                aux_logits.append(refine_logit)
    return aux_logits




def align_aux_logit_to_target_channels(aux_logit: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    让 aux 分支的通道数与 targets 对齐，但不改变其它训练策略。

    说明：
    - 当前 ResUNetLikeBoundaryReasoningNet 的主输出已经可设为 num_classes=3；
    - 但内部 reasoning/refine 的 aux_logit 仍可能固定输出 1 通道；
    - 为保持原有 aux loss 训练流程，这里在 loss 前做最小对齐：
        1 通道 aux -> 按类别数复制到 C 通道。
    """
    if aux_logit.dim() != 4 or targets.dim() != 4:
        raise ValueError(f"aux_logit/targets must be [B,C,H,W], got {tuple(aux_logit.shape)} and {tuple(targets.shape)}")

    if aux_logit.shape[-2:] != targets.shape[-2:]:
        aux_logit = F.interpolate(aux_logit, size=targets.shape[-2:], mode="bilinear", align_corners=False)

    c_aux = aux_logit.shape[1]
    c_tgt = targets.shape[1]

    if c_aux == c_tgt:
        return aux_logit

    if c_aux == 1 and c_tgt > 1:
        return aux_logit.repeat(1, c_tgt, 1, 1)

    raise ValueError(
        f"Aux logits channel mismatch: aux has {c_aux} channel(s), target has {c_tgt} channel(s). "
        f"Please modify the aux heads in ResUNet to output num_classes={c_tgt}."
    )

def train_one_epoch(model, loader, criterion, optimizer, device,
                    epoch: int, num_epochs: int,
                    aux_weight=0.4,
                    warmup_epochs: int = 10,
                    gate_start_epoch: int = 30):
    model.train()
    running = 0.0

    stage_mode, detach_pred, tau, temp = get_stage_params(
        epoch=epoch,
        warmup_epochs=warmup_epochs,
        gate_start_epoch=gate_start_epoch,
        total_epochs=num_epochs,
    )

    pbar = tqdm(loader, desc=f"Train [{stage_mode}]", ncols=120)
    for images, masks in pbar:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        logits, aux = model(
            images,
            stage_mode=stage_mode,
            detach_pred=detach_pred,
            tau=tau,
            temp=temp,
            return_aux=True,
        )

        loss = criterion(logits, masks)
        aux_logits = collect_aux_logits_from_aux(aux)
        for aux_logit in aux_logits:
            aux_logit = align_aux_logit_to_target_channels(aux_logit, masks)
            loss = loss + aux_weight * criterion(aux_logit, masks)

        loss.backward()
        optimizer.step()

        running += loss.item() * images.size(0)
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "tau": f"{tau:.3f}", "temp": f"{temp:.3f}"})

    return running / len(loader.dataset)


@torch.no_grad()
def validate(model, loader, criterion, device,
             epoch: int, num_epochs: int,
             aux_weight=0.4,
             warmup_epochs: int = 10,
             gate_start_epoch: int = 30):
    model.eval()
    running = 0.0
    dice_list = []
    iou_list = []
    cldice_list = []
    terminal_recall_list = []

    stage_mode, detach_pred, tau, temp = get_stage_params(
        epoch=epoch,
        warmup_epochs=warmup_epochs,
        gate_start_epoch=gate_start_epoch,
        total_epochs=num_epochs,
    )

    pbar = tqdm(loader, desc=f"Val [{stage_mode}]", ncols=120)
    for images, masks in pbar:
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

        loss = criterion(logits, masks)
        aux_logits = collect_aux_logits_from_aux(aux)
        for aux_logit in aux_logits:
            aux_logit = align_aux_logit_to_target_channels(aux_logit, masks)
            loss = loss + aux_weight * criterion(aux_logit, masks)

        d, j = dice_iou_from_logits(logits, masks)
        cd = cldice_from_logits(logits, masks)
        tr = terminal_recall_from_logits(logits, masks)

        dice_list.append(d)
        iou_list.append(j)
        cldice_list.append(cd)
        terminal_recall_list.append(tr)

        running += loss.item() * images.size(0)
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "dice": f"{d:.4f}",
            "iou": f"{j:.4f}",
            "clDice": f"{cd:.4f}",
            "termR": f"{tr:.4f}",
            "tau": f"{tau:.3f}",
            "temp": f"{temp:.3f}",
        })

    val_loss = running / len(loader.dataset)
    mean_dice = float(np.mean(dice_list)) if len(dice_list) else 0.0
    mean_iou = float(np.mean(iou_list)) if len(iou_list) else 0.0
    mean_cldice = float(np.mean(cldice_list)) if len(cldice_list) else 0.0
    mean_terminal_recall = float(np.mean(terminal_recall_list)) if len(terminal_recall_list) else 0.0
    return val_loss, mean_dice, mean_iou, mean_cldice, mean_terminal_recall


def main():
    train_image_dir = r""
    train_mask_dir  = r""
    val_image_dir   = r""
    val_mask_dir    = r""

    seed = 32
    num_classes = 3
    num_epochs = 500
    lr_init = 1e-3 # 5e-4  # 1e-4
    lr_min = 5e-6
    img_size = (320, 320)
    batch_size = 4
    weight_decay = 1e-4
    aux_weight = 0.4

    dice_weight = 0.5
    cldice_weight = 0.20
    branch_weight = 0.15
    terminal_weight = 0.10
    skeleton_iter = 10
    branch_focus_gamma = 2.0
    terminal_gamma = 4.0
    terminal_dilate_radius = 2

    num_workers = 4
    warmup_epochs = 40
    gate_start_epoch = 180

    save_dir = r""
    os.makedirs(save_dir, exist_ok=True)

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_ds = VesselDataset(
        train_image_dir,
        train_mask_dir,
        img_size=img_size,
        augment=True,
        hflip_prob=0.5,
        vflip_prob=0.0,
        noise_prob=0.35,
        local_enhance_prob=0.45,
        global_bc_prob=0.60,
    )
    val_ds = VesselDataset(val_image_dir, val_mask_dir, img_size=img_size, augment=False)
    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=False)

    model = ResUNetLikeBoundaryReasoningNet(
        in_channels=3,
        num_classes=num_classes,
        base_ch=32,
        use_reasoning_at_1_4=True,
        use_reasoning_at_1_2=True,
    ).to(device)

    criterion = VesselHybridTopologyLossV2(
        dice_weight=dice_weight,
        cldice_weight=cldice_weight,
        branch_weight=branch_weight,
        terminal_weight=terminal_weight,
        skeleton_iter=skeleton_iter,
        branch_focus_gamma=branch_focus_gamma,
        terminal_gamma=terminal_gamma,
        terminal_dilate_radius=terminal_dilate_radius,
    )

    optimizer = optim.AdamW(model.parameters(), lr=lr_init, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=lr_min)

    best_score = float("-inf")
    best_epoch = -1
    prev_best_path = None
    delete_prev_best = False

    for epoch in range(1, num_epochs + 1):
        print(f"\n===== Epoch [{epoch}/{num_epochs}] =====")
        stage_mode, detach_pred, tau, temp = get_stage_params(
            epoch=epoch,
            warmup_epochs=warmup_epochs,
            gate_start_epoch=gate_start_epoch,
            total_epochs=num_epochs,
        )
        print(f"Stage: {stage_mode} | detach_pred={detach_pred} | tau={tau:.4f} | temp={temp:.4f}")

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            epoch=epoch, num_epochs=num_epochs,
            aux_weight=aux_weight,
            warmup_epochs=warmup_epochs,
            gate_start_epoch=gate_start_epoch,
        )

        val_loss, val_dice, val_iou, val_cldice, val_terminal_recall = validate(
            model, val_loader, criterion, device,
            epoch=epoch, num_epochs=num_epochs,
            aux_weight=aux_weight,
            warmup_epochs=warmup_epochs,
            gate_start_epoch=gate_start_epoch,
        )

        scheduler.step()

        current_score = 0.65 * val_dice + 0.25 * val_cldice + 0.10 * val_terminal_recall

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"LR: {current_lr:.8f}")
        print(f"Train Loss: {train_loss:.6f}")
        print(
            f"Val   Loss: {val_loss:.6f} | Dice: {val_dice:.4f} | IoU: {val_iou:.4f} | "
            f"clDice: {val_cldice:.4f} | TerminalRecall: {val_terminal_recall:.4f}"
        )
        print(f"Selection Score (0.65*Dice + 0.25*clDice + 0.10*TerminalRecall): {current_score:.4f}")

        if current_score > best_score + 1e-8:
            best_score = float(current_score)
            best_epoch = int(epoch)

            ckpt = {
                "epoch": best_epoch,
                "model_state": model.state_dict(),
                "best_score": best_score,
                "best_val_dice": float(val_dice),
                "best_val_cldice": float(val_cldice),
                "best_val_terminal_recall": float(val_terminal_recall),
                "val_loss": float(val_loss),
                "val_iou": float(val_iou),
                "cfg": {
                    "model": "ResUNetLikeBoundaryReasoningNet",
                    "in_ch": 3,
                    "out_ch": num_classes,
                    "img_size": img_size,
                    "base_ch": 32,
                    "use_reasoning_at_1_4": True,
                    "use_reasoning_at_1_2": True,
                    "warmup_epochs": warmup_epochs,
                    "gate_start_epoch": gate_start_epoch,
                    "loss": "BCE + Dice + clDice + branch_recall + terminal_aware",
                    "dice_weight": dice_weight,
                    "cldice_weight": cldice_weight,
                    "branch_weight": branch_weight,
                    "terminal_weight": terminal_weight,
                    "skeleton_iter": skeleton_iter,
                    "branch_focus_gamma": branch_focus_gamma,
                    "terminal_gamma": terminal_gamma,
                    "terminal_dilate_radius": terminal_dilate_radius,
                }
            }

            if delete_prev_best and prev_best_path is not None and os.path.exists(prev_best_path):
                try:
                    os.remove(prev_best_path)
                except Exception as e:
                    print(f"Warning: 删除旧 best checkpoint 失败: {prev_best_path} | {repr(e)}")

            save_path = os.path.join(
                save_dir,
                f"best_score_{best_score:.4f}_dice_{val_dice:.4f}_cldice_{val_cldice:.4f}_termR_{val_terminal_recall:.4f}_epoch{best_epoch}.pt"
            )
            torch.save(ckpt, save_path)
            prev_best_path = save_path
            print(f"[SAVE] New best score: {best_score:.4f} (epoch {best_epoch}) -> {save_path}")

    print(f"\n训练结束！Best Score = {best_score:.4f} (epoch {best_epoch})")


if __name__ == "__main__":
    main()
