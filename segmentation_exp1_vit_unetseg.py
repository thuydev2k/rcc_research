import os
import gc
import math
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from models.ViT_UNetSeg3D import ViTUNetSeg3D, count_parameters
from losses.SoftDiceCrossEntropyLoss import SoftDiceCrossEntropyLoss3D


def _device_type(device):
    device_str = str(device)
    return 'cuda' if device_str.startswith('cuda') else 'cpu'


def _make_scaler(device, enabled=True):
    use_cuda_amp = enabled and _device_type(device) == 'cuda'
    try:
        return torch.amp.GradScaler('cuda', enabled=use_cuda_amp)
    except TypeError:
        return torch.cuda.amp.GradScaler(enabled=use_cuda_amp)


def _autocast(device, enabled=True):
    use_cuda_amp = enabled and _device_type(device) == 'cuda'
    return torch.amp.autocast(device_type=_device_type(device), enabled=use_cuda_amp)


@torch.no_grad()
def _update_multiclass_dice(logits, labels, numerator, denominator, out_classes):
    preds = torch.argmax(logits, dim=1)

    for c in range(out_classes):
        pred_c = preds == c
        label_c = labels == c

        numerator[c] += 2.0 * torch.logical_and(pred_c, label_c).sum()
        denominator[c] += pred_c.sum() + label_c.sum()


@torch.no_grad()
def _update_hec_dice(logits, labels, numerator, denominator):
    preds = torch.argmax(logits, dim=1)

    hec_defs = {
        'kidney_and_masses': [1, 2, 3],
        'kidney_mass': [2, 3],
        'tumor': [2],
    }

    for region_name, class_ids in hec_defs.items():
        pred_region = torch.zeros_like(preds, dtype=torch.bool)
        label_region = torch.zeros_like(labels, dtype=torch.bool)

        for c in class_ids:
            pred_region |= preds == c
            label_region |= labels == c

        numerator[region_name] += 2.0 * torch.logical_and(pred_region, label_region).sum()
        denominator[region_name] += pred_region.sum() + label_region.sum()


def train_fn_exp1(loader, model, optimizer, device, criterion, scaler, use_amp=True):
    model.train()
    total_loss = 0.0

    for images, labels in tqdm(loader, desc='Train', leave=False):
        images = images.float().to(device)
        labels = labels.long().to(device)

        optimizer.zero_grad(set_to_none=True)

        with _autocast(device, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def eval_fn_exp1(loader, model, device, criterion, out_classes=4, use_amp=True):
    model.eval()
    total_loss = 0.0

    class_num = torch.zeros(out_classes, device=device)
    class_den = torch.zeros(out_classes, device=device)

    hec_num = {
        'kidney_and_masses': torch.tensor(0.0, device=device),
        'kidney_mass': torch.tensor(0.0, device=device),
        'tumor': torch.tensor(0.0, device=device),
    }
    hec_den = {
        'kidney_and_masses': torch.tensor(0.0, device=device),
        'kidney_mass': torch.tensor(0.0, device=device),
        'tumor': torch.tensor(0.0, device=device),
    }

    for images, labels in tqdm(loader, desc='Valid', leave=False):
        images = images.float().to(device)
        labels = labels.long().to(device)

        with _autocast(device, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

        total_loss += loss.item()

        _update_multiclass_dice(logits, labels, class_num, class_den, out_classes)
        _update_hec_dice(logits, labels, hec_num, hec_den)

    eps = 1e-8
    class_dice = (class_num + eps) / (class_den + eps)

    metrics = {
        'valid_loss': total_loss / max(len(loader), 1),
        'background_dice': class_dice[0].item(),
        'kidney_dice': class_dice[1].item(),
        'tumor_dice': class_dice[2].item(),
        'cyst_dice': class_dice[3].item(),
        'mean_fg_dice': class_dice[1:].mean().item(),
        'kidney_and_masses_dice': ((hec_num['kidney_and_masses'] + eps) / (hec_den['kidney_and_masses'] + eps)).item(),
        'kidney_mass_dice': ((hec_num['kidney_mass'] + eps) / (hec_den['kidney_mass'] + eps)).item(),
        'tumor_region_dice': ((hec_num['tumor'] + eps) / (hec_den['tumor'] + eps)).item(),
    }

    return metrics


def segmentation_exp1_vit_unetseg(
    train_loader,
    valid_loader,
    device,
    epochs,
    lr,
    out_classes,
    roi_size=(128, 256, 256),
    save_dir='saved_Exp1_ViTUNetSeg3D_model',
    feature_size=16,
    hidden_size=384,
    mlp_dim=1536,
    num_heads=12,
    patch_size=16,
    use_amp=True,
):
    os.makedirs(save_dir, exist_ok=True)

    model = ViTUNetSeg3D(
        in_channels=1,
        out_channels=out_classes,
        img_size=roi_size,
        feature_size=feature_size,
        hidden_size=hidden_size,
        mlp_dim=mlp_dim,
        num_heads=num_heads,
        patch_size=patch_size,
    ).to(device)

    count_parameters(model)

    scaler = _make_scaler(device, enabled=use_amp)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-6,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=lr * 0.01,
    )

    criterion = SoftDiceCrossEntropyLoss3D(
        ce_weight=0.5,
        dice_weight=0.5,
        ignore_bg=False,
    ).to(device)

    best_mean_fg_dice = -1.0
    writer = SummaryWriter(log_dir='runs/exp1_vit_unetseg')

    for epoch in range(epochs):
        train_loss = train_fn_exp1(
            train_loader,
            model,
            optimizer,
            device,
            criterion,
            scaler,
            use_amp=use_amp,
        )

        metrics = eval_fn_exp1(
            valid_loader,
            model,
            device,
            criterion,
            out_classes=out_classes,
            use_amp=use_amp,
        )

        valid_loss = metrics['valid_loss']

        if math.isnan(train_loss) or math.isnan(valid_loss):
            print(f'Early stopping at epoch {epoch + 1} because loss is NaN')
            break

        scheduler.step()

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/valid', valid_loss, epoch)
        writer.add_scalar('Dice/mean_fg', metrics['mean_fg_dice'], epoch)
        writer.add_scalar('Dice/kidney', metrics['kidney_dice'], epoch)
        writer.add_scalar('Dice/tumor', metrics['tumor_dice'], epoch)
        writer.add_scalar('Dice/cyst', metrics['cyst_dice'], epoch)
        writer.add_scalar('HEC/kidney_and_masses', metrics['kidney_and_masses_dice'], epoch)
        writer.add_scalar('HEC/kidney_mass', metrics['kidney_mass_dice'], epoch)
        writer.add_scalar('HEC/tumor_region', metrics['tumor_region_dice'], epoch)

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Train Loss: {train_loss:.6f} "
            f"Valid Loss: {valid_loss:.6f} "
            f"Mean FG Dice: {metrics['mean_fg_dice']:.6f} "
            f"Kidney: {metrics['kidney_dice']:.6f} "
            f"Tumor: {metrics['tumor_dice']:.6f} "
            f"Cyst: {metrics['cyst_dice']:.6f} "
            f"K&M: {metrics['kidney_and_masses_dice']:.6f} "
            f"Mass: {metrics['kidney_mass_dice']:.6f} "
            f"TumorRegion: {metrics['tumor_region_dice']:.6f}"
        )

        last_path = os.path.join(save_dir, 'last_model_exp1.pt')
        torch.save(
            {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scaler': scaler.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch,
                'metrics': metrics,
                'roi_size': roi_size,
                'out_classes': out_classes,
                'feature_size': feature_size,
                'hidden_size': hidden_size,
                'mlp_dim': mlp_dim,
                'num_heads': num_heads,
                'patch_size': patch_size,
            },
            last_path,
        )

        if metrics['mean_fg_dice'] > best_mean_fg_dice:
            best_mean_fg_dice = metrics['mean_fg_dice']

            best_path = os.path.join(save_dir, 'best_model_exp1.pt')
            torch.save(
                {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scaler': scaler.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': epoch,
                    'best_mean_fg_dice': best_mean_fg_dice,
                    'metrics': metrics,
                    'roi_size': roi_size,
                    'out_classes': out_classes,
                    'feature_size': feature_size,
                    'hidden_size': hidden_size,
                    'mlp_dim': mlp_dim,
                    'num_heads': num_heads,
                    'patch_size': patch_size,
                },
                best_path,
            )

            print(f'Model saved: {best_path}')
            print(f'Best mean foreground Dice: {best_mean_fg_dice:.6f}')

        torch.cuda.empty_cache()
        gc.collect()

    writer.close()
