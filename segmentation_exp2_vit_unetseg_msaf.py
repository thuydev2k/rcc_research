import os
import gc
import math
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from models.ViT_UNetSeg3D_MSAF import ViTUNetSeg3D_MSAF, count_parameters
from losses.SoftDiceCrossEntropyLoss import SoftDiceCrossEntropyLoss3D
from segmentation_exp1_vit_unetseg import (
    _make_scaler,
    _autocast,
    _update_multiclass_dice,
    _update_hec_dice,
)
from utils.plot_training_history import save_and_plot_training_history


def train_fn_exp2(loader, model, optimizer, device, criterion, scaler, use_amp=True):
    model.train()
    total_loss = 0.0

    for images, labels, clinical in tqdm(loader, desc='Train', leave=False):
        images = images.float().to(device)
        labels = labels.long().to(device)
        clinical = clinical.float().to(device)

        optimizer.zero_grad(set_to_none=True)

        with _autocast(device, enabled=use_amp):
            logits = model(images, clinical)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def eval_fn_exp2(loader, model, device, criterion, out_classes=4, use_amp=True):
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

    for images, labels, clinical in tqdm(loader, desc='Valid', leave=False):
        images = images.float().to(device)
        labels = labels.long().to(device)
        clinical = clinical.float().to(device)

        with _autocast(device, enabled=use_amp):
            logits = model(images, clinical)
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


def segmentation_exp2_vit_unetseg_msaf(
    train_loader,
    valid_loader,
    device,
    epochs,
    lr,
    clinical_dim,
    out_classes,
    roi_size=(96, 128, 128),
    save_dir='saved_Exp2_ViTUNetSeg3D_MSAF_model',
    feature_size=16,
    hidden_size=384,
    mlp_dim=1536,
    num_heads=12,
    patch_size=16,
    num_clinical_tokens=1,
    use_amp=True,
    init_from_exp1=None,
    clinical_feature_names=None,
):
    os.makedirs(save_dir, exist_ok=True)

    model = ViTUNetSeg3D_MSAF(
        clinical_dim=clinical_dim,
        in_channels=1,
        out_channels=out_classes,
        img_size=roi_size,
        feature_size=feature_size,
        hidden_size=hidden_size,
        mlp_dim=mlp_dim,
        num_heads=num_heads,
        patch_size=patch_size,
        num_clinical_tokens=num_clinical_tokens,
    ).to(device)

    if init_from_exp1 is not None:
        if os.path.exists(init_from_exp1):
            print(f'Loading Exp1 checkpoint into Exp2 image branch: {init_from_exp1}')
            ckpt = torch.load(init_from_exp1, map_location=device)
            incompatible = model.load_state_dict(ckpt['model'], strict=False)
            print(f'Missing keys from Exp1 checkpoint: {len(incompatible.missing_keys)}')
            print(f'Unexpected keys from Exp1 checkpoint: {len(incompatible.unexpected_keys)}')
        else:
            print(f'[Warning] init_from_exp1 was set but checkpoint was not found: {init_from_exp1}')
            print('Training Exp2 from scratch instead.')

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
    writer = SummaryWriter(log_dir='runs/exp2_vit_unetseg_msaf')
    history = []

    for epoch in range(epochs):
        train_loss = train_fn_exp2(
            train_loader,
            model,
            optimizer,
            device,
            criterion,
            scaler,
            use_amp=use_amp,
        )

        metrics = eval_fn_exp2(
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

        current_lr = optimizer.param_groups[0]['lr']
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

        history_row = {
            'epoch': epoch + 1,
            'lr': current_lr,
            'train_loss': train_loss,
            'valid_loss': valid_loss,
            'background_dice': metrics['background_dice'],
            'kidney_dice': metrics['kidney_dice'],
            'tumor_dice': metrics['tumor_dice'],
            'cyst_dice': metrics['cyst_dice'],
            'mean_fg_dice': metrics['mean_fg_dice'],
            'kidney_and_masses_dice': metrics['kidney_and_masses_dice'],
            'kidney_mass_dice': metrics['kidney_mass_dice'],
            'tumor_region_dice': metrics['tumor_region_dice'],
        }
        history.append(history_row)
        save_and_plot_training_history(history, save_dir)

        checkpoint_common = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scaler': scaler.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epoch,
            'metrics': metrics,
            'history': history,
            'roi_size': roi_size,
            'clinical_dim': clinical_dim,
            'clinical_feature_names': clinical_feature_names,
            'out_classes': out_classes,
            'feature_size': feature_size,
            'hidden_size': hidden_size,
            'mlp_dim': mlp_dim,
            'num_heads': num_heads,
            'patch_size': patch_size,
            'num_clinical_tokens': num_clinical_tokens,
        }

        last_path = os.path.join(save_dir, 'last_model_exp2.pt')
        torch.save(checkpoint_common, last_path)

        if metrics['mean_fg_dice'] > best_mean_fg_dice:
            best_mean_fg_dice = metrics['mean_fg_dice']
            best_path = os.path.join(save_dir, 'best_model_exp2.pt')

            checkpoint_best = dict(checkpoint_common)
            checkpoint_best['best_mean_fg_dice'] = best_mean_fg_dice
            torch.save(checkpoint_best, best_path)

            print(f'Model saved: {best_path}')
            print(f'Best mean foreground Dice: {best_mean_fg_dice:.6f}')

        torch.cuda.empty_cache()
        gc.collect()

    writer.close()
