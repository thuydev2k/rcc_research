import os
import gc
import math
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from models.ViT_UNetSeg2D import ViTUNetSeg2D, count_parameters
from losses.SoftDiceCrossEntropyLoss2D import SoftDiceCrossEntropyLoss2D

HEC_REGION_DEFS = {
    'kidney_and_masses': [1, 2, 3],
    'masses': [2, 3],
    'tumor': [2],
}


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
def _update_multiclass_stats_2d(logits, labels, tp, fp, fn, out_classes):
    preds = torch.argmax(logits, dim=1)

    for c in range(out_classes):
        pred_c = preds == c
        label_c = labels == c

        tp[c] += torch.logical_and(pred_c, label_c).sum()
        fp[c] += torch.logical_and(pred_c, ~label_c).sum()
        fn[c] += torch.logical_and(~pred_c, label_c).sum()


@torch.no_grad()
def _update_hec_stats_2d(logits, labels, stats):
    preds = torch.argmax(logits, dim=1)

    for region_name, class_ids in HEC_REGION_DEFS.items():
        pred_region = torch.zeros_like(preds, dtype=torch.bool)
        label_region = torch.zeros_like(labels, dtype=torch.bool)

        for c in class_ids:
            pred_region |= preds == c
            label_region |= labels == c

        stats[region_name]['tp'] += torch.logical_and(pred_region, label_region).sum()
        stats[region_name]['fp'] += torch.logical_and(pred_region, ~label_region).sum()
        stats[region_name]['fn'] += torch.logical_and(~pred_region, label_region).sum()


def _compute_dice_iou_from_stats(tp, fp, fn, eps=1e-8):
    dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    return dice, iou


def train_fn_exp1_2d(loader, model, optimizer, device, criterion, scaler, use_amp=True):
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
def eval_fn_exp1_2d(loader, model, device, criterion, out_classes=4, use_amp=True):
    model.eval()
    total_loss = 0.0

    class_tp = torch.zeros(out_classes, device=device)
    class_fp = torch.zeros(out_classes, device=device)
    class_fn = torch.zeros(out_classes, device=device)

    hec_stats = {
        region: {
            'tp': torch.tensor(0.0, device=device),
            'fp': torch.tensor(0.0, device=device),
            'fn': torch.tensor(0.0, device=device),
        }
        for region in HEC_REGION_DEFS.keys()
    }

    for images, labels in tqdm(loader, desc='Valid', leave=False):
        images = images.float().to(device)
        labels = labels.long().to(device)

        with _autocast(device, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

        total_loss += loss.item()
        _update_multiclass_stats_2d(logits, labels, class_tp, class_fp, class_fn, out_classes)
        _update_hec_stats_2d(logits, labels, hec_stats)

    class_dice, class_iou = _compute_dice_iou_from_stats(class_tp, class_fp, class_fn)

    metrics = {
        'valid_loss': total_loss / max(len(loader), 1),
        'background_dice': class_dice[0].item(),
        'background_iou': class_iou[0].item(),
        'kidney_dice': class_dice[1].item(),
        'kidney_iou': class_iou[1].item(),
        'tumor_dice': class_dice[2].item(),
        'tumor_iou': class_iou[2].item(),
        'cyst_dice': class_dice[3].item(),
        'cyst_iou': class_iou[3].item(),
        'mean_fg_dice': class_dice[1:].mean().item(),
        'mean_fg_iou': class_iou[1:].mean().item(),
    }

    hec_dices = []
    hec_ious = []
    for region_name, stat in hec_stats.items():
        dice, iou = _compute_dice_iou_from_stats(stat['tp'], stat['fp'], stat['fn'])
        metrics[f'{region_name}_dice'] = dice.item()
        metrics[f'{region_name}_iou'] = iou.item()
        hec_dices.append(dice)
        hec_ious.append(iou)

    metrics['mean_hec_dice'] = torch.stack(hec_dices).mean().item()
    metrics['mean_hec_iou'] = torch.stack(hec_ious).mean().item()

    # Alias for clearer printed name.
    metrics['tumor_region_dice'] = metrics['tumor_dice']
    metrics['tumor_region_iou'] = metrics['tumor_iou']

    return metrics


def segmentation_exp1_vit_unetseg_2d(
    train_loader,
    valid_loader,
    device,
    epochs,
    lr,
    out_classes,
    img_size=(512, 512),
    save_dir='saved_Exp1_ViTUNetSeg2D_HEC_model',
    feature_size=16,
    hidden_size=384,
    mlp_dim=1536,
    num_heads=12,
    patch_size=16,
    use_amp=True,
    early_stopping_patience=25,
    early_stopping_min_delta=1e-5,
):
    os.makedirs(save_dir, exist_ok=True)

    model = ViTUNetSeg2D(
        in_channels=1,
        out_channels=out_classes,
        img_size=img_size,
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

    criterion = SoftDiceCrossEntropyLoss2D(
        ce_weight=0.5,
        dice_weight=0.5,
        ignore_bg=False,
    ).to(device)

    best_mean_hec_dice = -1.0
    epochs_without_improvement = 0
    history = []
    writer = SummaryWriter(log_dir='runs/exp1_vit_unetseg_2d_hec')

    for epoch in range(epochs):
        train_loss = train_fn_exp1_2d(
            train_loader,
            model,
            optimizer,
            device,
            criterion,
            scaler,
            use_amp=use_amp,
        )

        metrics = eval_fn_exp1_2d(
            valid_loader,
            model,
            device,
            criterion,
            out_classes=out_classes,
            use_amp=use_amp,
        )

        valid_loss = metrics['valid_loss']
        current_lr = optimizer.param_groups[0]['lr']

        if math.isnan(train_loss) or math.isnan(valid_loss):
            print(f'Early stopping at epoch {epoch + 1} because loss is NaN')
            break

        scheduler.step()

        history_row = {
            'epoch': epoch + 1,
            'lr': current_lr,
            'train_loss': train_loss,
            **metrics,
        }
        history.append(history_row)

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/valid', valid_loss, epoch)
        writer.add_scalar('Dice/mean_fg', metrics['mean_fg_dice'], epoch)
        writer.add_scalar('Dice/kidney', metrics['kidney_dice'], epoch)
        writer.add_scalar('Dice/tumor', metrics['tumor_dice'], epoch)
        writer.add_scalar('Dice/cyst', metrics['cyst_dice'], epoch)
        writer.add_scalar('HEC/kidney_and_masses_dice', metrics['kidney_and_masses_dice'], epoch)
        writer.add_scalar('HEC/masses_dice', metrics['masses_dice'], epoch)
        writer.add_scalar('HEC/tumor_dice', metrics['tumor_region_dice'], epoch)
        writer.add_scalar('HEC/mean_hec_dice', metrics['mean_hec_dice'], epoch)
        writer.add_scalar('HEC/mean_hec_iou', metrics['mean_hec_iou'], epoch)

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Train Loss: {train_loss:.6f} "
            f"Valid Loss: {valid_loss:.6f} "
            f"Mean FG Dice: {metrics['mean_fg_dice']:.6f} "
            f"Kidney: {metrics['kidney_dice']:.6f} "
            f"Tumor: {metrics['tumor_dice']:.6f} "
            f"Cyst: {metrics['cyst_dice']:.6f} "
            f"K&M Dice: {metrics['kidney_and_masses_dice']:.6f} "
            f"Masses Dice: {metrics['masses_dice']:.6f} "
            f"Tumor Dice: {metrics['tumor_region_dice']:.6f} "
            f"Mean HEC Dice: {metrics['mean_hec_dice']:.6f} "
            f"Mean HEC IoU: {metrics['mean_hec_iou']:.6f}"
        )

        checkpoint_common = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scaler': scaler.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epoch,
            'metrics': metrics,
            'img_size': img_size,
            'out_classes': out_classes,
            'feature_size': feature_size,
            'hidden_size': hidden_size,
            'mlp_dim': mlp_dim,
            'num_heads': num_heads,
            'patch_size': patch_size,
            'best_metric_name': 'mean_hec_dice',
        }

        last_path = os.path.join(save_dir, 'last_model_exp1_2d.pt')
        torch.save(checkpoint_common, last_path)

        score = metrics['mean_hec_dice']
        improved = score > best_mean_hec_dice + early_stopping_min_delta

        if improved:
            best_mean_hec_dice = score
            epochs_without_improvement = 0

            best_path = os.path.join(save_dir, 'best_model_exp1_2d.pt')
            checkpoint_best = dict(checkpoint_common)
            checkpoint_best['best_mean_hec_dice'] = best_mean_hec_dice
            torch.save(checkpoint_best, best_path)

            print(f'Model saved: {best_path}')
            print(f'Best mean HEC Dice: {best_mean_hec_dice:.6f}')
        else:
            epochs_without_improvement += 1
            print(
                f'No improvement in mean HEC Dice for '
                f'{epochs_without_improvement}/{early_stopping_patience} epochs.'
            )

        if epochs_without_improvement >= early_stopping_patience:
            print(
                f'Early stopping at epoch {epoch + 1}. '
                f'Best mean HEC Dice: {best_mean_hec_dice:.6f}'
            )
            break

        torch.cuda.empty_cache()
        gc.collect()

    writer.close()
