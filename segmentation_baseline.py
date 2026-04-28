import os
import gc
import math
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from models.UNet3D import UNet3D
from losses.SoftDiceCrossEntropyLoss import SoftDiceCrossEntropyLoss3D


def segmentation_baseline_3d(
    train_loader,
    valid_loader,
    device,
    epochs,
    lr,
    out_classes,
    save_dir="saved_UNet3D_model",
):
    os.makedirs(save_dir, exist_ok=True)

    model = UNet3D(
        in_channels=1,
        out_classes=out_classes,
        base_channels=16,
    ).to(device)

    def count_parameters(model):
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"Total params: {total:,}")
        print(f"Trainable params: {trainable:,}")

    count_parameters(model)

    scaler = torch.amp.GradScaler(device=device)

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

    best_valid_loss = np.inf
    writer = SummaryWriter(log_dir="runs/exp0_3d_unet")

    for epoch in range(epochs):
        train_loss = train_fn(
            train_loader,
            model,
            optimizer,
            device,
            criterion,
            scaler,
        )

        valid_loss = eval_fn(
            valid_loader,
            model,
            device,
            criterion,
        )

        if math.isnan(train_loss) or math.isnan(valid_loss):
            print(f"Early stopping at epoch {epoch + 1} because loss is NaN")
            break

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss

            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch,
                    "best_valid_loss": best_valid_loss,
                },
                os.path.join(save_dir, "best_model1.pt"),
            )

            print("Model saved")

        scheduler.step()

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/valid", valid_loss, epoch)

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Train Loss: {train_loss:.6f} "
            f"Valid Loss: {valid_loss:.6f}"
        )

        torch.cuda.empty_cache()
        gc.collect()

    writer.close()


def train_fn(loader, model, optimizer, device, criterion, scaler):
    model.train()
    total_loss = 0.0

    for images, labels in tqdm(loader):
        images = images.to(device)   # [B, 1, D, H, W]
        labels = labels.to(device)   # [B, D, H, W]

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type="cuda"):
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    return total_loss / len(loader)

def eval_fn(loader, model, device, criterion):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for images, labels in tqdm(loader):
            images = images.to(device)
            labels = labels.to(device)

            with torch.amp.autocast(device_type="cuda"):
                logits = model(images)
                loss = criterion(logits, labels)

            total_loss += loss.item()

    return total_loss / len(loader)