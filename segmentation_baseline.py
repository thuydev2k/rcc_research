import torch
import numpy as np
import math
import gc
import torch.nn.functional as F

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from models.UNet import UNet
from losses.SoftDiceCrossEntropyLoss import SoftDiceCrossEntropyLoss
    
def segmentation_baseline(train_loader, valid_loader, device, epoch, lr, out_classes, stop_training=False):
    model = UNet(out_classes=out_classes).to(device)
    # use amp to accelerate training => mixed float16, float32
    scaler = torch.amp.GradScaler(device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    # learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=lr * 0.01)

    criterion = SoftDiceCrossEntropyLoss().to(device)

    start_epoch = 0
    best_valid_loss = np.inf

    if stop_training:
        start_epoch = 6
        model, optimizer, lr_scheduler, scaler = set_weights(model, optimizer, scheduler, scaler, device)
        best_valid_loss = 0.3

    writer = SummaryWriter()

    # train model
    save_check = 0
    for i in range(start_epoch, epoch):
        if save_check > 20:
            print('Early stopping at epoch {}'.format(i))
            break

        train_loss = train_fn(train_loader, model, optimizer, device, criterion, scaler)
        valid_loss = eval_fn(valid_loader, model, device, criterion)

        if math.isnan(valid_loss) or math.isnan(train_loss):
            print('Early stopping at epoch {} by nan'.format(i))
            break

        if valid_loss < best_valid_loss:
            save_check = 0
            best_valid_loss = valid_loss
            torch.save({'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scaler': scaler.state_dict(),
                        'lrscheduler': scheduler.state_dict(),
                        }, f'saved_model/best_model.pt')
            print('Model Saved')
        else:
            save_check += 1

        scheduler.step()

        torch.cuda.empty_cache()
        gc.collect()

        writer.add_scalar(f"/Loss/train_total", train_loss, i)
        writer.add_scalar(f"/Loss/valid_total", valid_loss, i)

        print(f'EPOCH {i + 1} : train loss : {train_loss}, valid loss : {valid_loss}')

    writer.flush()
    writer.close()

def train_fn(loader, model, optimizer, device, criterion, scaler):
    model.train()

    total_loss = 0.0

    for images, labels in tqdm(loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        with torch.amp.autocast(device_type=device):
            seg_out = model(images)

            num_classes = seg_out.shape[1]

            lbl_onehot = F.one_hot(labels.long(), num_classes=num_classes)
            lbl_onehot = lbl_onehot.permute(0, 3, 1, 2).to(dtype=seg_out.dtype, device=seg_out.device)

            loss = criterion(seg_out, lbl_onehot)

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
            
            with torch.amp.autocast(device_type=device):
                predicted = model(images)

                num_classes = predicted.shape[1]

                lbl_onehot = F.one_hot(labels.long(), num_classes=num_classes)
                lbl_onehot = lbl_onehot.permute(0, 3, 1, 2).to(dtype=predicted.dtype, device=predicted.device)

                loss = criterion(predicted, lbl_onehot)

            total_loss += loss.item()

    return total_loss / len(loader)

def set_weights(model, optimizer, lr_scheduler, scaler, device):
    saved_model = torch.load(f'./saved_model/best_model.pt', map_location=device)
    model.load_state_dict(saved_model['model'])
    optimizer.load_state_dict(saved_model['optimizer'])
    lr_scheduler.load_state_dict(saved_model['lrscheduler'])
    scaler.load_state_dict(saved_model['scaler'])

    return model, optimizer, lr_scheduler, scaler
