import csv
import os
from typing import Dict, List, Sequence

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


LOSS_KEYS = ['train_loss', 'valid_loss']
DICE_KEYS = ['mean_fg_dice', 'kidney_dice', 'tumor_dice', 'cyst_dice']
HEC_KEYS = ['kidney_and_masses_dice', 'kidney_mass_dice', 'tumor_region_dice']


def _get_epochs(history: List[Dict]):
    return [row['epoch'] for row in history]


def _available_keys(history: List[Dict], keys: Sequence[str]):
    if len(history) == 0:
        return []
    return [k for k in keys if k in history[0]]


def save_history_csv(history: List[Dict], csv_path: str):
    """
    Save training history to CSV.

    Each row is one epoch.
    Example columns:
        epoch, lr, train_loss, valid_loss, mean_fg_dice, kidney_dice, ...
    """
    if len(history) == 0:
        return

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    fieldnames = list(history[0].keys())
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def _plot_metric_group(history: List[Dict], keys: Sequence[str], title: str, ylabel: str, save_path: str):
    if len(history) == 0:
        return

    keys = _available_keys(history, keys)
    if len(keys) == 0:
        return

    epochs = _get_epochs(history)

    plt.figure(figsize=(10, 6))
    for key in keys:
        values = [row[key] for row in history]
        plt.plot(epochs, values, marker='o', linewidth=2, label=key)

    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_training_history(history: List[Dict], plot_dir: str):
    """
    Create training/validation plots.

    Saved files:
        loss_curve.png
        dice_curve.png
        hec_dice_curve.png
        lr_curve.png
    """
    if len(history) == 0:
        return

    os.makedirs(plot_dir, exist_ok=True)

    _plot_metric_group(
        history=history,
        keys=LOSS_KEYS,
        title='Exp2 Training and Validation Loss',
        ylabel='Loss',
        save_path=os.path.join(plot_dir, 'loss_curve.png'),
    )

    _plot_metric_group(
        history=history,
        keys=DICE_KEYS,
        title='Exp2 Class Dice Curves',
        ylabel='Dice',
        save_path=os.path.join(plot_dir, 'dice_curve.png'),
    )

    _plot_metric_group(
        history=history,
        keys=HEC_KEYS,
        title='Exp2 KiTS HEC Region Dice Curves',
        ylabel='Dice',
        save_path=os.path.join(plot_dir, 'hec_dice_curve.png'),
    )

    _plot_metric_group(
        history=history,
        keys=['lr'],
        title='Learning Rate Schedule',
        ylabel='Learning Rate',
        save_path=os.path.join(plot_dir, 'lr_curve.png'),
    )


def save_and_plot_training_history(history: List[Dict], save_dir: str):
    """
    Convenience wrapper used inside the training loop.
    It updates CSV and plot PNGs after every epoch.
    """
    csv_path = os.path.join(save_dir, 'training_history_exp2.csv')
    plot_dir = os.path.join(save_dir, 'plots')

    save_history_csv(history, csv_path)
    plot_training_history(history, plot_dir)
