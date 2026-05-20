import os
import pandas as pd
import matplotlib.pyplot as plt


def _plot_lines(df, columns, title, ylabel, save_path):
    plt.figure(figsize=(10, 6))
    for col in columns:
        if col in df.columns:
            plt.plot(df['epoch'], df[col], label=col)
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def save_and_plot_training_history(history, save_dir, csv_name='training_history_exp1_2d.csv'):
    os.makedirs(save_dir, exist_ok=True)
    plot_dir = os.path.join(save_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)

    df = pd.DataFrame(history)
    csv_path = os.path.join(save_dir, csv_name)
    df.to_csv(csv_path, index=False)

    _plot_lines(
        df,
        ['train_loss', 'valid_loss'],
        'Train/Valid Loss',
        'Loss',
        os.path.join(plot_dir, 'loss_curve.png'),
    )

    _plot_lines(
        df,
        ['kidney_dice', 'tumor_dice', 'cyst_dice', 'mean_fg_dice'],
        'Per-class Dice',
        'Dice',
        os.path.join(plot_dir, 'class_dice_curve.png'),
    )

    _plot_lines(
        df,
        ['kidney_and_masses_dice', 'masses_dice', 'tumor_region_dice', 'mean_hec_dice'],
        'HEC Dice',
        'Dice',
        os.path.join(plot_dir, 'hec_dice_curve.png'),
    )

    _plot_lines(
        df,
        ['kidney_and_masses_iou', 'masses_iou', 'tumor_region_iou', 'mean_hec_iou'],
        'HEC IoU',
        'IoU',
        os.path.join(plot_dir, 'hec_iou_curve.png'),
    )

    _plot_lines(
        df,
        ['lr'],
        'Learning Rate',
        'LR',
        os.path.join(plot_dir, 'lr_curve.png'),
    )

    return csv_path, plot_dir
