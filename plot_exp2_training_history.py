import csv
import os

from utils.plot_training_history import plot_training_history


HISTORY_CSV = './saved_Exp2_ViTUNetSeg3D_MSAF_model/training_history_exp2.csv'
PLOT_DIR = './saved_Exp2_ViTUNetSeg3D_MSAF_model/plots'


def load_history_csv(csv_path):
    history = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = {}
            for key, value in row.items():
                if key == 'epoch':
                    parsed[key] = int(value)
                else:
                    parsed[key] = float(value)
            history.append(parsed)
    return history


if __name__ == '__main__':
    if not os.path.exists(HISTORY_CSV):
        raise FileNotFoundError(f'History CSV not found: {HISTORY_CSV}')

    history = load_history_csv(HISTORY_CSV)
    plot_training_history(history, PLOT_DIR)
    print(f'Saved plots to: {PLOT_DIR}')
