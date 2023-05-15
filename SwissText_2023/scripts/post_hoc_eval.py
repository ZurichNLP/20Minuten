
from pathlib import Path
import pandas as pd
import sys

def load_data(test_set):
    df = pd.read_json(test_set, lines=True)
    # tasks = df['task_prefix'].unique()
    return df

def load_generations(pred_file):
    with open(pred_file, 'r', encoding='utf8') as f:
        preds = [line.strip() for line in f]
    return preds


if __name__ == "__main__":

    exp_dir = Path(sys.argv[1])
    test_set = exp_dir / 'predict_dataset.json'
    pred_file = exp_dir / 'generated_predictions.txt'

    df = load_data(test_set)
    df['predictions'] = load_generations(pred_file)

    # load relevant metric for each task and evaluate the relevant predictions
    