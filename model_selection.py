import itertools

import numpy as np
import pandas as pd
import tqdm
from sklearn.metrics import average_precision_score, roc_auc_score

VERBOSE = False


testing_csv_path = ('/cluster/VAST/civalab/public_datasets/RIADD-RFMiD/'
                    'Validation_Set/RFMiD_Validation_Labels.csv')
testing_df = pd.read_csv(testing_csv_path)
testing_arr = testing_df.to_numpy()[:, 1:]
labels = np.concatenate([
  testing_arr[:, :28],
  testing_arr[:, 28:].max(axis=1, keepdims=True)
], axis=1)

all_outs = np.load('weights-resnext/checkpoints/best/val_outs.npy')
best_s = 0
best_params = {}
possibilities = list(itertools.product([True, False], repeat=10))
iterator = tqdm.tqdm(possibilities)
for selected in iterator:
  # Base model + combination of models
  selected = np.array([True, *selected])
  
  for threshold in np.linspace(0.1, 0.9, 9):
    if VERBOSE: print(f'{threshold:-^80}')
    outs = all_outs[..., selected]

    minval = np.min(outs, axis=2)
    maxval = np.max(outs, axis=2)
    mean = np.mean(outs, axis=2)
    condition1 = maxval - mean > threshold
    condition2 = mean - minval > threshold
    average = np.where(condition1, minval, np.where(condition2, maxval, mean))


    auc1 = roc_auc_score(labels[:,0], average[:,0])
    if VERBOSE: print(f'AUC of Challenge 1: {auc1:.4f}')

    auc2 = roc_auc_score(labels[:,1:], average[:,1:])
    if VERBOSE: print(f'AUC of Challenge 2: {auc2:.4f}')

    mAP = average_precision_score(labels[:,1:], average[:,1:])
    if VERBOSE: print(f'mAP of Challenge 2: {mAP:.4f}')

    C1_Score = auc1
    C2_Score = mAP * 0.5 + auc2 * 0.5
    final_Score =  C2_Score * 0.5 + C1_Score * 0.5

    if VERBOSE: print(f'C1 Score: {C1_Score:.4f} C2 Score: {C2_Score:.4f} '
                      f'Final Score: {final_Score:.4f}')
    if final_Score > best_s:
      best_s = final_Score
      best_params = {
        'models': np.where(selected),
        'threshold': threshold,
      }
    iterator.set_description(f'{best_s=:.4f}')

print(best_s)
print(best_params)
