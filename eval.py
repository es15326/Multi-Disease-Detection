import argparse
import glob
import os
from typing import Literal

import numpy as np
import torch
import tqdm
import ttach as tta
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

import data
import nets


def load_tta_model(ckpt_path, arch, use_tta=False):
  model = nets.load_model(arch)
  ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
  new_dict = {
      k.replace('vit.', 'model.'): v
      for k, v in ckpt['state_dict'].items()
  }
  model.load_state_dict(new_dict)
  model.eval()
  model.cuda()
  if use_tta:
    model = tta.ClassificationTTAWrapper(model, tta.Compose([
        tta.Rotate90(angles=[0, 90, 180, 270]),
    ]))
  return model

def aggregate_predictions(
    outs, method: Literal['average', 'heuristic'] = 'average',
    heuristic_threshold: float = 0.5
) -> np.ndarray:
  if method == 'average':
    return np.mean(outs, axis=2)
  else:
    minval = np.min(outs, axis=2)
    maxval = np.max(outs, axis=2)
    mean = np.mean(outs, axis=2)
    condition1 = (maxval - mean) > heuristic_threshold
    condition2 = (mean - minval) > heuristic_threshold
    return np.where(condition1, minval, np.where(condition2, maxval, mean))


def main(args):
  MODELS_DIR = args.models_dir
  TEST_IMG_PATH = args.test_img_path
  TEST_CSV = args.test_csv
  ARCH = args.arch
  AGGREGATION = args.aggregation
  INPUT_RESOLUTION = args.input_resolution
  OUT_FILE = args.out_file
  
  models = sorted(glob.glob(os.path.join(MODELS_DIR, '*.ckpt')))
  if len(models) == 0:
    raise ValueError(f'No models found in {MODELS_DIR}')
  models = [load_tta_model(m, ARCH) for m in models]

  valset = data.RIADDDataset(TEST_CSV, TEST_IMG_PATH, testing=True,
                             input_size=INPUT_RESOLUTION)
  N = len(valset)
  batch_size = 4
  num_workers = min(os.cpu_count() - 1, batch_size)
  dataloader = DataLoader(valset, batch_size=batch_size, shuffle=False, 
                          num_workers=num_workers)

  outs = np.zeros((N, 29, len(models)))
  labels = np.zeros((N, 29))
  with torch.no_grad():
    for n in range(len(models)):
      for i, (imgs, label) in enumerate(tqdm.tqdm(dataloader)):
        idx = i * batch_size
        imgs = imgs.cuda()
        out = models[n](imgs).detach().cpu().numpy()
        outs[idx:idx + len(out),:, n] = out
        if n ==0:
          labels[idx:idx + len(label),:]  = label.detach().cpu().numpy()
      
  sig = torch.nn.Sigmoid()
  outs = sig(torch.tensor(outs)).numpy()
  np.save(os.path.join(MODELS_DIR, OUT_FILE), outs)
  average = aggregate_predictions(outs, method=AGGREGATION)

  auc1 = roc_auc_score(labels[:, 0], average[:, 0])
  print(f'AUC of Challenge 1: {auc1:.4f}')

  auc2 = roc_auc_score(labels[:, 1:], average[:, 1:])
  print(f'AUC of Challenge 2: {auc2:.4f}')

  mAP = average_precision_score(labels[:, 1:], average[:, 1:])
  print(f'mAP of Challenge 2: {mAP:.4f}')

  C1_Score = auc1
  C2_Score = mAP * 0.5 + auc2 * 0.5
  final_Score =  C2_Score * 0.5 + C1_Score * 0.5
  print(f'S1 Score: {C1_Score:.4f} S2 Score: {C2_Score:.4f} '
        f'S Final Score: {final_Score:.4f}')


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Evaluate ensemble models')
  parser.add_argument('--models-dir', type=str, required=True,
                      help='Directory containing the models')
  parser.add_argument('--test-img-path', type=str, required=True,
                      help='Path to the test images')
  parser.add_argument('--test-csv', type=str, required=True,
                      help='Path to the test ground-truth csv')
  parser.add_argument('--arch', type=str, default='resnext',
                      choices=['efficientnet', 'resnext', 'vit-b', 'vit-l', 'vit-h', 'clip-l', 'vits8', 'vits16', 'vitb8', 'vitb16'],
                      help='Architecture of the models')
  parser.add_argument('--aggregation', type=str, default='heuristic',
                      choices=['average', 'heuristic'],
                      help='Aggregation method')
  parser.add_argument('--input-resolution', type=int, default=224,
                      help='Input resolution')
  parser.add_argument('--out_file', type=str, default='outs.npy',
                      help='Output file name')
  args = parser.parse_args()
  main(args)
