import os
from collections import defaultdict
import glob
import shutil


def select_best_model(filenames):
    best_models = defaultdict(lambda: ('', float('-inf')))

    for filename in filenames:

        parts = filename.split('-')
        b_number = parts[6]
        val_s_score = float(parts[-1].split('=')[-1][:-5])

        print(b_number)

        if val_s_score > best_models[b_number][1]:
            best_models[b_number] = (filename, val_s_score)

    return [filename for filename, score in best_models.values()]


ckpts = glob.glob('/usr/mvl2/esdft/transformer_based_retinal_classification/retinal-disease-classification_swin/weights-vitb8224-init-update-itdfhnorm/checkpoints/*.ckpt')

filenames = [os.path.basename(name) for name in ckpts]


'''filenames = [
    'boosting-vitb8224-init-update-itdfhnorm-224x224-b00-epoch=009-val_s_score=0.8682.ckpt',
    'boosting-vitb8224-init-update-itdfhnorm-224x224-b00-epoch=011-val_s_score=0.8801.ckpt',
    'boosting-vitb8224-init-update-itdfhnorm-224x224-b01-epoch=009-val_s_score=0.8943.ckpt',
    'boosting-vitb8224-init-update-itdfhnorm-224x224-b01-epoch=010-val_s_score=0.8867.ckpt',
]'''


dst_path = '/usr/mvl2/esdft/transformer_based_retinal_classification/retinal-disease-classification_swin/best_ckpt/'
src_path = '/usr/mvl2/esdft/transformer_based_retinal_classification/retinal-disease-classification_swin/weights-vitb8224-init-update-itdfhnorm/checkpoints/'


best_models = select_best_model(filenames)
print("Best models:")
for model in best_models:
    shutil.copy(f'{src_path}/{model}', dst_path)


