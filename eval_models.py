import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from pathlib import Path
from models.lcnn import LCNN
from models.lcnn_moe import LCNNMoE
from src.training_utils import ValDatasetLCNN, prefetch_test_data
from src.utils import read_yaml, seed_everything

import warnings

warnings.simplefilter("ignore", FutureWarning)
warnings.simplefilter("ignore", UserWarning)


@torch.inference_mode()
def eval_model(model, config, dataset_name, data_loader, save_path, device):
    model.eval()

    for batch_x, batch_y, utt_id in tqdm(data_loader, total=len(data_loader), desc=dataset_name):
        fname_list = []
        pred_list = []
        label_list = []

        batch_x = batch_x.to(device)

        if isinstance(model, LCNN):
            batch_logits = model(batch_x)
        else:
            batch_logits, __, __ = model(
                batch_x,
                k=config['moe']['eval']['k'],
                temperature=config['moe']['eval']['temperature'],
            )

        batch_pred = torch.softmax(batch_logits, dim=1)
        batch_score = (batch_pred[:, 0]).data.cpu().numpy().ravel()

        fname_list.extend(utt_id)
        pred_list.extend(batch_score.tolist())
        label_list.extend(batch_y.tolist())

        with open(save_path, 'a+') as fh:
            for f, pred, lab in zip(fname_list, pred_list, label_list):
                fh.write('{};{};{}\n'.format(f, pred, lab))
        fh.close()

    print('Scores saved to {}'.format(save_path))


def main(checkpoint_path, save_folder, config, device: str = 'cuda'):
    print(f"*** Evaluating model {checkpoint_path} ***")

    device = device if torch.cuda.is_available() else 'cpu'

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    if checkpoint['model_type'] == 'LCNNMoE':
        model = LCNNMoE(noisy_gating=False, **checkpoint["model_config"])
    elif checkpoint['model_type'] == 'LCNN':
        model = LCNN(**checkpoint["model_config"])
    else:
        raise ValueError('Unknown model type')

    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device)

    print('EVALUATION - Model loaded : {}'.format(os.path.basename(checkpoint_path)))

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    dataset_names, dataset_dfs = prefetch_test_data(config)

    for dataset_name, df_eval in zip(dataset_names, dataset_dfs):
        result_path = os.path.join(save_folder, dataset_name, os.path.basename(checkpoint_path).replace('.pth', '.txt'))

        if not os.path.exists(os.path.join(save_folder, dataset_name)):
            os.makedirs(os.path.join(save_folder, dataset_name))

        if os.path.exists(result_path):
            print("Save path exists - Deleting file")
            os.remove(result_path)

        d_label_eval = dict(zip(df_eval['path'], df_eval['label']))
        file_eval = list(df_eval['path'])

        # As of now, evaluation uses ValDatasetLCNN: it is performed on the middle window of each test file
        eval_set = ValDatasetLCNN(list_IDs=file_eval, labels=d_label_eval, win_len=config['win_len'],
                                  return_trackID=True)
        eval_loader = DataLoader(
            eval_set,
            batch_size=config['batch_size'],
            shuffle=False,
            drop_last=False,
            num_workers=config['num_thread'],
            prefetch_factor=config['prefetch_factor'],
            persistent_workers=True
        )

        eval_model(model, config, dataset_name, eval_loader, result_path, device)


if __name__ == '__main__':
    config = read_yaml('config/lcnn_config.yaml')

    seed_everything(config['seed'])

    save_model_folder = Path(config['save_model_folder'])

    model_list = [p.name for p in save_model_folder.glob('*.pth')]

    for model_name in model_list:
        model_path = save_model_folder.joinpath(model_name)
        main(model_path, config['save_results_folder'], config)
