import os
import torch
import random
import pandas as pd
from glob import glob
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from .utils import read_audio, fix_length


def validate_dataset_selection(config, key):
    """Return a non-empty list of unique string dataset names."""
    if key not in config:
        raise ValueError(f"Missing required configuration key: {key}")
    names = config[key]
    if not isinstance(names, list) or not names:
        raise ValueError(f"{key} must be a non-empty list of dataset names")
    seen = set()
    for name in names:
        if not isinstance(name, str):
            raise ValueError(f"Dataset entries in {key} must be strings, got {name!r}")
        if name in seen:
            raise ValueError(f"Duplicate dataset {name!r} in {key}")
        seen.add(name)
    return names


def load_asvspoof_data(file_name, base_path, label_dir):
    df = pd.read_csv(os.path.join(label_dir, file_name), sep=' ', header=None)
    df['label'] = df[4].map({'bonafide': 0, 'spoof': 1})
    df['path'] = df[1].apply(lambda x: os.path.join(base_path, x + '.flac'))
    return df[['path', 'label']]


def load_for_data(base_dir, subset):
    real_files = glob(os.path.join(base_dir, subset, 'real', '*.wav'))
    fake_files = glob(os.path.join(base_dir, subset, 'fake', '*.wav'))

    df_real = pd.DataFrame(real_files, columns=['path'])
    df_real['label'] = 0

    df_fake = pd.DataFrame(fake_files, columns=['path'])
    df_fake['label'] = 1

    return pd.concat([df_real, df_fake])


def prefetch_training_data(config):
    dataset_names = validate_dataset_selection(config, 'training_datasets')
    loaded_datasets = set()
    # LOAD DATASETS
    df_train_list = []
    df_dev_list = []
    df_eval_list = []

    # ====== ASVSPOOF 2019 DATASET ====== #
    if 'asvspoof19' in dataset_names:
        label_dir = f'{config["asvspoof2019_path"]}/LA/ASVspoof2019_LA_cm_protocols'
        base_paths = {
            'train': f'{config["asvspoof2019_path"]}/LA/ASVspoof2019_LA_train/flac/',
            'dev': f'{config["asvspoof2019_path"]}/LA/ASVspoof2019_LA_dev/flac/',
            'eval': f'{config["asvspoof2019_path"]}/LA/ASVspoof2019_LA_eval/flac/'
        }

        df_train_asv = load_asvspoof_data('ASVspoof2019.LA.cm.train.trn.txt', base_paths['train'], label_dir)
        df_dev_asv = load_asvspoof_data('ASVspoof2019.LA.cm.dev.trl.txt', base_paths['dev'], label_dir)
        df_eval_asv = load_asvspoof_data('ASVspoof2019.LA.cm.eval.trl.txt', base_paths['eval'], label_dir)

        df_train_list.append(df_train_asv)
        df_dev_list.append(df_dev_asv)
        df_eval_list.append(df_eval_asv)
        loaded_datasets.add('asvspoof19')

    # ====== FAKE OR REAL DATASET ====== #
    if 'fakeorreal' in dataset_names:
        base_dir = f'{config["for_path"]}/for-original-wav'

        df_train_for = load_for_data(base_dir, 'training')
        df_dev_for = load_for_data(base_dir, 'validation')
        df_eval_for = load_for_data(base_dir, 'testing')

        df_train_list.append(df_train_for)
        df_dev_list.append(df_dev_for)
        df_eval_list.append(df_eval_for)
        loaded_datasets.add('fakeorreal')

    # ====== ADD 2022 DATASET ====== #
    if 'add2022' in dataset_names:
        df_train_ADD = pd.read_csv(f'{config["add2022_path"]}/ADD_train_dev/label/train_label.txt', sep=' ',
                                   header=None)
        df_train_ADD['path'] = f'{config["add2022_path"]}/ADD_train_dev/train/' + df_train_ADD[0].astype(str)
        df_train_ADD['label'] = df_train_ADD[1].map({'genuine': 0, 'fake': 1})
        df_train_ADD = df_train_ADD[['path', 'label']]

        df_dev_ADD = pd.read_csv(f'{config["add2022_path"]}/ADD_train_dev/label/dev_label.txt', sep=' ',
                                 header=None)
        df_dev_ADD['path'] = f'{config["add2022_path"]}/ADD_train_dev/dev/' + df_dev_ADD[0].astype(str)
        df_dev_ADD['label'] = df_dev_ADD[1].map({'genuine': 0, 'fake': 1})
        df_dev_ADD = df_dev_ADD[['path', 'label']]

        df_eval_ADD = pd.read_csv(f'{config["add2022_path"]}/ADD2022_test/track1_label.txt', sep=' ', header=None)
        df_eval_ADD['path'] = f'{config["add2022_path"]}/ADD2022_test/track1test/' + df_eval_ADD[0].astype(str)
        df_eval_ADD['label'] = df_eval_ADD[1].map({'genuine': 0, 'fake': 1})
        df_eval_ADD = df_eval_ADD[['path', 'label']]

        df_train_list.append(df_train_ADD)
        df_dev_list.append(df_dev_ADD)
        df_eval_list.append(df_eval_ADD)
        loaded_datasets.add('add2022')

    # ====== IN THE WILD DATASET ====== #
    if 'inthewild' in dataset_names:
        df_meta = pd.read_csv(f'{config["InTheWild_path"]}/meta.csv')
        df_meta['path'] = df_meta['file'].apply(lambda x: f'{config["InTheWild_path"]}/wavs/' + x)
        df_meta['label'] = df_meta['label'].map({'spoof': 1, 'bona-fide': 0})
        df_meta = df_meta[['path', 'label']]

        df_train_itw, df_dev_itw = train_test_split(
            df_meta,
            test_size=0.4,
            stratify=df_meta['label'],
            random_state=config['seed']
        )

        df_dev_itw, df_eval_itw = train_test_split(
            df_dev_itw,
            test_size=0.5,
            stratify=df_dev_itw['label'],
            random_state=config['seed']
        )

        df_train_list.append(df_train_itw)
        df_dev_list.append(df_dev_itw)
        df_eval_list.append(df_eval_itw)
        loaded_datasets.add('inthewild')

    unsupported = [name for name in dataset_names if name not in loaded_datasets]
    if unsupported:
        raise ValueError(f"Unsupported datasets in training_datasets: {', '.join(unsupported)}")

    df_train = pd.concat(df_train_list, ignore_index=True)
    df_dev = pd.concat(df_dev_list, ignore_index=True)
    df_eval = pd.concat(df_eval_list, ignore_index=True)

    d_label_train = dict(zip(df_train['path'], df_train['label']))
    file_train = list(df_train['path'])

    d_label_dev = dict(zip(df_dev['path'], df_dev['label']))
    file_dev = list(df_dev['path'])

    d_label_eval = dict(zip(df_eval['path'], df_eval['label']))
    file_eval = list(df_eval['path'])

    return file_train, d_label_train, file_dev, d_label_dev, file_eval, d_label_eval


def prefetch_test_data(config):
    dataset_names = validate_dataset_selection(config, 'test_datasets')
    random.seed(config['seed'])
    datasets = {}

    if 'asvspoof19' in dataset_names:
        # ====== ASVSPOOF 2019 DATASET ====== #
        label_dir = f'{config["asvspoof2019_path"]}/LA/ASVspoof2019_LA_cm_protocols'
        base_paths = {
            'train': f'{config["asvspoof2019_path"]}/LA/ASVspoof2019_LA_train/flac/',
            'dev': f'{config["asvspoof2019_path"]}/LA/ASVspoof2019_LA_dev/flac/',
            'eval': f'{config["asvspoof2019_path"]}/LA/ASVspoof2019_LA_eval/flac/'
        }

        df_eval_asv19 = load_asvspoof_data('ASVspoof2019.LA.cm.eval.trl.txt', base_paths['eval'], label_dir)
        datasets['asvspoof19'] = df_eval_asv19

    if 'fakeorreal' in dataset_names:
        # ====== FAKE OR REAL DATASET ====== #
        base_dir = f'{config["for_path"]}/for-original-wav'
        df_eval_for = load_for_data(base_dir, 'testing')
        datasets['fakeorreal'] = df_eval_for

    if 'add2022' in dataset_names:
        # ====== ADD 2022 DATASET ====== #
        df_eval_ADD = pd.read_csv(f'{config["add2022_path"]}/ADD2022_test/track1_label.txt', sep=' ', header=None)
        df_eval_ADD['path'] = f'{config["add2022_path"]}/ADD2022_test/track1test/' + df_eval_ADD[0].astype(str)
        df_eval_ADD['label'] = df_eval_ADD[1].map({'genuine': 0, 'fake': 1})
        df_eval_ADD = df_eval_ADD[['path', 'label']]
        datasets['add2022'] = df_eval_ADD

    if 'inthewild' in dataset_names:
        # ====== IN THE WILD DATASET ====== #
        df_meta = pd.read_csv(f'{config["InTheWild_path"]}/meta.csv')
        df_meta['path'] = df_meta['file'].apply(lambda x: f'{config["InTheWild_path"]}/wavs/' + x)
        df_meta['label'] = df_meta['label'].map({'spoof': 1, 'bona-fide': 0})
        df_meta = df_meta[['path', 'label']]

        __, df_dev_itw = train_test_split(df_meta, test_size=0.4, stratify=df_meta['label'], random_state=config['seed'])
        __, df_eval_itw = train_test_split(df_dev_itw, test_size=0.5, stratify=df_dev_itw['label'],
                                          random_state=config['seed'])
        datasets['inthewild'] = df_eval_itw

    if 'purdue' in dataset_names:
        # ====== PURDUE DATASET ====== #
        df_fake1 = glob(f'{config["purdue_path"]}/*/*.wav')
        df_fake2 = glob(f'{config["purdue_path"]}/*/*/*.wav')

        df_lj = glob(f'{config["ljspeech_path"]}/*.wav')
        df_ls1 = glob(f'{config["librispeech_path"]}/dev-clean/*/*/*.flac')
        df_ls2 = glob(f'{config["librispeech_path"]}/train-clean-100/*/*/*.flac')
        df_ls2 = random.sample(df_ls2, 11126 - len(df_ls1))  # take the correct number of files

        df_fake = pd.DataFrame(df_fake1 + df_fake2, columns=['path'])
        df_fake['label'] = 1

        df_real = pd.DataFrame(df_lj + df_ls1 + df_ls2, columns=['path'])
        df_real['label'] = 0

        df_purdue = pd.concat([df_fake, df_real], ignore_index=True)
        datasets['purdue'] = df_purdue

    if 'timit' in dataset_names:
        # ====== TIMIT-TTS DATASET ====== #
        df_timit1 = glob(f'{config["timit_tts_path"]}/CLEAN/single_speaker/*/*.wav')
        df_timit2 = glob(f'{config["timit_tts_path"]}/CLEAN/multi_speaker/*/*/*.wav')
        df_timit_real = glob(f'{config["vidtimit_path"]}/*/audio/*.wav')

        df_fake = pd.DataFrame(df_timit1 + df_timit2, columns=['path'])
        df_fake['label'] = 1
        df_real = pd.DataFrame(df_timit_real, columns=['path'])
        df_real['label'] = 0

        df_timit = pd.concat([df_fake, df_real], ignore_index=True)
        datasets['timit'] = df_timit

    if 'mlaad' in dataset_names:
        # ====== MLAAD (ENGLISH ONLY) - M-AILABS ====== #
        fake_data = glob(os.path.join(config['MLAAD_path'], 'en/*/*.wav'))
        fake_labels = [1] * len(fake_data)

        real_data = glob(os.path.join(config['MAILABS_path'], '*/*/*/*/*/*/*.wav'))
        real_data = random.sample(real_data, len(fake_data))
        real_labels = [0] * len(real_data)

        data_all = real_data + fake_data
        labels_all = real_labels + fake_labels

        df_all_mlaad = pd.DataFrame({'path': data_all, 'label': labels_all})
        datasets['mlaad'] = df_all_mlaad

    if 'asvspoof5' in dataset_names:
        # ====== ASVSPOOF 5 ====== #
        df_asv5 = pd.read_csv(config["asvspoof5_meta"], sep=' ', header=None)
        df_asv5 = df_asv5[df_asv5[3] == '-']  # exclude codec compressions
        df_asv5 = df_asv5[df_asv5[7].isin(['bonafide','A17','A19','A21','A22','A24','A25','A26','A28','A29'])]  # exclude adv attacks
        df_asv5 = df_asv5.groupby(7, group_keys=False).apply(lambda x: x.sample(n=7760, random_state=42)).reset_index(drop=True) # keep equal number of samples (7760 is from the less represented class)
        # df_asv5 = df_asv5.groupby(7, group_keys=False).apply(lambda x: x.sample(n=5000, random_state=42)).reset_index(drop=True) # keep equal number of samples (5000 each for codec scenario)
        df_asv5['label'] = df_asv5[8].map({'bonafide': 0, 'spoof': 1})
        df_asv5['path'] = df_asv5[1].apply(lambda x: os.path.join(config["asvspoof5_path"], f'{x}.flac'))
        df_asv5 = df_asv5.reset_index(drop=True)
        datasets['asvspoof5'] = df_asv5

    # ASVspoof 2021 DF remains disabled because of corrupted files.
    unsupported = [name for name in dataset_names if name not in datasets]
    if unsupported:
        raise ValueError(f"Unsupported datasets in test_datasets: {', '.join(unsupported)}")

    dataset_dfs = [datasets[name] for name in dataset_names]

    print('> Eval dataset lengths:')
    for n, d in zip(dataset_names, dataset_dfs):
        print(f'\t{n}: {len(d)}')

    return dataset_names, dataset_dfs


class TrainDatasetLCNN(Dataset):
    def __init__(self, list_IDs, labels, win_len, fs: int = 16000):
        """
        self.list_IDs    : list of strings (each string: utt key),
        self.labels      : dictionary (key: utt key, value: label integer)
        """

        # Define training dataloader
        self.list_IDs = list_IDs
        self.labels = labels
        self.win_len = win_len
        self.fs = fs
        self.win_len_samples = int(self.win_len * self.fs)

        df = pd.DataFrame(self.labels.items(), columns=['path', 'label'])
        self.real_list = list(df[df['label'] == 0]['path'])
        self.fake_list = list(df[df['label'] == 1]['path'])

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        # Approximately balance real and fake samples in a batch
        if index % 2 == 0:
            track_name = random.choice(self.real_list)
        else:
            track_name = random.choice(self.fake_list)

        x, fs = read_audio(track_name, fs=self.fs, trim=False)
        y = self.labels[track_name]

        if len(x) < self.win_len_samples:
            x = fix_length(x, self.win_len_samples)

        last_valid_start_sample = len(x) - self.win_len_samples
        if not last_valid_start_sample == 0:
            start_sample = random.randrange(start=0, stop=last_valid_start_sample)
        else:
            start_sample = 0
        x_win = x[start_sample : start_sample + self.win_len_samples]
        x_win = torch.Tensor(x_win)

        return x_win, y


class ValDatasetLCNN(Dataset):
    def __init__(self, list_IDs, labels, win_len, fs: int = 16000, return_trackID: bool = False):
        self.list_IDs = list_IDs
        self.labels = labels
        self.win_len = win_len
        self.fs = fs
        self.win_len_samples = int(self.win_len * self.fs)
        self.return_trackID = return_trackID

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        track = self.list_IDs[index]

        x, fs = read_audio(track, fs=self.fs, trim=False)
        y = self.labels[track]

        if len(x) < self.win_len_samples:
            x = fix_length(x, self.win_len_samples)

        # Test on the middle window
        start_sample = int(0.5 * (len(x) - self.win_len_samples))
        x_win = x[start_sample: start_sample + self.win_len_samples]
        x_inp = torch.Tensor(x_win)

        if self.return_trackID:
            return x_inp, y, track

        return x_inp, y
