import os
import argparse
import warnings
import torch
import torch.nn as nn
from tqdm import tqdm
from src.utils import EpochMetrics
from torch.utils.data import DataLoader
from models.lcnn import LCNN
from src.training_utils import prefetch_training_data, TrainDatasetLCNN, ValDatasetLCNN, validate_dataset_selection
from src.utils import read_yaml, init_weights, seed_everything


def train_epoch(data_loader, model, optim, criterion, device, log_interval=24):
    metrics = EpochMetrics(device, num_losses=1)
    model.train()

    pbar = tqdm(data_loader, total=len(data_loader))
    for step, (batch_x, batch_y) in enumerate(pbar, start=1):
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.view(-1).long().to(device, non_blocking=True)

        optim.zero_grad(set_to_none=True)
        batch_logits = model(batch_x)
        batch_loss = criterion(batch_logits, batch_y)

        batch_loss.backward()
        optim.step()

        metrics.update(batch_logits, batch_y, batch_loss)

        if step % log_interval == 0 or step == len(data_loader):
            losses, accuracy = metrics.compute()
            pbar.set_description(
                f'train_loss: {losses[0]:.4f} - '
                f'train_acc: {accuracy:.4f}'
            )

    losses, accuracy = metrics.compute()
    return losses[0], accuracy


@torch.inference_mode()
def valid_epoch(data_loader, model, criterion, device, log_interval=24):
    metrics = EpochMetrics(device, num_losses=1)
    model.eval()

    pbar = tqdm(data_loader, total=len(data_loader))
    for step, (batch_x, batch_y) in enumerate(pbar, start=1):
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.view(-1).long().to(device, non_blocking=True)

        batch_logits = model(batch_x)
        batch_loss = criterion(batch_logits, batch_y)

        metrics.update(batch_logits, batch_y, batch_loss)

        if step % log_interval == 0 or step == len(data_loader):
            losses, accuracy = metrics.compute()
            pbar.set_description(
                f'val_loss: {losses[0]:.4f} - '
                f'val_acc: {accuracy:.4f}'
            )

    losses, accuracy = metrics.compute()
    return losses[0], accuracy


def main(config):
    assert config['num_thread'] > 0, 'num_thread must be greater than 0'

    device = config['device'] if torch.cuda.is_available() else 'cpu'
    pin_memory = torch.cuda.is_available()

    if not os.path.exists(config['save_model_folder']):
        os.makedirs(config['save_model_folder'])

    model_config = config['lcnn']['architecture']

    model = LCNN(**model_config)
    model = model.to(device)
    model.apply(init_weights)

    pretrained_model_path = os.path.join(config['save_model_folder'], config['model_pretrained'])
    if os.path.exists(pretrained_model_path):
        checkpoint = torch.load(pretrained_model_path, map_location=device, weights_only=True)
        if checkpoint.get("model_type") == "LCNN":
            model.load_state_dict(checkpoint["state_dict"])
            print(f'Model loaded: {config["model_pretrained"]}')
        else:
            print("No pretrained model loaded: checkpoint is not an LCNN.")
    else:
        print("No pretrained model loaded: checkpoint file not found.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['T_max'], eta_min=config['eta_min'])
    criterion = nn.CrossEntropyLoss(label_smoothing=0.2)

    file_train, d_label_trn, file_dev, d_label_dev, __, __ = prefetch_training_data(config)

    # TrainDatasetLCNN extracts random windows from train files
    train_set = TrainDatasetLCNN(list_IDs=file_train, labels=d_label_trn, win_len=config['win_len'])
    train_loader = DataLoader(
        train_set,
        batch_size=config['batch_size'],
        shuffle=True,
        drop_last=True,
        num_workers=config['num_thread'],
        prefetch_factor=config['prefetch_factor'],
        persistent_workers=True,
        pin_memory=pin_memory
    )

    # ValDatasetLCNN loads the middle window of every dev files
    dev_set = ValDatasetLCNN(list_IDs=file_dev, labels=d_label_dev, win_len=config['win_len'])
    dev_loader = DataLoader(
        dev_set,
        batch_size=config['batch_size'],
        shuffle=False,
        drop_last=False,
        num_workers=config['num_thread'],
        prefetch_factor=config['prefetch_factor'],
        persistent_workers=True,
        pin_memory=pin_memory
    )

    # Start training and validation
    best_acc = 0
    best_loss = float("inf")
    early_stopping = 0
    num_epochs = config['num_epochs']

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')

        if early_stopping < config['early_stopping']:
            running_loss, train_acc = train_epoch(train_loader, model, optimizer, criterion, device)
            val_loss, val_acc = valid_epoch(dev_loader, model, criterion, device)

            if scheduler is not None:
                scheduler.step()

            print(
                "==========================================================================================\n"
                f'Epoch: {epoch + 1} - Train Loss: {running_loss:.5f} - Val Loss: {val_loss:.5f} '
                f'- Train Acc: {train_acc:.2f} - Val Acc: {val_acc:.2f}\n'
                "=========================================================================================="
            )

            if val_loss < best_loss:
                print(f'> Best model found at epoch {epoch + 1}\n')

                torch.save(
                    {"model_type": "LCNN",
                     "model_config": model_config,
                     "state_dict": model.state_dict()},
                    os.path.join(config['save_model_folder'], config['model_name'])
                )

                best_loss = val_loss
                best_acc = val_acc
                early_stopping = 0
            else:
                early_stopping += 1
                print()
        else:
            print(f'Training stopped after {epoch + 1} epochs - Best Val Acc {best_acc:.2f}')
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train LCNN model")

    parser.add_argument(
        "-d", "--device",
        type=str,
        default="cpu",
        help="Device to use: cpu, cuda, cuda:0, cuda:1, etc."
    )

    args = parser.parse_args()

    config = read_yaml('config/lcnn_config.yaml')

    seed_everything(config['seed'])

    training_datasets = validate_dataset_selection(config, 'training_datasets')

    model_name = "LCNN_" + "+".join(training_datasets) + ".pth"
    print(f"Model name: {model_name}")

    config['model_name'] = model_name
    config['device'] = args.device

    if config['device'] == 'cpu' or not torch.cuda.is_available():
        warnings.warn("You are training on CPU. This will be slow!", UserWarning)

    main(config)
