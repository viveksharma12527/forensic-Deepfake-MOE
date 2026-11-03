import os
import argparse
import warnings
import torch
import torch.nn as nn
from tqdm import tqdm
from src.utils import EpochMetrics
from torch.utils.data import DataLoader
from models.lcnn_moe import LCNNMoE
from models.moe_layer import MoELayer
from src.training_utils import prefetch_training_data, TrainDatasetLCNN, ValDatasetLCNN, validate_dataset_selection
from src.utils import read_yaml, init_weights, seed_everything
from src.schedulers import TopKScheduler, TemperatureScheduler


def train_epoch(data_loader, model, optim, criterion, device,
                k: int = 1,
                temperature: float = 1.0,
                lambda_importance: float = 1.0,
                lambda_load: float = 1.0,
                log_interval: int = 24):
    metrics = EpochMetrics(device, num_losses=4)
    model.train()

    pbar = tqdm(data_loader, total=len(data_loader))
    for step, (batch_x, batch_y) in enumerate(pbar, start=1):
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.view(-1).long().to(device, non_blocking=True)

        optim.zero_grad(set_to_none=True)
        batch_logits, importance_loss, load_loss = model(batch_x, k=k, temperature=temperature)
        criterion_loss = criterion(batch_logits, batch_y)
        batch_loss = criterion_loss + lambda_importance * importance_loss + lambda_load * load_loss

        batch_loss.backward()
        optim.step()

        metrics.update(batch_logits, batch_y, batch_loss, criterion_loss, importance_loss, load_loss)

        if step % log_interval == 0 or step == len(data_loader):
            losses, accuracy = metrics.compute()
            pbar.set_description(
                f'running_loss: {losses[0]:.4f} - '
                f'entropy_loss: {losses[1]:.4f} - '
                f'importance_loss: {losses[2]:.4f} - '
                f'load_loss: {losses[3]:.4f} - '
                f'train_acc: {accuracy:.4f}'
            )

    losses, accuracy = metrics.compute()
    epoch_total_loss = losses[0]

    return epoch_total_loss, accuracy


@torch.inference_mode()
def valid_epoch(data_loader, model, criterion, device,
                k: int = 1,
                temperature: float = 1.0,
                log_interval: int = 24):
    metrics = EpochMetrics(device, num_losses=3)
    model.eval()

    pbar = tqdm(data_loader, total=len(data_loader))
    for step, (batch_x, batch_y) in enumerate(pbar, start=1):
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.view(-1).long().to(device, non_blocking=True)

        batch_logits, importance_loss, load_loss = model(batch_x, k=k, temperature=temperature)
        criterion_loss = criterion(batch_logits, batch_y)

        metrics.update(batch_logits, batch_y, criterion_loss, importance_loss, load_loss)

        if step % log_interval == 0 or step == len(data_loader):
            losses, accuracy = metrics.compute()
            pbar.set_description(
                f'entropy_loss: {losses[0]:.4f} - '
                f'importance_loss: {losses[1]:.4f} - '
                f'load_loss: {losses[2]:.4f} - '
                f'val_acc: {accuracy:.4f}'
            )

    losses, accuracy = metrics.compute()
    epoch_criterion_loss = losses[0]

    return epoch_criterion_loss, accuracy


def main(config):
    assert config['num_thread'] > 0, 'num_thread must be greater than 0'

    device = config['device'] if torch.cuda.is_available() else 'cpu'
    pin_memory = torch.cuda.is_available()

    if not os.path.exists(config['save_model_folder']):
        os.makedirs(config['save_model_folder'])

    model_config = config['moe']['architecture']

    model = LCNNMoE(noisy_gating=True, **model_config)
    model = model.to(device)

    # TODO: possibly load a pretrained LCNN and implement explicit mapping into MoE experts
    # load pretrained model...
    # pretrained_model_path = os.path.join(config['save_model_folder'], config['model_pretrained'])
    # map each layer to the corresponding MoE layer...
    # start training from from learned weights instead of random weights...

    # Initialize expert layers without touching their gating networks. Do not use model.apply(init_weights)
    for module in model.modules():
        if isinstance(module, MoELayer):
            for expert in module.experts:
                expert.apply(init_weights)

    # Initialize the final shared classifier.
    model.fc32.apply(init_weights)

    num_epochs = config['num_epochs']
    top_k_scheduler = TopKScheduler(num_epochs=num_epochs)
    temperature_scheduler = TemperatureScheduler(
        min_temperature=config['moe']['train']['min_temperature'],
        max_temperature=config['moe']['train']['max_temperature'],
    )

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

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')

        k = top_k_scheduler(epoch)
        temperature = temperature_scheduler(epoch)

        if early_stopping < config['early_stopping']:
            running_loss, train_acc = train_epoch(
                train_loader, model, optimizer, criterion, device,
                k=k,
                temperature=temperature,
                lambda_importance=config['moe']['train']['lambda_importance'],
                lambda_load=config['moe']['train']['lambda_load']
            )

            val_loss, val_acc = valid_epoch(
                dev_loader, model, criterion, device,
                k=config['moe']['val']['k'],
                temperature=config['moe']['val']['temperature']
            )

            if scheduler is not None:
                scheduler.step()

            print(
                "==========================================================================================\n"
                f'Epoch: {epoch} - Train Loss: {running_loss:.5f} - Val Loss: {val_loss:.5f} '
                f'- Train Acc: {train_acc:.2f} - Val Acc: {val_acc:.2f}\n'
                "=========================================================================================="
            )

            if val_loss < best_loss:
                print(f'> Best model found at epoch {epoch + 1}\n')

                torch.save(
                    {"model_type": "LCNNMoE",
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

    model_name = "MoE-LCNN_kschedule_" + "+".join(training_datasets) + ".pth"
    print(f"Model name: {model_name}")

    config['model_name'] = model_name
    config['device'] = args.device

    if config['device'] == 'cpu' or not torch.cuda.is_available():
        warnings.warn("You are training on CPU. This will be slow!", UserWarning)

    main(config)
