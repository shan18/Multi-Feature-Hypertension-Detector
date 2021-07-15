import os
import json
import random
import argparse
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

from utils import ProgressBar
from dataset import PhysioBank
from model import HypertensionDetectorBiLSTM, HypertensionDetectorConvGRU, count_parameters


def create_dataset(data_json, train_batch_size, val_batch_size, test_batch_size):
    dataset = PhysioBank(
        data_json,
        train_batch_size=train_batch_size,
        val_batch_size=val_batch_size,
        test_batch_size=test_batch_size,
        cuda=torch.cuda.is_available()
    )

    train_loader = dataset.loader(type='train')
    val_loader = dataset.loader(type='val')
    test_loader = dataset.loader(type='test')

    return dataset, train_loader, val_loader, test_loader


def create_model(
    model_type, hidden_dim, seq_meta_length, n_layers, cnn_feature_dim=64, dropout=0.1, device='cpu'
):
    if model_type == 'bilstm':
        model = HypertensionDetectorBiLSTM(
            hidden_dim, seq_meta_length, n_layers, dropout, device
        ).to(device)
    else:
        model = HypertensionDetectorConvGRU(
            cnn_feature_dim, hidden_dim, seq_meta_length, n_layers, dropout
        ).to(device)
    
    print(f'The model has {count_parameters(model):,} trainable parameters')
    
    return model


def train(model, loader, optimizer, criterion, device):
    model.train()
    pbar = ProgressBar(target=len(loader), width=8)
    correct = 0
    processed = 0

    for batch_idx, data in enumerate(loader, 0):
        (source, source_meta), target  = data
        source = source.to(device)
        source_meta = source_meta.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        output = model(source, source_meta)

        loss = criterion(output, target)
        loss.backward()

        optimizer.step()

        pred = (output > 0.5).float()
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(target)
        accuracy = 100 * correct / processed

        pbar.update(batch_idx, values=[
            ('Loss', round(loss.item(), 2)), ('Accuracy', round(accuracy, 2))
        ])
    
    pbar.add(1, values=[
        ('Loss', round(loss.item(), 2)), ('Accuracy', round(accuracy, 2))
    ])


def eval(model, loader, criterion, device, type='val'):
    model.eval()
    correct = 0
    loss = 0

    with torch.no_grad():
        for (source, source_meta), target in loader:
            source = source.to(device)
            source_meta = source_meta.to(device)
            target = target.to(device)

            output = model(source, source_meta)

            cost = criterion(output, target)
            loss += cost.item()

            pred = (output > 0.5).float()
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    loss /= len(loader)
    accuracy = correct / len(loader)
    print(
        f'{"Validation" if type == "val" else "Test"} set: Average loss: {loss:.4f}, Accuracy: {accuracy:.2f}%\n'
    )

    return accuracy


def training_epoch(model, train_loader, val_loader, optimizer, criterion, epochs, device='cpu'):
    best_val_accuracy = 0

    for epoch in range(1, epochs + 1):
        print(f'Epoch {epoch}:')
        train(model, train_loader, optimizer, criterion, device)
        accuracy = eval(model, val_loader, criterion, device)

        if accuracy > best_val_accuracy:
            print(f'Validation accuracy improved from {best_val_accuracy:.2f}% to {accuracy:.2f}%\n')
            best_val_accuracy = accuracy
            torch.save(model.state_dict(), 'hypertension_detector.pt')


if __name__ == '__main__':
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data', default=os.path.join(BASE_DIR, 'data', 'physiobank_dataset.json'),
        help='Path to dataset json file'
    )
    parser.add_argument('--train_batch_size', type=int, default=128, help='Training batch size')
    parser.add_argument('--val_batch_size', type=int, default=128, help='Validation batch size')
    parser.add_argument('--test_batch_size', type=int, default=128, help='Test batch size')
    parser.add_argument(
        '--model', required=True, choices=['bilstm', 'cnn_gru'],
        help='Type of model to run training on'
    )
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--rnn_dim', type=int, default=float, help='Hidden dimension of RNN')
    parser.add_argument('--cnn_dim', type=int, default=128, help='Output dimenstion of CNN')
    parser.add_argument('--n_layers', type=int, default=2, help='Number of RNN layers')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    args = parser.parse_args()

    # Initialize random seed
    random.seed(0)
    torch.manual_seed(0)

    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create dataset
    dataset, train_loader, val_loader, test_loader = create_dataset(
        args.data_json, args.train_batch_size, args.val_batch_size, args.test_batch_size
    )

    # Create model
    model = create_model(
        args.model,
        args.rnn_dim,
        dataset.train_data[0][0][1].shape[0],
        args.n_layers,
        args.cnn_dim,
        args.dropout,
        device
    )

    # Create optimizer and criterion
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.BCEWithLogitsLoss().to(device)

    # Train model
    training_epoch(
        model, train_loader, val_loader, optimizer, criterion, args.epochs, device
    )

    # Test model
    eval(model, test_loader, criterion, device, type='test')
