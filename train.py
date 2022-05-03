import os
import random
import argparse
import torch
from torch import nn, optim
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from utils import ProgressBar
from dataset import PhysioBankDataset
from model import HypertensionDetectorBiLSTM, HypertensionDetectorConvGRU, count_parameters


def create_dataset(data_file, user_info_file, batch_size, num_workers):
    dataset = PhysioBankDataset(data_file, user_info_file)
    return dataset, dataset.loader(batch_size, num_workers)


def create_model(
    model_type, hidden_dim, seq_meta_length, n_layers, cnn_feature_dim, fc_dim, dropout=0.1, device='cpu'
):
    if model_type == 'bilstm':
        model = HypertensionDetectorBiLSTM(
            hidden_dim, seq_meta_length, n_layers, fc_dim, dropout, device
        ).to(device)
    else:
        model = HypertensionDetectorConvGRU(
            cnn_feature_dim, hidden_dim, fc_dim, seq_meta_length, n_layers, dropout
        ).to(device)

    print(f'The model has {count_parameters(model):,} trainable parameters')

    return model


def train(model, loader, optimizer, criterion, device):
    model.train()
    pbar = ProgressBar(target=len(loader), width=8)
    y_true = None
    y_pred = None

    for batch_idx, data in enumerate(loader, 0):
        source, source_meta, target  = data
        source = source.to(device)
        source_meta = source_meta.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        output = model(source, source_meta)

        loss = criterion(output, target)
        loss.backward()

        optimizer.step()

        pred = (output > 0.5).float()
        y_true = target if y_true is None else torch.cat((y_true, target), dim=0)
        y_pred = pred if y_pred is None else torch.cat((y_pred, pred), dim=0)

        pbar.update(batch_idx, values=[('Loss', round(loss.item(), 4))])

    y_true, y_pred = y_true.to('cpu'), y_pred.to('cpu')
    pbar.add(1, values=[
        ('Loss', round(loss.item(), 4)),
        ('Accuracy', round(accuracy_score(y_true, y_pred), 4) * 100),
        ('Precision', round(precision_score(y_true, y_pred), 4)),
        ('Recall', round(recall_score(y_true, y_pred), 4)),
        ('F1', round(f1_score(y_true, y_pred), 4))
    ])


def eval(model, loader, criterion, device, type='val'):
    model.eval()
    loss = 0
    y_true = None
    y_pred = None

    with torch.no_grad():
        for source, source_meta, target in loader:
            source = source.to(device)
            source_meta = source_meta.to(device)
            target = target.to(device)

            output = model(source, source_meta)

            cost = criterion(output, target)
            loss += cost.item()

            pred = (output > 0.5).float()
            y_true = target if y_true is None else torch.cat((y_true, target), dim=0)
            y_pred = pred if y_pred is None else torch.cat((y_pred, pred), dim=0)

    loss /= len(loader.dataset)
    y_true, y_pred = y_true.to('cpu'), y_pred.to('cpu')
    f1 = f1_score(y_true, y_pred)
    print(
        f'{"Validation" if type == "val" else "Test"} set:'
        f'Average loss: {loss:.4f},'
        f'Accuracy: {accuracy_score(y_true, y_pred) * 100:.2f}%',
        f'Precision: {precision_score(y_true, y_pred):.2f}',
        f'Recall: {recall_score(y_true, y_pred):.2f}',
        f'F1: {f1:.2f}\n'
    )

    return f1


def training_epoch(model, train_loader, val_loader, optimizer, criterion, epochs, checkpoint, device='cpu'):
    best_val_f1 = 0

    for epoch in range(1, epochs + 1):
        print(f'Epoch {epoch}:')
        train(model, train_loader, optimizer, criterion, device)
        f1 = eval(model, val_loader, criterion, device)

        if f1 > best_val_f1:
            print(f'Validation f1 improved from {best_val_f1:.2f}% to {f1:.2f}%\n')
            best_val_f1 = f1
            torch.save(model.state_dict(), checkpoint)


if __name__ == '__main__':
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train_data', default=os.path.join(BASE_DIR, 'data', 'physiobank_dataset_train.hdf5'),
        help='Path to train dataset json file'
    )
    parser.add_argument(
        '--val_data', default=os.path.join(BASE_DIR, 'data', 'physiobank_dataset_val.hdf5'),
        help='Path to val dataset json file'
    )
    parser.add_argument(
        '--test_data', default=os.path.join(BASE_DIR, 'data', 'physiobank_dataset_test.hdf5'),
        help='Path to test dataset json file'
    )
    parser.add_argument(
        '--user_info', default=os.path.join(BASE_DIR, 'data', 'user_info.json'),
        help='Path to user info json file'
    )
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument(
        '--model', required=True, choices=['bilstm', 'cnn_gru'],
        help='Type of model to run training on'
    )
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--rnn_dim', type=int, default=256, help='Hidden dimension of RNN')
    parser.add_argument('--n_layers', type=int, default=1, help='Number of RNN layers')
    parser.add_argument('--cnn_dim', type=int, default=128, help='Output dimenstion of CNN')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--fc_dim', type=int, default=50, help='Hidden dimenstion of FC layers')
    parser.add_argument('--checkpoint', required=True, help='Path to checkpoint')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers')
    parser.add_argument('--test', action='store_true', help='Run test only')
    args = parser.parse_args()

    # Initialize random seed
    random.seed(0)
    torch.manual_seed(0)

    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create dataset
    print('Loading dataset')
    train_dataset, train_loader = create_dataset(args.train_data, args.user_info, args.batch_size, args.num_workers)
    _, val_loader = create_dataset(args.val_data, args.user_info, args.batch_size, args.num_workers)

    # Create model
    print('Creating model')
    model = create_model(
        args.model,
        args.rnn_dim,
        len(train_dataset[0][1]),
        args.n_layers,
        args.cnn_dim,
        args.fc_dim,
        args.dropout,
        device
    )

    # Create criterion
    criterion = nn.BCEWithLogitsLoss().to(device)

    if not args.test:
        if os.path.exists(args.checkpoint):
            raise ValueError('Checkpoint already exists')

        # Create optimizer and criterion
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Train model
        print('Training...')
        training_epoch(
            model, train_loader, val_loader, optimizer, criterion,
            args.epochs, args.checkpoint, device
        )

    # Test model
    print('Testing...')
    _, test_loader = create_dataset(args.test_data, args.user_info, args.batch_size, args.num_workers)
    model.load_state_dict(torch.load(args.checkpoint))
    eval(model, test_loader, criterion, device, type='test')
