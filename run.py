import os
import random
import argparse
import torch
from torch import nn, optim
from dataset import create_dataset
from model import create_model, save_model, load_model
from engine import fit


def run_training(args, device):
    # Create dataset
    print('Loading dataset')
    train_dataset, train_loader = create_dataset(args.data_file, args.train_data, args.batch_size, args.num_workers)
    _, val_loader = create_dataset(args.data_file, args.val_data, args.batch_size, args.num_workers)

    # Create model
    print('Creating model')
    model = create_model(
        args.model, args.rnn_dim, len(train_dataset[0][1]),
        args.n_layers, args.fc_dim, args.dropout, device
    )

    # Create optimizer and criterion
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train model
    print('Training...')
    try:
        fit(
            model, train_loader, val_loader, optimizer, criterion,
            args.epochs, args.checkpoint, device
        )
    except KeyboardInterrupt:
        print('Training stopped early.')
        print('Saving model...')
        save_model(model, args.checkpoint)

    print('Saving the model with the most recent weights...')
    weight_path_recent = os.path.splitext(args.checkpoint)[0] + '_last.pt'
    save_model(model, weight_path_recent)
    print('Model saved to ', weight_path_recent)


def run_test(args, device):
    print('Testing model:')

    # Create dataset
    print('Creating dataset...')
    _, test_loader = create_dataset(args.data_file, args.test_data, args.batch_size, args.num_workers)

    # Create model
    print('Loading model...')
    model = load_model(args.checkpoint)
    model = model.to(device)

    # Create loss function
    criterion = nn.BCEWithLogitsLoss()

    # Test model
    print('Testing model:')
    eval(model, test_loader, criterion, device, type='test')


if __name__ == '__main__':
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_file', default=os.path.join(BASE_DIR, 'data/files/physiobank_dataset.h5'),
        help='Path to the dataset file containing sequences'
    )
    parser.add_argument(
        '--train_data', default=os.path.join(BASE_DIR, 'data/files/user_info_train.json'),
        help='Path to train dataset json file'
    )
    parser.add_argument(
        '--val_data', default=os.path.join(BASE_DIR, 'data/files/user_info_val.json'),
        help='Path to validation dataset json file'
    )
    parser.add_argument(
        '--test_data', default=os.path.join(BASE_DIR, 'data/files/user_info_test.json'),
        help='Path to test dataset json file'
    )
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--model', required=True, choices=['bilstm', 'cnn_gru'], help='Type of model')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--rnn_dim', type=int, default=256, help='Hidden dimension of RNN')
    parser.add_argument('--n_layers', type=int, default=1, help='Number of RNN layers')
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

    if not args.test:
        if os.path.exists(args.checkpoint):
            raise ValueError('Checkpoint already exists')
        run_training(args, device)

    run_test(args, device)
