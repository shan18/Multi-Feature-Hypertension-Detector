import torch
from torchinfo import summary

from .bilstm import HypertensionDetectorBiLSTM
from .cnn_gru import HypertensionDetectorConvGRU


def create_model(
    model_type, rnn_dim, seq_meta_length, n_layers, fc_dim, dropout, device
):
    if model_type == 'bilstm':
        model = HypertensionDetectorBiLSTM(
            rnn_dim, seq_meta_length, n_layers, fc_dim, dropout, device
        ).to(device)
    else:
        model = HypertensionDetectorConvGRU(
            rnn_dim, seq_meta_length, n_layers, fc_dim, dropout
        ).to(device)

    print('\nModel Summary:')
    summary(model)
    print()

    return model


def save_model(model, path):
    ckpt_args = {'state_dict': model.state_dict(), 'arch': model.arch}
    if model.arch == 'bilstm':
        ckpt_args['args'] = [
            model.hidden_dim, model.seq_meta_len, model.n_layers,
            model.fc_dim, model.dropout, model.device
        ]
    elif model.arch == 'cnn_gru':
        ckpt_args['args'] = [
            model.rnn_hidden_dim,  model.seq_meta_len, model.n_layers,
            model.fc_dim, model.dropout, None
        ]
    torch.save(ckpt_args, path)


def load_model(path):
    checkpoint = torch.load(path)
    model = create_model(checkpoint['arch'], *checkpoint['args'])
    model.load_state_dict(checkpoint['state_dict'])
    return model
