import torch
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from model import save_model
from utils import ProgressBar


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
        f'{"Validation" if type == "val" else "Test"} set: '
        f'Average loss: {loss:.4f}, '
        f'Accuracy: {accuracy_score(y_true, y_pred) * 100:.2f}% ',
        f'Precision: {precision_score(y_true, y_pred):.2f} ',
        f'Recall: {recall_score(y_true, y_pred):.2f} ',
        f'F1: {f1:.2f}\n'
    )

    return f1


def fit(model, train_loader, val_loader, optimizer, criterion, epochs, checkpoint, device='cpu'):
    best_val_f1 = 0

    for epoch in range(1, epochs + 1):
        print(f'Epoch {epoch}:')
        train(model, train_loader, optimizer, criterion, device)
        f1 = eval(model, val_loader, criterion, device)

        if f1 > best_val_f1:
            print(f'Validation f1 improved from {best_val_f1:.2f}% to {f1:.2f}%\n')
            best_val_f1 = f1
            save_model(model, checkpoint)
