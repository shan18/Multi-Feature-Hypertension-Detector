import torch
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, roc_auc_score
from model import save_model
from utils import ProgressBar


def calculate_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)
    return (
        round(accuracy_score(y_true, y_pred) * 100, 4),
        round(precision_score(y_true, y_pred, zero_division=0), 4),
        round(recall_score(y_true, y_pred, zero_division=0), 4),
        round(f1_score(y_true, y_pred, zero_division=0), 4),
        round(specificity, 4),
        round(roc_auc_score(y_true, y_pred) * 100, 4),
    )


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
    accuracy, precision, recall, f1, specificity, auc = calculate_metrics(y_true, y_pred)
    pbar.add(1, values=[
        ('Loss', round(loss.item(), 4)),
        ('Accuracy', accuracy),
        ('Precision', precision),
        ('Recall', recall),
        ('F1', f1),
        ('Specificity', specificity),
        ('AUC', auc),
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
    accuracy, precision, recall, f1, specificity, auc = calculate_metrics(y_true, y_pred)
    print(
        f'{"Validation" if type == "val" else "Test"} set: '
        f'Average loss: {loss:.4f}, '
        f'Accuracy: {accuracy}% '
        f'Precision: {precision} '
        f'Recall: {recall} '
        f'F1: {f1:.2f} '
        f'Specificity: {specificity} '
        f'AUC: {auc}%\n'
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
