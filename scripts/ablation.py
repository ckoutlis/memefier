import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data import FBHM
from src.models import MemeFier
from src.utils import seed_everything, loss_function

from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np

datdir = "data/fbhm/"
device = "cuda:0"
workers = 10
patience = 4

vocab_size = 5000
seq_len = 34
seed = 42
batch_size = 32
weight_decay = 0.0001

lr = 0.0001
epochs = 16
alpha = 0.2
d = 1024
enc = {"h": 16, "dff": 2048, "L": 3}
dec = {"d": 64, "h": 4, "dff": 64, "L": 1}

for components in ["C12", "E12", "EC2", "EC1"]:
    print(components)
    max_auc = 0
    seed_everything(seed)

    model = MemeFier(
        caption_max_len=18,
        caption_vocab_size=2853,
        d=d,
        enc=enc,
        dec=dec,
        components=components,
        device=device,
    )

    train_ds = FBHM(
        directory=datdir,
        train=True,
        vocab_size=vocab_size,
        seq_len=seq_len,
        transform=model.preprocess,
    )
    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        drop_last=False,
    )
    val_ds = FBHM(
        directory=datdir,
        train=False,
        vocab_size=vocab_size,
        seq_len=seq_len,
        transform=model.preprocess,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        drop_last=False,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss = nn.BCEWithLogitsLoss(reduction="mean")
    if "C" in model.components:
        caption_loss = torch.nn.CrossEntropyLoss(reduction="none")

    metrics = {"l_cap": [], "l_hate": [], "loss": [], "acc": [], "auc": []}
    for epoch in range(epochs):
        if epoch >= epochs // 2:
            optimizer.param_groups[0]["lr"] = lr / 10

        model.train()
        losses = []
        l_cap = 0
        l_hate = 0
        accs = []
        for i, data in enumerate(train_dl):
            labels = data["label"].to(device)
            outputs = model(data)
            optimizer.zero_grad()
            if "C" in model.components:
                hate_pred, caption_pred = outputs
                l_cap_ = loss_function(
                    caption_loss,
                    caption_pred,
                    data["caption_index"],
                    device,
                )
                l_hate_ = loss(hate_pred, labels.float().view(-1, 1))
                loss_ = alpha * l_cap_ + l_hate_
            else:
                hate_pred = outputs
                l_hate_ = loss(hate_pred, labels.float().view(-1, 1))
                loss_ = l_hate_
            loss_.backward()

            optimizer.step()

        model.eval()
        val_loss = 0
        l_cap = 0
        l_hate = 0
        y_score = []
        y_true = []
        y_pred = []
        with torch.no_grad():
            for data in val_dl:
                labels = data["label"].to(device)
                outputs = model(data)
                if "C" in model.components:
                    hate_pred, caption_pred = outputs
                    l_cap_ = loss_function(
                        caption_loss,
                        caption_pred,
                        data["caption_index"],
                        device,
                    )
                    l_hate_ = loss(
                        hate_pred,
                        labels.float().view(-1, 1),
                    )
                    l_cap += l_cap_.item()
                    l_hate += l_hate_.item()
                    val_loss += (alpha * l_cap_ + l_hate_).item()
                else:
                    hate_pred = outputs
                    l_hate_ = loss(
                        hate_pred,
                        labels.float().view(-1, 1),
                    )
                    val_loss += l_hate_.item()
                score = torch.sigmoid(hate_pred)
                y_score.extend(score.cpu().numpy().tolist())
                y_true.extend(labels.cpu().numpy().tolist())
                y_pred.extend((score > 0.5).cpu().numpy().tolist())

        metrics["l_cap"].append(l_cap / len(val_dl))
        metrics["l_hate"].append(l_hate / len(val_dl))
        metrics["loss"].append(val_loss / len(val_dl))
        metrics["acc"].append(accuracy_score(y_true, y_pred))
        metrics["auc"].append(roc_auc_score(y_true, y_score))

        if metrics["auc"][-1] > max_auc:
            max_auc = metrics["auc"][-1]

        if (epoch >= patience) and all(
                [
                    metrics["auc"][-patience - 1] > metrics["auc"][x]
                    for x in -np.arange(1, patience + 1)
                ]
        ):
            break

    print(max_auc)

# C12
# 0.7873439999999999
# E12
# 0.7911600000000001
# EC2
# 0.728032
# EC1
# 0.52