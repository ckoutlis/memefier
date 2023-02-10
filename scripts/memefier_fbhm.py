import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data import FBHM
from src.models import MemeFier
from src.utils import seed_everything, experiments, loss_function

from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np
import pickle
import json

datdir = "data/fbhm/"
savpath = "results/memefier_fbhm.pickle"
device = "cuda:0"
workers = 10
patience = 4

vocab_size = 5000
seq_len = 34
seed = 42
batch_size = 32
weight_decay = 0.0001
components = "EC"

experiments = experiments()
total = len(experiments)
results = []
for indx, lr, epochs, alpha, d, enc, dec in experiments:
    max_auc = max([max(r["metrics"]["auc"]) for r in results]) if results else 0.0
    config = {
        "dataset": "fbhm",
        "seed": seed,
        "batch_size": batch_size,
        "weight_decay": weight_decay,
        "components": components,
        "indx": indx,
        "lr": lr,
        "epochs": epochs,
        "alpha": alpha,
        "d": d,
        "enc": enc,
        "dec": dec,
    }
    print(f"max auc:{max_auc}")
    print(json.dumps(config, indent=4))

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

            losses.append(loss_.item())
            if "C" in model.components:
                l_cap += l_cap_.item()
            l_hate += l_hate_.item()
            accs.append(
                accuracy_score(
                    labels.cpu().numpy().tolist(),
                    (torch.sigmoid(hate_pred) > 0.5).cpu().numpy().tolist(),
                )
            )
            print(
                f"\r[e:{epoch + 1},b:{i + 1}/{len(train_dl)}] loss:({l_cap / len(losses):1.3f},{l_hate / len(losses):1.3f},{sum(losses) / len(losses):1.3f}), acc:{sum(accs) / len(accs):1.3f}",
                end="",
            )

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

        print(
            f" || val_loss:({l_cap / len(val_dl):1.3f},{l_hate / len(val_dl):1.3f},{val_loss / len(val_dl):1.3f}), val_acc:{accuracy_score(y_true, y_pred):1.3f}, val_auc:{roc_auc_score(y_true, y_score):1.3f}"
        )

        if (epoch >= patience) and all(
            [
                metrics["auc"][-patience - 1] > metrics["auc"][x]
                for x in -np.arange(1, patience + 1)
            ]
        ):
            break
    results.append({"config": config, "metrics": metrics})
    with open(savpath, "wb") as h:
        pickle.dump(results, h, protocol=pickle.HIGHEST_PROTOCOL)
