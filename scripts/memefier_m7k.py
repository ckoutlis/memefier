import torch
from torch.utils.data import DataLoader

from src.data import Memotion7k
from src.models import MemeFier
from src.utils import seed_everything, experiments, loss_function

from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import pickle
import json
import clip

datdir = "data/m7k/"
savpath = "results/memefier_m7k.pickle"
ckpt = "ckpt/memefier_m7k.pt"
device = "cuda:0"
workers = 10
_, preprocess = clip.load("ViT-L/14", device=device)

vocab_size = 1834
seq_len = 23
seed = 42
batch_size = 32
weight_decay = 0.0001
components = "EC"

tasks = ["a", "b", "c"]
categories = ["humour", "sarcasm", "offensive", "motivational"]
experiments = experiments()
total = len(experiments)
results = []
for task in tasks:
    for indx, lr, epochs, alpha, d, enc, dec in experiments:
        config = {
            "dataset": "m7k",
            "seed": seed,
            "batch_size": batch_size,
            "weight_decay": weight_decay,
            "components": components,
            "task": task,
            "indx": indx,
            "lr": lr,
            "epochs": epochs,
            "alpha": alpha,
            "d": d,
            "enc": enc,
            "dec": dec,
        }
        print(json.dumps(config, indent=4))

        seed_everything(seed)

        train_ds = Memotion7k(
            split="train",
            vocab_size=vocab_size,
            seq_len=seq_len,
            task=task,
            transform=preprocess,
        )
        train_dl = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers,
            pin_memory=True,
            drop_last=False,
        )
        val_ds = Memotion7k(
            split="val",
            vocab_size=vocab_size,
            seq_len=seq_len,
            task=task,
            transform=preprocess,
        )
        val_dl = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=True,
            drop_last=False,
        )
        test_ds = Memotion7k(
            split="test",
            vocab_size=vocab_size,
            seq_len=seq_len,
            task=task,
            transform=preprocess,
        )
        test_dl = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=True,
            drop_last=False,
        )

        num_classes = train_ds.__num_classes__()

        model = MemeFier(
            caption_max_len=18,
            caption_vocab_size=4265,
            d=d,
            enc=enc,
            dec=dec,
            num_classes=num_classes[task],
            components=components,
            device=device,
        )

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        if task == "a":
            loss = torch.nn.CrossEntropyLoss()
        elif task == "b":
            loss = torch.nn.BCEWithLogitsLoss()
        elif task == "c":
            loss = {c: torch.nn.CrossEntropyLoss() for c in categories}
            loss["motivational"] = torch.nn.BCEWithLogitsLoss()
        else:
            raise Exception(f"Task {task} is not a valid name.")
        if "C" in model.components:
            caption_loss = torch.nn.CrossEntropyLoss(reduction="none")

        metrics = {"l_cap": [], "l_cls": [], "loss": [], "acc": [], "f1": []}
        max_f1 = 0
        for epoch in range(epochs):
            if epoch >= epochs // 2:
                optimizer.param_groups[0]["lr"] = lr / 10

            model.train()
            for i, data in enumerate(train_dl):
                labels = {
                    "a": data["overall_sentiment"].to(device),
                    "b": torch.stack([data[c] for c in categories], dim=1)
                    .float()
                    .to(device),
                    "c": {c: data[c].to(device) for c in categories},
                }
                labels["c"]["motivational"] = labels["c"]["motivational"].float()
                optimizer.zero_grad()
                outputs = model(data)

                if "C" in model.components:
                    hate_pred, caption_pred = outputs
                    l_cap_ = loss_function(
                        caption_loss,
                        caption_pred,
                        data["caption_index"],
                        device,
                    )
                    l_cls_ = (
                        loss(hate_pred, labels[task])
                        if task in ["a", "b"]
                        else sum(
                            loss[c](torch.squeeze(hate_pred[i]), labels[task][c])
                            for i, c in enumerate(categories)
                        )
                        / len(categories)
                    )
                    loss_ = alpha * l_cap_ + l_cls_
                else:
                    hate_pred = outputs
                    l_cls_ = (
                        loss(hate_pred, labels[task])
                        if task in ["a", "b"]
                        else sum(
                            loss[c](torch.squeeze(hate_pred[i]), labels[task][c])
                            for i, c in enumerate(categories)
                        )
                        / len(categories)
                    )
                    loss_ = l_cls_

                loss_.backward()
                optimizer.step()

            model.eval()
            val_loss = 0
            l_cap = 0
            l_cls = 0
            y_true = (
                {"overall_sentiment": []}
                if task == "a"
                else {c: [] for c in categories}
            )
            y_pred = (
                {"overall_sentiment": []}
                if task == "a"
                else {c: [] for c in categories}
            )
            with torch.no_grad():
                for data in val_dl:
                    labels = {
                        "a": data["overall_sentiment"].to(device),
                        "b": torch.stack([data[c] for c in categories], dim=1)
                        .float()
                        .to(device),
                        "c": {c: data[c].to(device) for c in categories},
                    }
                    labels["c"]["motivational"] = labels["c"]["motivational"].float()
                    outputs = model(data)
                    if "C" in model.components:
                        hate_pred, caption_pred = outputs
                        l_cap_ = loss_function(
                            caption_loss,
                            caption_pred,
                            data["caption_index"],
                            device,
                        )
                        l_cls_ = (
                            loss(hate_pred, labels[task])
                            if task in ["a", "b"]
                            else sum(
                                loss[c](
                                    torch.squeeze(hate_pred[i]),
                                    labels[task][c],
                                )
                                for i, c in enumerate(categories)
                            )
                            / len(categories)
                        )
                        l_cap += l_cap_.item()
                        l_cls += l_cls_.item()
                        val_loss += (alpha * l_cap_ + l_cls_).item()
                    else:
                        hate_pred = outputs
                        l_cls_ = (
                            loss(hate_pred, labels[task])
                            if task in ["a", "b"]
                            else sum(
                                loss[c](
                                    torch.squeeze(hate_pred[i]),
                                    labels[task][c],
                                )
                                for i, c in enumerate(categories)
                            )
                            / len(categories)
                        )
                        val_loss += l_cls_.item()
                    predicted = (
                        torch.max(hate_pred.data, 1)[1].view(-1, 1)
                        if task == "a"
                        else torch.sigmoid(hate_pred.data) > 0.5
                        if task == "b"
                        else torch.stack(
                            [
                                torch.max(hate_pred[i].data, 1)[1]
                                for i, _ in enumerate(categories[:-1])
                            ]
                            + [torch.sigmoid(hate_pred[3].data).view(-1) > 0.5],
                            dim=1,
                        )
                    )
                    L = (
                        labels[task].view(-1, 1)
                        if task == "a"
                        else labels[task]
                        if task == "b"
                        else torch.stack([labels["c"][c] for c in categories], dim=1)
                    )
                    for i, c in enumerate(y_true.keys()):
                        y_true[c].extend(L[:, i].cpu().numpy().tolist())
                        y_pred[c].extend(predicted[:, i].cpu().numpy().tolist())

            metrics["l_cap"].append(l_cap / len(val_dl))
            metrics["l_cls"].append(l_cls / len(val_dl))
            metrics["loss"].append(val_loss / len(val_dl))
            val_acc_ = {}
            val_f1_ = {}
            for c in y_true.keys():
                val_acc_[c] = accuracy_score(y_true[c], y_pred[c])
                val_f1_[c] = f1_score(y_true[c], y_pred[c], average="macro")
            metrics["acc"].append(val_acc_)
            metrics["f1"].append(val_f1_)

            if np.mean([val_f1_[c] for c in val_f1_]) > max_f1:
                max_f1 = np.mean([val_f1_[c] for c in val_f1_])
                torch.save(model, ckpt)

        model = torch.load(ckpt)
        model.eval()
        y_true = (
            {"overall_sentiment": []} if task == "a" else {c: [] for c in categories}
        )
        y_pred = (
            {"overall_sentiment": []} if task == "a" else {c: [] for c in categories}
        )
        with torch.no_grad():
            for data in test_dl:
                labels = {
                    "a": data["overall_sentiment"].to(device),
                    "b": torch.stack([data[c] for c in categories], dim=1)
                    .float()
                    .to(device),
                    "c": {c: data[c].to(device) for c in categories},
                }
                labels["c"]["motivational"] = labels["c"]["motivational"].float()
                outputs = model(data)
                if "C" in model.components:
                    hate_pred, caption_pred = outputs
                else:
                    hate_pred = outputs

                predicted = (
                    torch.max(hate_pred.data, 1)[1].view(-1, 1)
                    if task == "a"
                    else torch.sigmoid(hate_pred.data) > 0.5
                    if task == "b"
                    else torch.stack(
                        [
                            torch.max(hate_pred[i].data, 1)[1]
                            for i, _ in enumerate(categories[:-1])
                        ]
                        + [torch.sigmoid(hate_pred[3].data).view(-1) > 0.5],
                        dim=1,
                    )
                )
                L = (
                    labels[task].view(-1, 1)
                    if task == "a"
                    else labels[task]
                    if task == "b"
                    else torch.stack([labels["c"][c] for c in categories], dim=1)
                )
                for i, c in enumerate(y_true.keys()):
                    y_true[c].extend(L[:, i].cpu().numpy().tolist())
                    y_pred[c].extend(predicted[:, i].cpu().numpy().tolist())

        test_acc_ = {}
        test_f1_ = {}
        for c in y_true.keys():
            test_acc_[c] = accuracy_score(y_true[c], y_pred[c])
            test_f1_[c] = f1_score(y_true[c], y_pred[c], average="macro")

        result = {
            "config": config,
            "val_metrics": metrics,
            "test_metrics": {"acc": test_acc_, "f1": test_f1_},
        }
        print(json.dumps(result, indent=4))
        results.append(result)
        with open(savpath, "wb") as h:
            pickle.dump(results, h, protocol=pickle.HIGHEST_PROTOCOL)
