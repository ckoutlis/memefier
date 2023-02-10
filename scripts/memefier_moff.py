import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data import MultiOFF
from src.models import MemeFier
from src.decoder import loss_function
from src.utils import seed_everything, experiments, evaluate_moff_memefier

import numpy as np
import pickle
import json

datdir = "data/moff/"
savpath = "results/memefier_moff.pickle"
ckpt = "ckpt/memefier_moff.pt"
device = "cuda:1"
workers = 10
patience = 4

vocab_size = 1462
seq_len = 77
seed = 42
batch_size = 32
weight_decay = 0.0001
components = "EC"

experiments = experiments()
total = len(experiments)
results = []
for indx, lr, epochs, alpha, d, enc, dec in experiments:
    config = {
        "dataset": "moff",
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

    seed_everything(seed)

    model = MemeFier(
        d=d,
        enc=enc,
        dec=dec,
        components=components,
        caption_max_len=18,
        caption_vocab_size=1427,
        device=device,
    )

    train_ds = MultiOFF(
        directory=datdir,
        split="train",
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
    val_ds = MultiOFF(
        directory=datdir,
        split="val",
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
    test_ds = MultiOFF(
        directory=datdir,
        split="test",
        vocab_size=vocab_size,
        seq_len=seq_len,
        transform=model.preprocess,
    )
    test_dl = DataLoader(
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

    metrics = {"l_cap": [], "l_hate": [], "loss": [], "acc": [], "f1": []}
    max_f1 = 0
    for epoch in range(epochs):
        if epoch >= epochs // 2:
            optimizer.param_groups[0]["lr"] = lr / 10

        model.train()
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

        (
            val_loss_cap,
            val_loss_hate,
            val_loss_,
            val_acc_,
            val_f1_,
        ) = evaluate_moff_memefier(
            model, val_dl, loss, caption_loss, loss_function, alpha, device
        )
        metrics["l_cap"].append(val_loss_cap)
        metrics["l_hate"].append(val_loss_hate)
        metrics["loss"].append(val_loss_)
        metrics["acc"].append(val_acc_)
        metrics["f1"].append(val_f1_)

        if val_f1_ > max_f1:
            max_f1 = val_f1_
            torch.save(model, ckpt)

        if (epoch >= patience) and all(
            [
                metrics["f1"][-patience - 1] > metrics["f1"][x]
                for x in -np.arange(1, patience + 1)
            ]
        ):
            break

    model = torch.load(ckpt)
    model.eval()
    (_, _, _, test_acc_, test_f1_) = evaluate_moff_memefier(
        model, test_dl, loss, caption_loss, loss_function, alpha, device
    )
    result = {
        "config": config,
        "val_metrics": metrics,
        "test_metrics": {"acc": test_acc_, "f1": test_f1_},
    }
    print(json.dumps(result, indent=4))
    results.append(result)
    with open(savpath, "wb") as h:
        pickle.dump(results, h, protocol=pickle.HIGHEST_PROTOCOL)
