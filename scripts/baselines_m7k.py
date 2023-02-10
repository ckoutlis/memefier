import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from src.models import ImageOnly, TextOnly, MultiModal
from src.data import Memotion7k
from sklearn.metrics import f1_score, accuracy_score
import time
import random
import numpy as np
import pickle
import json

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    raise Exception("No GPUs available!")

datdir = "data/m7k/"
savpath = "results/baselines_m7k.pickle"
ckpt = "ckpt/baseline_m7k.pt"
imnet_mean = [0.485, 0.456, 0.406]
imnet_std = [0.229, 0.224, 0.225]
INIT_IMG_SIZE = 256
IMG_SIZE = 224
VOCAB_SIZE = 1834  # 1834 words have at least 5 occurrences
SEQ_LEN = 23  # 90% quantile is 23, 95% is 28 and 99% is 40
hidden_dims = [64, 128, 256]
lstm_layers = [1, 3]
batch_size = 128
epochs = 10
categories = ["humour", "sarcasm", "offensive", "motivational"]
tasks = ["a", "b", "c"]
modalities = ["image", "text", "multi"]
learning_rates = [1e-2, 1e-3, 1e-4, 1e-5]
pretrained = [False, True]

results = []
for task in tasks:
    train_ds = Memotion7k(
        split="train",
        vocab_size=VOCAB_SIZE,
        seq_len=SEQ_LEN,
        task=task,
        transform=transforms.Compose(
            [
                transforms.Resize((INIT_IMG_SIZE, INIT_IMG_SIZE)),
                transforms.RandomCrop(IMG_SIZE),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(imnet_mean, imnet_std),
            ]
        ),
    )
    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=False,
    )
    val_ds = Memotion7k(
        split="val",
        vocab_size=VOCAB_SIZE,
        seq_len=SEQ_LEN,
        task=task,
        transform=transforms.Compose(
            [
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(imnet_mean, imnet_std),
            ]
        ),
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=False,
    )
    test_ds = Memotion7k(
        split="test",
        vocab_size=VOCAB_SIZE,
        seq_len=SEQ_LEN,
        task=task,
        transform=transforms.Compose(
            [
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(imnet_mean, imnet_std),
            ]
        ),
    )
    test_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=False,
    )

    num_classes = train_ds.__num_classes__()

    for modality in modalities:
        for learning_rate in learning_rates:
            for pretrained_ in pretrained:
                for hidden_dim in hidden_dims:
                    for lstm_layers_ in lstm_layers:

                        if (modality == "text" and pretrained_ is True) or (
                            modality == "image" and lstm_layers_ in lstm_layers[1:]
                        ):
                            continue

                        print(
                            f"t={task} - m={modality} - lr={learning_rate} - pt={pretrained_} - hd={hidden_dim} - l={lstm_layers_}"
                        )

                        if modality == "image":
                            model = ImageOnly(
                                pretrained=pretrained_,
                                hidden_dim=hidden_dim,
                                num_classes=num_classes[task],
                                device=device,
                            ).to(device)
                        elif modality == "text":
                            model = TextOnly(
                                max_words=VOCAB_SIZE + 2,
                                hidden_dim=hidden_dim,
                                lstm_layers=lstm_layers_,
                                num_classes=num_classes[task],
                                device=device,
                            ).to(device)
                        elif modality == "multi":
                            model = MultiModal(
                                pretrained=pretrained_,
                                max_words=VOCAB_SIZE + 2,
                                hidden_dim=hidden_dim,
                                lstm_layers=lstm_layers_,
                                num_classes=num_classes[task],
                                device=device,
                            ).to(device)
                        else:
                            raise Exception(f"Modality name {modality} is invalid.")

                        if task == "a":
                            criterion = torch.nn.CrossEntropyLoss()
                        elif task == "b":
                            criterion = torch.nn.BCEWithLogitsLoss()
                        elif task == "c":
                            criterion = {
                                c: torch.nn.CrossEntropyLoss() for c in categories
                            }
                            criterion["motivational"] = torch.nn.BCEWithLogitsLoss()
                        else:
                            raise Exception(f"Task {task} is not a valid name.")

                        optimizer = torch.optim.Adam(
                            model.parameters(), lr=learning_rate
                        )

                        start = time.time()
                        max_f1 = 0
                        val_loss = []
                        val_acc = []
                        val_f1 = []
                        for epoch in range(epochs):
                            model.train()
                            for data in train_dl:
                                labels = {
                                    "a": data["overall_sentiment"].to(device),
                                    "b": torch.stack(
                                        [data[c] for c in categories], dim=1
                                    )
                                    .float()
                                    .to(device),
                                    "c": {c: data[c].to(device) for c in categories},
                                }
                                labels["c"]["motivational"] = labels["c"][
                                    "motivational"
                                ].float()
                                optimizer.zero_grad()
                                outputs = model(data)
                                loss_ = (
                                    criterion(outputs, labels[task])
                                    if task in ["a", "b"]
                                    else sum(
                                        criterion[c](
                                            torch.squeeze(outputs[i]), labels[task][c]
                                        )
                                        for i, c in enumerate(categories)
                                    )
                                    / len(categories)
                                )
                                loss_.backward()
                                optimizer.step()

                            model.eval()
                            val_loss_ = 0
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
                                        "b": torch.stack(
                                            [data[c] for c in categories], dim=1
                                        )
                                        .float()
                                        .to(device),
                                        "c": {
                                            c: data[c].to(device) for c in categories
                                        },
                                    }
                                    labels["c"]["motivational"] = labels["c"][
                                        "motivational"
                                    ].float()
                                    outputs = model(data)
                                    val_loss_ += (
                                        criterion(outputs, labels[task])
                                        if task in ["a", "b"]
                                        else sum(
                                            criterion[c](
                                                torch.squeeze(outputs[i]),
                                                labels[task][c],
                                            )
                                            for i, c in enumerate(categories)
                                        )
                                        / len(categories)
                                    ).item()
                                    predicted = (
                                        torch.max(outputs.data, 1)[1].view(-1, 1)
                                        if task == "a"
                                        else torch.sigmoid(outputs.data) > 0.5
                                        if task == "b"
                                        else torch.stack(
                                            [
                                                torch.max(outputs[i].data, 1)[1]
                                                for i, _ in enumerate(categories[:-1])
                                            ]
                                            + [
                                                torch.sigmoid(outputs[3].data).view(-1)
                                                > 0.5
                                            ],
                                            dim=1,
                                        )
                                    )
                                    L = (
                                        labels[task].view(-1, 1)
                                        if task == "a"
                                        else labels[task]
                                        if task == "b"
                                        else torch.stack(
                                            [labels["c"][c] for c in categories], dim=1
                                        )
                                    )
                                    for i, c in enumerate(y_true.keys()):
                                        y_true[c].extend(L[:, i].cpu().numpy().tolist())
                                        y_pred[c].extend(
                                            predicted[:, i].cpu().numpy().tolist()
                                        )

                            val_loss_ /= len(val_dl)
                            val_acc_ = {}
                            val_f1_ = {}
                            for c in y_true.keys():
                                val_acc_[c] = accuracy_score(y_true[c], y_pred[c])
                                val_f1_[c] = f1_score(
                                    y_true[c], y_pred[c], average="macro"
                                )

                            if np.mean([val_f1_[c] for c in val_f1_]) > max_f1:
                                max_f1 = np.mean([val_f1_[c] for c in val_f1_])
                                torch.save(model, ckpt)

                            val_loss.append(val_loss_)
                            val_acc.append(val_acc_)
                            val_f1.append(val_f1_)

                            if epoch == int(epochs / 2) - 1:
                                for g in optimizer.param_groups:
                                    g["lr"] = learning_rate / 10

                        model = torch.load(ckpt)
                        model.eval()
                        test_loss_ = 0
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
                            for data in test_dl:
                                labels = {
                                    "a": data["overall_sentiment"].to(device),
                                    "b": torch.stack(
                                        [data[c] for c in categories], dim=1
                                    )
                                    .float()
                                    .to(device),
                                    "c": {c: data[c].to(device) for c in categories},
                                }
                                labels["c"]["motivational"] = labels["c"][
                                    "motivational"
                                ].float()
                                outputs = model(data)
                                test_loss_ += (
                                    criterion(outputs, labels[task])
                                    if task in ["a", "b"]
                                    else sum(
                                        criterion[c](
                                            torch.squeeze(outputs[i]),
                                            labels[task][c],
                                        )
                                        for i, c in enumerate(categories)
                                    )
                                    / len(categories)
                                ).item()
                                predicted = (
                                    torch.max(outputs.data, 1)[1].view(-1, 1)
                                    if task == "a"
                                    else torch.sigmoid(outputs.data) > 0.5
                                    if task == "b"
                                    else torch.stack(
                                        [
                                            torch.max(outputs[i].data, 1)[1]
                                            for i, _ in enumerate(categories[:-1])
                                        ]
                                        + [
                                            torch.sigmoid(outputs[3].data).view(-1)
                                            > 0.5
                                        ],
                                        dim=1,
                                    )
                                )
                                L = (
                                    labels[task].view(-1, 1)
                                    if task == "a"
                                    else labels[task]
                                    if task == "b"
                                    else torch.stack(
                                        [labels["c"][c] for c in categories], dim=1
                                    )
                                )
                                for i, c in enumerate(y_true.keys()):
                                    y_true[c].extend(L[:, i].cpu().numpy().tolist())
                                    y_pred[c].extend(
                                        predicted[:, i].cpu().numpy().tolist()
                                    )

                        test_loss_ /= len(test_dl)
                        test_acc_ = {}
                        test_f1_ = {}
                        for c in y_true.keys():
                            test_acc_[c] = accuracy_score(y_true[c], y_pred[c])
                            test_f1_[c] = f1_score(
                                y_true[c], y_pred[c], average="macro"
                            )
                        result = {
                            "task": task,
                            "modality": modality,
                            "lr": learning_rate,
                            "pretrained": pretrained_,
                            "hidden_dim": hidden_dim,
                            "lstm_layers": lstm_layers_,
                            "val_loss": val_loss,
                            "val_acc": val_acc,
                            "val_f1": val_f1,
                            "test_loss": test_loss_,
                            "test_acc": test_acc_,
                            "test_f1": test_f1_,
                            "time": time.time() - start,
                        }
                        print(json.dumps(result, indent=4))
                        results.append(result)

                        with open(savpath, "wb") as h:
                            pickle.dump(results, h, protocol=pickle.HIGHEST_PROTOCOL)
