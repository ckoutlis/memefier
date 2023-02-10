import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from sklearn.metrics import roc_auc_score, accuracy_score

from src.models import ImageOnly, TextOnly, MultiModal
from src.data import FBHM

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

datdir = "data/fbhm/"
savpath = "results/baselines_fbhm.pickle"
imnet_mean = [0.485, 0.456, 0.406]
imnet_std = [0.229, 0.224, 0.225]
INIT_IMG_SIZE = 256
IMG_SIZE = 224
VOCAB_SIZE = 2222  # 2222 words have at least 5 occurrences
SEQ_LEN = 20  # 90% quantile is 20, 95% is 24 and 99% is 34
hidden_dims = [64, 128, 256]
lstm_layers = [1, 3]
batch_size = 128
epochs = 10
modalities = ["image", "text", "multi"]
learning_rates = [1e-2, 1e-3, 1e-4, 1e-5]
pretrained = [False, True]

results = []
train_ds = FBHM(
    directory=datdir,
    train=True,
    vocab_size=VOCAB_SIZE,
    seq_len=SEQ_LEN,
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
val_ds = FBHM(
    directory=datdir,
    train=False,
    vocab_size=VOCAB_SIZE,
    seq_len=SEQ_LEN,
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

for modality in modalities:
    ckpt = f"ckpt/baseline_fbhm_{modality}.pt"
    max_auc = 0
    for learning_rate in learning_rates:
        for pretrained_ in pretrained:
            for hidden_dim in hidden_dims:
                for lstm_layers_ in lstm_layers:

                    if (modality == "text" and pretrained_ is True) or (
                        modality == "image" and lstm_layers_ in lstm_layers[1:]
                    ):
                        continue

                    print(
                        f"m={modality} - lr={learning_rate} - pt={pretrained_} - hd={hidden_dim} - l={lstm_layers_}"
                    )

                    if modality == "image":
                        model = ImageOnly(
                            pretrained=pretrained_,
                            hidden_dim=hidden_dim,
                            num_classes=1,
                            device=device,
                        ).to(device)
                    elif modality == "text":
                        model = TextOnly(
                            max_words=VOCAB_SIZE + 2,
                            hidden_dim=hidden_dim,
                            lstm_layers=lstm_layers_,
                            num_classes=1,
                            device=device,
                        ).to(device)
                    elif modality == "multi":
                        model = MultiModal(
                            pretrained=pretrained_,
                            max_words=VOCAB_SIZE + 2,
                            hidden_dim=hidden_dim,
                            lstm_layers=lstm_layers_,
                            num_classes=1,
                            device=device,
                        ).to(device)
                    else:
                        raise Exception(f"Modality name {modality} is invalid.")

                    criterion = torch.nn.BCEWithLogitsLoss()
                    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

                    start = time.time()
                    val_loss = []
                    val_acc = []
                    val_auc = []
                    for epoch in range(epochs):
                        model.train()
                        for data in train_dl:
                            labels = data["label"].to(device)
                            optimizer.zero_grad()
                            outputs = model(data)
                            loss_ = criterion(outputs, labels.float().view(-1, 1))
                            loss_.backward()
                            optimizer.step()

                        model.eval()
                        val_loss_ = 0
                        y_score = []
                        y_true = []
                        y_pred = []
                        with torch.no_grad():
                            for data in val_dl:
                                labels = data["label"].to(device)
                                outputs = model(data)
                                val_loss_ += criterion(
                                    outputs, labels.float().view(-1, 1)
                                ).item()
                                score = torch.sigmoid(outputs.data)
                                y_score.extend(score.cpu().numpy().tolist())
                                y_true.extend(labels.cpu().numpy().tolist())
                                y_pred.extend((score > 0.5).cpu().numpy().tolist())

                        val_loss_ /= len(val_dl)
                        val_acc_ = accuracy_score(y_true, y_pred)
                        val_auc_ = roc_auc_score(y_true, y_score)

                        if val_auc_ > max_auc:
                            max_auc = val_auc_
                            torch.save(model, ckpt)
                            with open(f"{ckpt[:-3]}.txt", "w") as file:
                                file.write(
                                    json.dumps(
                                        {
                                            "modality": modality,
                                            "lr": learning_rate,
                                            "pretrained": pretrained_,
                                            "hidden_dim": hidden_dim,
                                            "lstm_layers": lstm_layers_,
                                            "val_loss": val_loss_,
                                            "val_acc": val_acc_,
                                            "val_auc": val_auc_,
                                            "epoch": epoch,
                                        }
                                    )
                                )

                        val_loss.append(val_loss_)
                        val_acc.append(val_acc_)
                        val_auc.append(val_auc_)

                        if epoch == int(epochs / 2) - 1:
                            for g in optimizer.param_groups:
                                g["lr"] = learning_rate / 10

                    result = {
                        "modality": modality,
                        "lr": learning_rate,
                        "pretrained": pretrained_,
                        "hidden_dim": hidden_dim,
                        "lstm_layers": lstm_layers_,
                        "val_loss": val_loss,
                        "val_acc": val_acc,
                        "val_auc": val_auc,
                        "time": time.time() - start,
                    }
                    print(json.dumps(result, indent=4))
                    results.append(result)

                    with open(savpath, "wb") as h:
                        pickle.dump(results, h, protocol=pickle.HIGHEST_PROTOCOL)
