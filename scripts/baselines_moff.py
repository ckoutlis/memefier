import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from src.models import ImageOnly, TextOnly, MultiModal
from src.data import MultiOFF
from src.utils import evaluate_moff

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

datdir = "data/moff/"
savpath = "results/baselines_moff.pickle"
imnet_mean = [0.485, 0.456, 0.406]
imnet_std = [0.229, 0.224, 0.225]
INIT_IMG_SIZE = 256
IMG_SIZE = 224
VOCAB_SIZE = 1462  # 1462 words have at least 2 occurrences
SEQ_LEN = 77  # 90% quantile is 77, 95% is 102 and 99% is 142
hidden_dims = [64, 128, 256]
lstm_layers = [1, 3]
batch_size = 128
epochs = 10
modalities = ["image", "text", "multi"]
learning_rates = [1e-2, 1e-3, 1e-4, 1e-5]
pretrained = [False, True]

train_ds = MultiOFF(
    directory=datdir,
    split="train",
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
val_ds = MultiOFF(
    directory=datdir,
    split="val",
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
test_ds = MultiOFF(
    directory=datdir,
    split="test",
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
test_dl = DataLoader(
    val_ds,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
    drop_last=False,
)

results = []
for modality in modalities:
    ckpt = f"ckpt/baseline_moff_{modality}.pt"
    max_f1 = 0
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
                    val_f1 = []
                    for epoch in range(epochs):
                        model.train()
                        for data in train_dl:
                            labels = data["label"].to(device)
                            optimizer.zero_grad()
                            outputs = model(data)
                            loss_ = criterion(outputs, labels.float().view(-1, 1))
                            loss_.backward()
                            optimizer.step()

                        val_loss_, val_acc_, val_f1_ = evaluate_moff(
                            model, val_dl, criterion, device
                        )

                        if val_f1_ > max_f1:
                            max_f1 = val_f1_
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
                                            "val_f1": val_f1_,
                                            "epoch": epoch,
                                        },
                                        indent=4,
                                    )
                                )

                        val_loss.append(val_loss_)
                        val_acc.append(val_acc_)
                        val_f1.append(val_f1_)

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
                        "val_f1": val_f1,
                        "time": time.time() - start,
                    }
                    print(json.dumps(result, indent=4))
                    results.append(result)

                    with open(savpath, "wb") as h:
                        pickle.dump(results, h, protocol=pickle.HIGHEST_PROTOCOL)

# TEST SET EVALUATION OF 3 MODELS: image, text, multi
print("\nTest set performance:")
for modality in modalities:
    model = torch.load(f"../ckpt/baseline_moff_{modality}.pt")
    test_loss_, test_acc_, test_f1_ = evaluate_moff(model, test_dl, criterion, device)
    print(
        f"modality: {modality}, loss={test_loss_:1.4f}, acc={test_acc_:1.4f}, f1={test_f1_:1.4f}"
    )

# Test set performance:
# modality: image, loss=1.1297, acc=0.6376, f1=0.6189
# modality: text, loss=0.6823, acc=0.5705, f1=0.5083
# modality: multi, loss=0.7887, acc=0.6711, f1=0.6255
