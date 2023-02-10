import string
import json
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from textdistance import levenshtein


def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def experiments(start=0):
    x = []
    indx = 0
    for lr in [0.00001, 0.0001]:
        for epochs in [16, 32]:
            for alpha in [0.2, 0.8]:
                for d in [512, 1024]:
                    for enc in [
                        {"h": 4, "dff": 512, "L": 1},
                        {"h": 16, "dff": 2048, "L": 3},
                    ]:
                        for dec in [
                            {"d": 64, "h": 4, "dff": 64, "L": 1},
                            {"d": 256, "h": 16, "dff": 256, "L": 3},
                        ]:
                            if indx >= start:
                                x.append(
                                    [
                                        indx,
                                        lr,
                                        epochs,
                                        alpha,
                                        d,
                                        enc,
                                        dec,
                                    ]
                                )
                            indx += 1
    return x


def text_clean(text):
    clean = []
    for i, text_ in enumerate(text):
        if type(text_) is str:
            text_ = text_.lower()  # lower case
            text_ = text_.translate(
                str.maketrans("", "", string.punctuation)
            )  # remove punctuation
            text_ = "".join([i for i in text_ if not i.isdigit()])  # remove numbers
            text_ = " ".join(text_.split())  # remove double space
            clean.append(text_)
        else:
            clean.append("notext")
    return clean


def labels(task):
    humour = {
        "not_funny": 0,
        "funny": 1,
        "very_funny": 2 if task == "c" else 1,
        "hilarious": 3 if task == "c" else 1,
    }
    sarcasm = {
        "not_sarcastic": 0,
        "general": 1,
        "twisted_meaning": 2 if task == "c" else 1,
        "very_twisted": 3 if task == "c" else 1,
    }
    offensive = {
        "not_offensive": 0,
        "slight": 1,
        "very_offensive": 2 if task == "c" else 1,
        "hateful_offensive": 3 if task == "c" else 1,
    }
    motivational = {
        "not_motivational": 0,
        "motivational": 1,
    }
    overall_sentiment = {
        "very_negative": 2,
        "negative": 2,
        "neutral": 0,
        "positive": 1,
        "very_positive": 1,
    }
    return humour, sarcasm, offensive, motivational, overall_sentiment


def split_m7k():
    data = pd.read_csv("data/m7k/labels.csv")
    data = data.drop(["Unnamed: 0"], axis=1)

    np.random.seed(42)
    index = np.random.uniform(0, 1, len(data)) < 0.9
    train = data[index]
    val = data[np.logical_not(index)]
    train.index = np.arange(len(train))
    val.index = np.arange(len(val))

    test = pd.read_csv("data/m7k/test_data/2000_testdata.csv")
    test_target = pd.read_csv("data/m7k/test_data/Meme_groundTruth .csv")
    test.rename(
        columns={
            "Image_name": "image_name",
            "OCR_extracted_text": "text_ocr",
            "corrected_text": "text_corrected",
        },
        inplace=True,
    )
    test = test.drop(["Image_URL"], axis=1)
    test_target.rename(
        columns={"Image_name": "image_name"},
        inplace=True,
    )
    test.sort_index(inplace=True)
    test_target.sort_index(inplace=True)

    humour = {
        0: "not_funny",
        1: "funny",
        2: "very_funny",
        3: "hilarious",
    }
    sarcasm = {
        0: "not_sarcastic",
        1: "general",
        2: "twisted_meaning",
        3: "very_twisted",
    }
    offensive = {
        0: "not_offensive",
        1: "slight",
        2: "very_offensive",
        3: "hateful_offensive",
    }
    motivational = {
        0: "not_motivational",
        1: "motivational",
    }
    overall_sentiment = {
        -1: "negative",
        0: "neutral",
        1: "positive",
    }

    test["humour"] = [humour[int(x[-4])] for x in test_target["Labels"].values]
    test["sarcasm"] = [sarcasm[int(x[-3])] for x in test_target["Labels"].values]
    test["offensive"] = [offensive[int(x[-2])] for x in test_target["Labels"].values]
    test["motivational"] = [
        motivational[int(x[-1])] for x in test_target["Labels"].values
    ]
    test["overall_sentiment"] = [
        overall_sentiment[int(x[: x.find("_")])] for x in test_target["Labels"].values
    ]

    return train, val, test


def split_fbhm(directory):
    train = []
    with open(f"{directory}train.jsonl") as f:
        for line in f:
            sample = json.loads(line)
            train.append([sample["img"], sample["text"], sample["label"]])

    dev = []
    with open(f"{directory}dev.jsonl") as f:
        for line in f:
            sample = json.loads(line)
            dev.append([sample["img"], sample["text"], sample["label"]])

    return pd.DataFrame(train, columns=["img", "text", "label"]), pd.DataFrame(
        dev, columns=["img", "text", "label"]
    )


def split_moff(directory):
    train = pd.read_csv(
        f"{directory}Split Dataset/Training_meme_dataset.csv",
        header=0,
        names=["image", "text", "label"],
    )
    val = pd.read_csv(
        f"{directory}Split Dataset/Validation_meme_dataset.csv",
        header=0,
        names=["image", "text", "label"],
    )
    test = pd.read_csv(
        f"{directory}Split Dataset/Testing_meme_dataset.csv",
        header=0,
        names=["image", "text", "label"],
    )
    return train, val, test


def evaluate_moff(model, loader, criterion, device):
    model.eval()
    loss_ = 0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for data in loader:
            labels = data["label"].to(device)
            outputs = model(data)
            loss_ += criterion(outputs, labels.float().view(-1, 1)).item()
            score = torch.sigmoid(outputs.data)
            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend((score > 0.5).cpu().numpy().tolist())

    loss_ /= len(loader)
    acc_ = accuracy_score(y_true, y_pred)
    f1_ = f1_score(y_true, y_pred, average="macro")

    return loss_, acc_, f1_


def evaluate_moff_memefier(model, loader, loss1, loss2, loss2_agg, alpha, device):
    model.eval()
    loss_cap = 0
    loss_hate = 0
    loss_ = 0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for data in loader:
            labels = data["label"].to(device)
            outputs = model(data)
            if "C" in model.components:
                hate_pred, caption_pred = outputs
                l_cap_ = loss2_agg(
                    loss2,
                    caption_pred,
                    data["caption_index"],
                    device,
                )
                loss_cap += l_cap_.item()
                l_hate_ = loss1(hate_pred, labels.float().view(-1, 1))
                loss_hate += l_hate_.item()
                loss_ += (alpha * l_cap_ + l_hate_).item()
            else:
                hate_pred = outputs
                loss_ += loss1(hate_pred, labels.float().view(-1, 1))
            score = torch.sigmoid(hate_pred.data)
            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend((score > 0.5).cpu().numpy().tolist())

    loss_cap /= len(loader)
    loss_hate /= len(loader)
    loss_ /= len(loader)
    acc_ = accuracy_score(y_true, y_pred)
    f1_ = f1_score(y_true, y_pred, average="macro")

    return loss_cap, loss_hate, loss_, acc_, f1_


def search(handle, descriptions=None, min_sim=0.9):
    if descriptions is None:
        descriptions = {}
    for line in handle:
        dictionary = json.loads(line)
        query = dictionary["query"]
        response = dictionary["response"]["docs"]
        comment = None
        if len(response):
            max_sim = 0.0
            for r in response:
                label = r["label"][0].replace("<B>", "").replace("</B>", "").lower()
                similarity = levenshtein.normalized_similarity(query, label)
                if (
                    ("comment" in r)
                    and (similarity > min_sim)
                    and (similarity > max_sim)
                ):
                    comment = r["comment"][0].replace("<B>", "").replace("</B>", "")
                    max_sim = similarity

        descriptions[query] = comment
    return descriptions


def set_lr(warmup_lr, warmup_iter, batches, lr, epoch, i, epoch_reduce_lr):
    return (
        warmup_lr
        if epoch * batches + i < warmup_iter
        else lr
        if epoch <= epoch_reduce_lr
        else lr / 10
    )


def loss_function(loss_object, prediction, target, device):
    return sum(
        [
            (
                loss_object(prediction[:, w, :], target[:, w + 1].to(device))
                * (target[:, w + 1].to(device) != 0)
            ).mean()
            for w in range(prediction.shape[1] - 1)
        ]
    ) / (prediction.shape[1] - 1)
