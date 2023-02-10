import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

modalities = ["image", "text", "multi"]
datasets = ["fbhm", "m7k", "moff"]  # "fbhm", "m7k", "moff"
nbest = 10

if "fbhm" in datasets:
    print("=========================")
    print("Facebook Hateful Memes")
    print("=========================")

    with open("results/baselines_fbhm.pickle", "rb") as h:
        baseline = pickle.load(h)
    with open("results/memefier_fbhm.pickle", "rb") as h:
        memefier = pickle.load(h)

    score = pd.DataFrame(
        data=np.full((len(modalities) + nbest, 2), np.nan),
        index=modalities + [x + 1 for x in range(nbest)],
        columns=["acc", "auc"],
    )
    for modality in modalities:
        constrained = [r for r in baseline if r["modality"] == modality]
        score_ = [(max(c["val_acc"]), max(c["val_auc"])) for c in constrained]
        score_ = sorted(score_, key=lambda x: x[1], reverse=True)
        score["acc"][modality] = score_[0][0]
        score["auc"][modality] = score_[0][1]

    memefier = sorted(memefier, key=lambda x: max(x["metrics"]["auc"]), reverse=True)
    print("Minimum MemeFier AUC:", max(memefier[-1]["metrics"]["auc"]))
    for i in range(nbest):
        lr = memefier[i]["config"]["lr"]
        epochs = memefier[i]["config"]["epochs"]
        alpha = memefier[i]["config"]["alpha"]
        d = memefier[i]["config"]["d"]
        enc = str(memefier[i]["config"]["enc"])
        dec = str(memefier[i]["config"]["dec"])

        score["acc"][i + 1] = max(memefier[i]["metrics"]["acc"])
        score["auc"][i + 1] = max(memefier[i]["metrics"]["auc"])

        score.rename(
            index={
                i + 1: f"lr:{lr},epochs:{epochs},a:{alpha},d:{d},enc:{enc},dec:{dec}"
            },
            inplace=True,
        )

    print(score.round(3))

    plt.figure(figsize=(12, 7))
    plt.suptitle("AUC")
    plt.savefig("bar plots.png")
    for i, Q in enumerate(["lr", "epochs", "alpha", "d", "enc", "dec"]):
        qs = sorted(
            list(
                set(
                    [
                        x["config"][Q]
                        if not isinstance(x["config"][Q], dict)
                        else str(x["config"][Q])
                        for x in memefier
                    ]
                )
            )
        )
        aucs = [
            [max(x["metrics"]["auc"]) for x in memefier if x["config"][Q] == q]
            if not isinstance(q, str)
            else [
                max(x["metrics"]["auc"]) for x in memefier if str(x["config"][Q]) == q
            ]
            for q in qs
        ]
        plt.subplot(2, 3, i + 1)
        plt.boxplot(aucs)
        plt.xticks(np.arange(len(qs)) + 1, qs, rotation=8)
        plt.xlabel(Q)
    plt.savefig("results/bar_fbhm.png")


if "m7k" in datasets:
    print("\n\n=========================")
    print("Memotion7k")
    print("=========================")

    with open("results/baselines_m7k.pickle", "rb") as h:
        baseline = pickle.load(h)
    with open("results/memefier_m7k.pickle", "rb") as h:
        memefier = pickle.load(h)

    tasks = ["a", "b", "c"]
    acc_df = pd.DataFrame(
        data=np.full((len(modalities) + 1, len(tasks)), np.nan),
        index=modalities + ["memefier"],
        columns=tasks,
    )
    f1_df = pd.DataFrame(
        data=np.full((len(modalities) + 1, len(tasks)), np.nan),
        index=modalities + ["memefier"],
        columns=tasks,
    )
    for task in tasks:
        for modality in modalities:
            constrained = [
                r for r in baseline if r["task"] == task and r["modality"] == modality
            ]

            acc = max(
                [
                    np.mean([c["test_acc"][x] for x in c["test_acc"]])
                    for c in constrained
                ]
            )
            f1 = max(
                [np.mean([c["test_f1"][x] for x in c["test_f1"]]) for c in constrained]
            )

            acc_df[task][modality] = acc
            f1_df[task][modality] = f1

        constrained = [r for r in memefier if r["config"]["task"] == task]
        acc = max(
            [
                np.mean([c["test_metrics"]["acc"][x] for x in c["test_metrics"]["acc"]])
                for c in constrained
            ]
        )
        f1 = max(
            [
                np.mean([c["test_metrics"]["f1"][x] for x in c["test_metrics"]["f1"]])
                for c in constrained
            ]
        )
        acc_df[task]["memefier"] = acc
        f1_df[task]["memefier"] = f1

    print("Test accuracy:")
    print(acc_df.round(3))
    print("\nTest f1:")
    print(f1_df.round(3))

if "moff" in datasets:
    print("\n\n=========================")
    print("MultiOFF")
    print("=========================")

    with open("results/baselines_moff.pickle", "rb") as h:
        baseline = pickle.load(h)
    
    with open("results/memefier_moff.pickle", "rb") as h:
        memefier = pickle.load(h)

    acc_df = pd.DataFrame(
        data=np.full((len(modalities) + 1, 1), np.nan), 
        index=modalities + ["memefier"], 
        columns=["acc"]
    )
    f1_df = pd.DataFrame(
        data=np.full((len(modalities) + 1, 1), np.nan), 
        index=modalities + ["memefier"], 
        columns=["f1"]
    )
    for modality in modalities:
        constrained = [r for r in baseline if r["modality"] == modality]

        acc = max([max(c["val_acc"]) for c in constrained])
        f1 = max([max(c["val_f1"]) for c in constrained])

        acc_df["acc"][modality] = acc
        f1_df["f1"][modality] = f1
    
    acc_df["acc"]["memefier"] = max([r["test_metrics"]["acc"] for r in memefier])
    f1_df["f1"]["memefier"] = max([r["test_metrics"]["f1"] for r in memefier])

    print("test accuracy:")
    print(acc_df.round(3))
    print("\ntest f1:")
    print(f1_df.round(3))
