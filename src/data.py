import pandas as pd
import numpy as np
from src.utils import text_clean, labels, split_m7k, split_fbhm, split_moff
import os
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torchtext.vocab import vocab
from collections import Counter, OrderedDict
import pickle
import json

ImageFile.LOAD_TRUNCATED_IMAGES = True


class Memotion7k(Dataset):
    def __init__(
        self,
        split,
        vocab_size,
        seq_len,
        task,
        transform=None,
    ):
        self.train, self.val, self.test = split_m7k()
        self.directory = (
            "data/m7k/images"
            if split in ["train", "val"]
            else "data/m7k/test_data/2000_data"
        )

        # Create the training text corpus vocabulary
        sentences = text_clean(self.train["text_ocr"].tolist())
        words = [w for s in sentences for w in s.split()]
        counts = sorted(Counter(words).items(), key=lambda x: x[1], reverse=True)[
            :vocab_size
        ]
        self.vocabulary = vocab(OrderedDict(counts))
        self.vocabulary.insert_token("<pad>", 0)
        self.vocabulary.insert_token("<unk>", 1)
        self.vocabulary.set_default_index(self.vocabulary["<unk>"])

        with open("data/captions_m7k.pickle", "rb") as handle:
            self.captions = pickle.load(handle)
        caps_sentences = text_clean([self.captions[x] for x in self.captions])
        self.caps_seq_len = max([len(s.split()) for s in caps_sentences])
        caps_words = [w for s in caps_sentences for w in s.split()]
        caps_counts = sorted(
            Counter(caps_words).items(), key=lambda x: x[1], reverse=True
        )
        self.caps_vocabulary = vocab(OrderedDict(caps_counts))
        self.caps_vocabulary.insert_token("<pad>", 0)
        self.caps_vocabulary.insert_token("<unk>", 1)
        self.caps_vocabulary.insert_token("<sos>", 2)
        self.caps_vocabulary.insert_token("<eos>", 3)
        self.caps_vocabulary.set_default_index(self.caps_vocabulary["<unk>"])

        # External knowledge
        self.attributes = pd.read_csv("data/protected_attributes_m7k.csv")[
            ["face_name_align", "race", "gender", "age"]
        ]
        self.max_attr = 7 * 3
        self.attribute_map = {
            "<pad>": 0,
            "White": 1,
            "Indian": 2,
            "Southeast Asian": 3,
            "Middle Eastern": 4,
            "Latino_Hispanic": 5,
            "Black": 6,
            "East Asian": 7,
            "Male": 8,
            "Female": 9,
            "0-2": 10,
            "3-9": 11,
            "10-19": 12,
            "20-29": 13,
            "30-39": 14,
            "40-49": 15,
            "50-59": 16,
            "60-69": 17,
            "70+": 18,
        }

        self.transform = transform
        self.seq_len = seq_len
        self.memes = (
            self.train
            if split == "train"
            else self.val
            if split == "val"
            else self.test
        )

        # Map class names to indices
        self.task = task
        (
            self.humour_classes,
            self.sarcasm_classes,
            self.offensive_classes,
            self.motivational_classes,
            self.overall_sentiment_classes,
        ) = labels(task=task)

    def __len__(self):
        return len(self.memes)

    def __num_classes__(self):
        return {
            "a": 3,
            "b": 4,
            "c": {"humour": 4, "sarcasm": 4, "offensive": 4, "motivational": 1},
        }

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_name = self.memes["image_name"][idx]
        image = Image.open(os.path.join(self.directory, image_name)).convert("RGB")
        text = text_clean([self.memes["text_ocr"][idx]])[0]
        text_index = self.vocabulary(text.split())
        humour = self.memes["humour"][idx]
        sarcasm = self.memes["sarcasm"][idx]
        offensive = self.memes["offensive"][idx]
        motivational = self.memes["motivational"][idx]
        overall_sentiment = self.memes["overall_sentiment"][idx]
        caption = text_clean([self.captions[image_name]])[0]
        caption_index = self.caps_vocabulary(caption.split())
        image_attributes = self.attributes[
            self.attributes["face_name_align"].str.contains(image_name[:-4])
        ].values[:, 1:]
        external_index = [self.attribute_map[y] for x in image_attributes for y in x]
        sample = {
            "image": image,
            "text": text,
            "text_index": text_index[: self.seq_len]
            if len(text_index) >= self.seq_len
            else text_index
            + [self.vocabulary["<pad>"]] * (self.seq_len - len(text_index)),
            "humour": self.humour_classes[str(humour)],
            "sarcasm": self.sarcasm_classes[str(sarcasm)],
            "offensive": self.offensive_classes[str(offensive)],
            "motivational": self.motivational_classes[str(motivational)],
            "overall_sentiment": self.overall_sentiment_classes[str(overall_sentiment)],
            "caption": caption,
            "caption_index": [self.caps_vocabulary["<sos>"]]
            + caption_index
            + [self.caps_vocabulary["<eos>"]]
            + [self.caps_vocabulary["<pad>"]]
            * (self.caps_seq_len - len(caption_index)),
            "external_index": external_index[: self.max_attr]
            if len(external_index) >= self.max_attr
            else external_index
            + [self.attribute_map["<pad>"]] * (self.max_attr - len(external_index)),
        }
        sample["text_index"] = torch.tensor(sample["text_index"])
        sample["caption_index"] = torch.tensor(sample["caption_index"])
        sample["external_index"] = torch.tensor(sample["external_index"])
        if self.transform:
            sample["image"] = self.transform(sample["image"])

        return sample


class FBHM(Dataset):
    def __init__(
        self,
        directory,
        train,
        vocab_size,
        seq_len,
        transform=None,
    ):
        self.directory = directory
        self.train, self.dev = split_fbhm(self.directory)

        # Create the training text corpus vocabulary
        sentences = text_clean(self.train["text"].tolist())
        words = [w for s in sentences for w in s.split()]
        counts = sorted(Counter(words).items(), key=lambda x: x[1], reverse=True)[
            :vocab_size
        ]
        self.vocabulary = vocab(OrderedDict(counts))
        self.vocabulary.insert_token("<pad>", 0)
        self.vocabulary.insert_token("<unk>", 1)
        self.vocabulary.set_default_index(self.vocabulary["<unk>"])

        with open("data/captions_fbhm.pickle", "rb") as handle:
            self.captions = pickle.load(handle)
        caps_sentences = text_clean([self.captions[x] for x in self.captions])
        self.caps_seq_len = max([len(s.split()) for s in caps_sentences])
        caps_words = [w for s in caps_sentences for w in s.split()]
        caps_counts = sorted(
            Counter(caps_words).items(), key=lambda x: x[1], reverse=True
        )
        self.caps_vocabulary = vocab(OrderedDict(caps_counts))
        self.caps_vocabulary.insert_token("<pad>", 0)
        self.caps_vocabulary.insert_token("<unk>", 1)
        self.caps_vocabulary.insert_token("<sos>", 2)
        self.caps_vocabulary.insert_token("<eos>", 3)
        self.caps_vocabulary.set_default_index(self.caps_vocabulary["<unk>"])

        # External knowledge
        self.attributes = pd.read_csv("data/protected_attributes_fbhm.csv")[
            ["face_name_align", "race", "gender", "age"]
        ]
        self.max_attr = 7 * 3
        self.attribute_map = {
            "<pad>": 0,
            "White": 1,
            "Indian": 2,
            "Southeast Asian": 3,
            "Middle Eastern": 4,
            "Latino_Hispanic": 5,
            "Black": 6,
            "East Asian": 7,
            "Male": 8,
            "Female": 9,
            "0-2": 10,
            "3-9": 11,
            "10-19": 12,
            "20-29": 13,
            "30-39": 14,
            "40-49": 15,
            "50-59": 16,
            "60-69": 17,
            "70+": 18,
        }

        # Pick the split's examples to make the dataset
        self.memes = self.train if train else self.dev

        self.seq_len = seq_len
        self.transform = transform

    def __len__(self):
        return len(self.memes)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_name = self.memes["img"][idx]
        image = Image.open(os.path.join(self.directory, image_name)).convert("RGB")
        text = text_clean([self.memes["text"][idx]])[0]
        text_index = self.vocabulary(text.split())
        label = self.memes["label"][idx]
        caption = text_clean([self.captions[image_name.replace("img/", "")]])[0]
        caption_index = self.caps_vocabulary(caption.split())
        image_attributes = self.attributes[
            self.attributes["face_name_align"].str.contains(image_name[4:-4])
        ].values[:, 1:]
        external_index = [self.attribute_map[y] for x in image_attributes for y in x]
        sample = {
            "image": image,
            "text": text,
            "text_index": text_index[: self.seq_len]
            if len(text_index) >= self.seq_len
            else text_index
            + [self.vocabulary["<pad>"]] * (self.seq_len - len(text_index)),
            "label": label,
            "caption": caption,
            "caption_index": [self.caps_vocabulary["<sos>"]]
            + caption_index
            + [self.caps_vocabulary["<eos>"]]
            + [self.caps_vocabulary["<pad>"]]
            * (self.caps_seq_len - len(caption_index)),
            "external_index": external_index[: self.max_attr]
            if len(external_index) >= self.max_attr
            else external_index
            + [self.attribute_map["<pad>"]] * (self.max_attr - len(external_index)),
        }
        sample["text_index"] = torch.tensor(sample["text_index"])
        sample["caption_index"] = torch.tensor(sample["caption_index"])
        sample["external_index"] = torch.tensor(sample["external_index"])
        if self.transform:
            sample["image"] = self.transform(sample["image"])
        return sample


class MultiOFF(Dataset):
    def __init__(
        self,
        directory,
        split,
        vocab_size,
        seq_len,
        transform=None,
    ):
        self.directory = directory
        self.train, self.val, self.test = split_moff(self.directory)

        # Create the training text corpus vocabulary
        sentences = text_clean(self.train["text"].tolist())
        words = [w for s in sentences for w in s.split()]
        counts = sorted(Counter(words).items(), key=lambda x: x[1], reverse=True)[
            :vocab_size
        ]
        self.vocabulary = vocab(OrderedDict(counts))
        self.vocabulary.insert_token("<pad>", 0)
        self.vocabulary.insert_token("<unk>", 1)
        self.vocabulary.set_default_index(self.vocabulary["<unk>"])

        with open("data/captions_moff.pickle", "rb") as handle:
            self.captions = pickle.load(handle)
        caps_sentences = text_clean([self.captions[x] for x in self.captions])
        self.caps_seq_len = max([len(s.split()) for s in caps_sentences])
        caps_words = [w for s in caps_sentences for w in s.split()]
        caps_counts = sorted(
            Counter(caps_words).items(), key=lambda x: x[1], reverse=True
        )
        self.caps_vocabulary = vocab(OrderedDict(caps_counts))
        self.caps_vocabulary.insert_token("<pad>", 0)
        self.caps_vocabulary.insert_token("<unk>", 1)
        self.caps_vocabulary.insert_token("<sos>", 2)
        self.caps_vocabulary.insert_token("<eos>", 3)
        self.caps_vocabulary.set_default_index(self.caps_vocabulary["<unk>"])

        # External knowledge
        self.attributes = pd.read_csv("data/protected_attributes_moff.csv")[
            ["face_name_align", "race", "gender", "age"]
        ]
        self.max_attr = 7 * 3
        self.attribute_map = {
            "<pad>": 0,
            "White": 1,
            "Indian": 2,
            "Southeast Asian": 3,
            "Middle Eastern": 4,
            "Latino_Hispanic": 5,
            "Black": 6,
            "East Asian": 7,
            "Male": 8,
            "Female": 9,
            "0-2": 10,
            "3-9": 11,
            "10-19": 12,
            "20-29": 13,
            "30-39": 14,
            "40-49": 15,
            "50-59": 16,
            "60-69": 17,
            "70+": 18,
        }

        # Pick the split's examples to make the dataset
        self.memes = (
            self.train
            if split == "train"
            else self.val
            if split == "val"
            else self.test
        )

        self.seq_len = seq_len
        self.transform = transform
        self.classes = {"Non-offensiv": 0, "offensive": 1}

    def __len__(self):
        return len(self.memes)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_name = self.memes["image"][idx]
        image = Image.open(f"{self.directory}Labelled Images/{image_name}").convert(
            "RGB"
        )
        text = text_clean([self.memes["text"][idx]])[0]
        text_index = self.vocabulary(text.split())
        label = self.memes["label"][idx]
        caption = text_clean([self.captions[image_name]])[0]
        caption_index = self.caps_vocabulary(caption.split())
        image_attributes = self.attributes[
            self.attributes["face_name_align"].str.contains(image_name[:-4])
        ].values[:, 1:]
        external_index = [self.attribute_map[y] for x in image_attributes for y in x]
        sample = {
            "image": image,
            "text": text,
            "text_index": text_index[: self.seq_len]
            if len(text_index) >= self.seq_len
            else text_index
            + [self.vocabulary["<pad>"]] * (self.seq_len - len(text_index)),
            "label": self.classes[str(label)],
            "caption": caption,
            "caption_index": [self.caps_vocabulary["<sos>"]]
            + caption_index
            + [self.caps_vocabulary["<eos>"]]
            + [self.caps_vocabulary["<pad>"]]
            * (self.caps_seq_len - len(caption_index)),
            "external_index": external_index[: self.max_attr]
            if len(external_index) >= self.max_attr
            else external_index
            + [self.attribute_map["<pad>"]] * (self.max_attr - len(external_index)),
        }
        sample["text_index"] = torch.tensor(sample["text_index"])
        sample["caption_index"] = torch.tensor(sample["caption_index"])
        sample["external_index"] = torch.tensor(sample["external_index"])
        if self.transform:
            sample["image"] = self.transform(sample["image"])

        return sample


class COCO(Dataset):
    def __init__(
        self,
        split,
        seq_len,
        vocab_size,
        transform=None,
    ):
        self.directory = "/home/ckoutlis/disk_2_ubuntu/home/ckoutlis/DataStorage/coco"
        images = json.load(open(f"{self.directory}/split/dataset_coco.json"))["images"]

        train = []
        val = []
        test = []
        for image in images:
            image_path = f"{self.directory}/{image['filepath']}/{image['filename']}"
            sentences = [s["tokens"] for s in image["sentences"]]
            if image["split"] in ["restval", "train"]:
                for s in sentences:
                    train.append((image_path, s))
            elif image["split"] == "val":
                for s in sentences:
                    val.append((image_path, s))
            elif image["split"] == "test":
                for s in sentences:
                    test.append((image_path, s))

        self.pairs = train if split == "train" else val if split == "val" else test

        # Create the training text corpus vocabulary
        sentences = text_clean([" ".join(sample[1]) for sample in train])
        words = [w for s in sentences for w in s.split()]
        counts = sorted(Counter(words).items(), key=lambda x: x[1], reverse=True)[
            :vocab_size
        ]
        self.vocabulary = vocab(OrderedDict(counts))
        self.vocabulary.insert_token("<pad>", 0)
        self.vocabulary.insert_token("<unk>", 1)
        self.vocabulary.insert_token("<sos>", 2)
        self.vocabulary.insert_token("<eos>", 3)
        self.vocabulary.set_default_index(self.vocabulary["<unk>"])

        self.seq_len = seq_len
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_name, sentence = self.pairs[idx]
        image = Image.open(image_name).convert("RGB")
        text_init = text_clean([" ".join(sentence)])[0]
        text = self.vocabulary(text_init.split())
        sample = {
            "image": image,
            "text_init": text_init,
            "caption": [self.vocabulary["<sos>"]]
            + text[: self.seq_len - 2]
            + [self.vocabulary["<eos>"]]
            if len(text) >= self.seq_len
            else [self.vocabulary["<sos>"]]
            + text
            + [self.vocabulary["<eos>"]]
            + [self.vocabulary["<pad>"]] * (self.seq_len - len(text) - 2),
        }
        sample["caption"] = torch.tensor(sample["caption"])
        if self.transform:
            sample["image"] = self.transform(sample["image"])

        return sample
