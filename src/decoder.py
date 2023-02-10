import torch
import math
import clip
import json
from PIL import Image
from nltk.translate.bleu_score import corpus_bleu
from src.data import COCO
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


def clip_vis_hook(module, input, output):
    global region_features
    region_features = output


class Decoder(torch.nn.Module):
    def __init__(
        self,
        device,
        hyperparams,
        caption_max_len,
        caption_vocab_size,
    ):
        super().__init__()
        self.device = device
        self.caption_max_len = caption_max_len
        self.caption_vocab_size = caption_vocab_size
        self.model_dim = hyperparams["model_dim"]

        self.clip, self.preprocess = clip.load(
            "ViT-B/32", device=device
        )  # The CLIP model
        for param in self.clip.parameters():
            param.requires_grad = False
        self.clip.visual.transformer.resblocks[11].ln_2.register_forward_hook(
            clip_vis_hook
        )  # A hook on the CLIP model (provides the activations of the visual part's last layer)

        self.fc_image = torch.nn.Linear(
            in_features=768, out_features=hyperparams["model_dim"]
        ).to(device)
        self.caption_vocab_emb = torch.nn.Embedding(
            caption_vocab_size, hyperparams["model_dim"]
        ).to(device)
        self.mem_pos_enc = torch.nn.Embedding(50, hyperparams["model_dim"]).to(device)
        self.tgt_pos_enc = torch.nn.Embedding(
            caption_max_len, hyperparams["model_dim"]
        ).to(device)
        self.caption = torch.nn.TransformerDecoder(
            torch.nn.TransformerDecoderLayer(
                d_model=hyperparams["model_dim"],
                nhead=hyperparams["nhead"],
                dim_feedforward=hyperparams["dim_feedforward"],
                batch_first=True,
            ),
            num_layers=hyperparams["num_layers"],
        ).to(device)
        self.fc = torch.nn.Linear(
            in_features=hyperparams["model_dim"], out_features=caption_vocab_size
        ).to(device)

    def forward(self, x):
        with torch.no_grad():
            _ = self.clip.encode_image(x["image"].to(self.device))
            image_token_features = self.fc_image(
                region_features.permute(1, 0, 2).float()
            )

        tgt_indx = x["caption"].to(self.device)
        tgt = self.caption_vocab_emb(tgt_indx) * math.sqrt(self.model_dim)
        image_token_features += self.mem_pos_enc(
            torch.arange(image_token_features.shape[1]).to(self.device)
        ).unsqueeze(0)
        tgt += self.tgt_pos_enc(
            torch.arange(tgt_indx.shape[1]).to(self.device)
        ).unsqueeze(0)
        caption = self.fc(
            self.caption(
                tgt=tgt,
                memory=image_token_features,
                tgt_mask=self.get_tgt_mask(tgt_indx.shape[1]).to(self.device),
                tgt_key_padding_mask=self.create_pad_mask(matrix=tgt_indx, pad_token=0),
            )
        )

        return caption

    def get_tgt_mask(self, size) -> torch.tensor:
        # Generates a squeare matrix where the each row allows one word more to be seen
        mask = torch.tril(torch.ones(size, size) == 1)  # Lower triangular matrix
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float("-inf"))  # Convert zeros to -inf
        mask = mask.masked_fill(mask == 1, float(0.0))  # Convert ones to 0

        # EX for size=5:
        # [[0., -inf, -inf, -inf, -inf],
        #  [0.,   0., -inf, -inf, -inf],
        #  [0.,   0.,   0., -inf, -inf],
        #  [0.,   0.,   0.,   0., -inf],
        #  [0.,   0.,   0.,   0.,   0.]]

        return mask

    def create_pad_mask(self, matrix: torch.tensor, pad_token: int) -> torch.tensor:
        # If matrix = [1,2,3,0,0,0] where pad_token=0, the result mask is
        # [False, False, False, True, True, True]
        return matrix == pad_token


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


def predict_sentence(model, data, voc, length=51, sos=2, eos=3):
    captions = []
    for i in range(data["image"].shape[0]):
        y = torch.tensor([[sos]], dtype=torch.int, device="cuda")
        for _ in range(length):
            next = (
                model({"image": data["image"][i, :, :, :].unsqueeze(0), "caption": y})
                .max(dim=-1)[1][:, -1]
                .view(-1, 1)
            )
            y = torch.cat((y, next), dim=1)
            if next[0][0] == eos:
                captions.append(voc.lookup_tokens(y.tolist()[0][1:-1]))
                break
    return captions


def evaluate_model(
    model,
    split,
    voc,
    transform,
    print_txt,
    weights=None,
):
    directory = "/home/ckoutlis/disk_2_ubuntu/home/ckoutlis/DataStorage/coco"
    images = json.load(open(f"{directory}/split/dataset_coco.json"))["images"]
    actual, predicted = list(), list()
    for image in images:
        if image["split"] == split:
            image_path = f"{directory}/{image['filepath']}/{image['filename']}"
            sentences = [s["tokens"] for s in image["sentences"]]
            data = {
                "image": transform(Image.open(image_path).convert("RGB")).unsqueeze(0)
            }
            yhat = predict_sentence(model, data, voc)
            actual.append(sentences)
            predicted.append(yhat[0] if yhat else yhat)
            print(f"\r{print_txt}bleu calcutation: [{len(predicted)}]", end="")
    if weights is None:
        weights = [
            (1.0, 0, 0, 0),
            (0.5, 0.5, 0, 0),
            (0.3, 0.3, 0.3, 0),
            (0.25, 0.25, 0.25, 0.25),
        ]
    return [corpus_bleu(actual, predicted, weights=weights_) for weights_ in weights]


def generate_loaders(vocab_size, seq_len, batch_size):
    imnet_mean = [0.485, 0.456, 0.406]
    imnet_std = [0.229, 0.224, 0.225]

    train_ds = COCO(
        split="train",
        vocab_size=vocab_size,
        seq_len=seq_len,
        transform=transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.RandomCrop(224),
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
    val_ds = COCO(
        split="val",
        vocab_size=vocab_size,
        seq_len=seq_len,
        transform=transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(imnet_mean, imnet_std),
            ]
        ),
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=False,
    )
    test_ds = COCO(
        split="test",
        vocab_size=vocab_size,
        seq_len=seq_len,
        transform=transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(imnet_mean, imnet_std),
            ]
        ),
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=False,
    )

    return train_dl, val_dl, test_dl, train_ds, val_ds, test_ds
