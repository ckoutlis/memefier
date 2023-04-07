import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
import clip

import math


def clip_vis_hook(module, input, output):
    global region_features
    region_features = output


def clip_text_hook(module, input, output):
    global word_features
    word_features = output


class MemeFier(nn.Module):
    def __init__(
        self,
        d=None,
        enc=None,
        dec=None,
        num_classes=1,
        components="",
        caption_max_len=None,
        caption_vocab_size=None,
        device="cuda:0",
    ):

        """
        MemeFier is a Transformer-based neural network architecture that utilizes dual-stage
        modality fusion for image meme classification.

        :param d: (int) the model's dimension
        :param enc: (dict) encoder hyperparameters, default {"h": 16, "dff": 2048, "L": 3}
        :param dec: (dict) decoder hyperparameters, default {"d": 64, "h": 4, "dff": 64, "L": 1}
        :param components:  (str) model components to include, default '', options:
                                E: External knowledge
                                C: Caption
                                P: Pretrained decoder
                                1: Fusion Stage 1
                                2: Fusion Stage 2
        :param caption_max_len: (int) maximum length of the target captions, default None
        :param caption_vocab_size: (int) size of the captions vocabulary, default None
        :param device: (str) device to use, default 'cuda'
        """
        super().__init__()
        if enc is None:
            enc = {"h": 16, "dff": 2048, "L": 3}
        if dec is None:
            dec = {"d": 64, "h": 4, "dff": 64, "L": 1}

        self.device = device
        self.components = components
        self.caption_max_len = caption_max_len
        self.caption_vocab_size = caption_vocab_size
        self.dec_dim = dec["d"]
        self.num_classes = num_classes

        # Encoding
        self.clip, self.preprocess = clip.load(
            "ViT-L/14", device=device
        )  # The CLIP model
        # Freeze CLIP
        for name, param in self.clip.named_parameters():
            param.requires_grad = False
        self.clip.visual.transformer.resblocks[23].ln_2.register_forward_hook(
            clip_vis_hook
        )  # A hook on the CLIP model (provides the activations of the visual part's last layer)
        self.clip.transformer.resblocks[11].ln_2.register_forward_hook(
            clip_text_hook
        )  # A hook on the CLIP model (provides the activations of the textual part's last layer)
        self.project_patches = nn.Sequential(
            *[nn.Linear(1024, d), nn.Dropout(p=0.4)]
        ).to(device)
        self.project_tokens = nn.Sequential(*[nn.Linear(768, d), nn.Dropout(p=0.4)]).to(
            device
        )
        self.project_image_embedding = nn.Sequential(
            *[nn.Linear(768, d), nn.Dropout(p=0.4)]
        ).to(device)
        self.project_text_embedding = nn.Sequential(
            *[nn.Linear(768, d), nn.Dropout(p=0.4)]
        ).to(device)
        if "E" in components:
            self.external_embedding = nn.Sequential(
                *[nn.Embedding(19, 32), nn.Dropout(p=0.5)]
            ).to(device)
            self.project_external = nn.Sequential(
                *[nn.Linear(32, d), nn.Dropout(p=0.5)]
            ).to(device)

        # Fusion
        self.cls = nn.Embedding(1, d).to(device)
        self.type_embedding = nn.Embedding(4, d).to(device)
        if "2" in components:
            self.fusion = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=d,
                    nhead=enc["h"],
                    dim_feedforward=enc["dff"],
                    dropout=0.5,
                    batch_first=True,
                ),
                num_layers=enc["L"],
            ).to(device)

        # Classification output
        if isinstance(num_classes, dict):
            self.fc_hum = nn.Sequential(
                *[nn.Dropout(p=0.1), torch.nn.Linear(d, num_classes["humour"])]
            ).to(device)
            self.fc_sar = nn.Sequential(
                *[nn.Dropout(p=0.1), torch.nn.Linear(d, num_classes["sarcasm"])]
            ).to(device)
            self.fc_off = nn.Sequential(
                *[nn.Dropout(p=0.1), torch.nn.Linear(d, num_classes["offensive"])]
            ).to(device)
            self.fc_mot = nn.Sequential(
                *[nn.Dropout(p=0.1), torch.nn.Linear(d, num_classes["motivational"])]
            ).to(device)
        elif isinstance(num_classes, int):
            self.fc_cls = nn.Sequential(
                *[nn.Dropout(p=0.1), nn.Linear(d, num_classes)]
            ).to(device)
        else:
            raise Exception(
                f"num_classes can only be either a dict or an int, {num_classes} is found"
            )

        # Caption supervision
        if "C" in components:
            self.linear_enc_dec = nn.Linear(d, dec["d"]).to(device)
            self.caption_vocab_emb = nn.Embedding(caption_vocab_size, dec["d"]).to(
                device
            )
            self.mem_pos_enc = nn.Embedding(257, dec["d"]).to(device)
            self.tgt_pos_enc = nn.Embedding(caption_max_len, dec["d"]).to(device)
            self.caption = nn.TransformerDecoder(
                nn.TransformerDecoderLayer(
                    d_model=dec["d"],
                    nhead=dec["h"],
                    dim_feedforward=dec["dff"],
                    batch_first=True,
                ),
                num_layers=dec["L"],
            ).to(device)

            if "P" in components:
                decoder = torch.load("ckpt/decoder.pt")
                with torch.no_grad():
                    # TODO: preserve the common word embeddings in self.caption_vocab_emb
                    self.mem_pos_enc.weight.copy_(decoder.mem_pos_enc.weight)
                    self.tgt_pos_enc.weight[:51, :].copy_(
                        decoder.tgt_pos_enc.weight[:caption_max_len, :]
                    )
                    for name, param in self.named_parameters():
                        if "caption." in name:
                            for x in range(dec["L"]):
                                name = name.replace(f".{x}.", f"[{x}].")
                            eval(f"self.{name}.copy_(decoder.{name})")

            self.fc_cap = nn.Linear(
                in_features=dec["d"], out_features=caption_vocab_size
            ).to(device)

    def forward(self, sample):
        # Inputs
        g = sample["image"].to(self.device)
        x = clip.tokenize(sample["text"], truncate=True).to(self.device)
        if "E" in self.components:
            e = sample["external_index"].to(self.device)

        # Encoding
        index_base = torch.ones(g.shape[0], dtype=torch.int).to(self.device)
        cls = F.normalize(
            self.cls(0 * index_base).unsqueeze(1),
            p=2,
            dim=2,
        )
        og = F.normalize(
            self.project_image_embedding(self.clip.encode_image(g).float()),
            p=2,
            dim=1,
        )
        ogi = F.normalize(
            self.project_patches(region_features.permute(1, 0, 2).float())
            + self.type_embedding(0 * index_base).unsqueeze(1),
            p=2,
            dim=2,
        )
        ox = F.normalize(
            self.project_text_embedding(self.clip.encode_text(x).float()),
            p=2,
            dim=1,
        )
        oxi = F.normalize(
            self.project_tokens(word_features.permute(1, 0, 2).float())
            + self.type_embedding(index_base).unsqueeze(1),
            p=2,
            dim=2,
        )
        if "E" in self.components:
            oei = F.normalize(
                self.project_external(self.external_embedding(e))
                + self.type_embedding(2 * index_base).unsqueeze(1),
                p=2,
                dim=2,
            )

        # Fusion (stage 1)
        fgi = torch.mul(ogi, ox.unsqueeze(1)) if "1" in self.components else ogi
        fxi = torch.mul(oxi, og.unsqueeze(1)) if "1" in self.components else oxi

        # Fusion (stage 2)
        fusion_input = torch.cat(
            (
                cls,
                fgi,
                fxi,
            ),
            dim=1,
        )
        if "E" in self.components:
            fusion_input = torch.cat(
                (
                    fusion_input,
                    oei,
                ),
                dim=1,
            )
        fusion_output = self.fusion(fusion_input) if "2" in self.components else fusion_input.mean(dim=1, keepdims=True)

        # Classification output
        if isinstance(self.num_classes, dict):
            logits = [
                self.fc_hum(fusion_output[:, 0, :]),
                self.fc_sar(fusion_output[:, 0, :]),
                self.fc_off(fusion_output[:, 0, :]),
                self.fc_mot(fusion_output[:, 0, :]),
            ]
        elif isinstance(self.num_classes, int):
            logits = self.fc_cls(fusion_output[:, 0, :])

        # Caption supervision
        if "C" in self.components:
            image_token_features = self.linear_enc_dec(
                fusion_output[:, 1 : ogi.shape[1] + 1, :]
            )
            tgt_indx = sample["caption_index"].to(self.device)
            tgt = self.caption_vocab_emb(tgt_indx) * math.sqrt(self.dec_dim)
            image_token_features += self.mem_pos_enc(
                torch.arange(image_token_features.shape[1]).to(self.device)
            ).unsqueeze(0)
            tgt += self.tgt_pos_enc(
                torch.arange(tgt_indx.shape[1]).to(self.device)
            ).unsqueeze(0)
            caption = self.fc_cap(
                self.caption(
                    tgt=tgt,
                    memory=image_token_features,
                    tgt_mask=self.get_tgt_mask(tgt_indx.shape[1]).to(self.device),
                    tgt_key_padding_mask=self.create_pad_mask(
                        matrix=tgt_indx, pad_token=0
                    ),
                )
            )

            return logits, caption

        return logits

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


class ImageOnly(torch.nn.Module):
    def __init__(self, pretrained, hidden_dim, num_classes, device):
        super().__init__()
        self.device = device
        self.num_classes = num_classes

        img_fe = resnet18(pretrained=pretrained)
        img_fe.fc = torch.nn.Identity()
        self.img_fe = img_fe
        self.fc_img1 = torch.nn.Linear(512, hidden_dim * 2)

        if isinstance(num_classes, dict):
            self.fc_hum = torch.nn.Linear(hidden_dim * 2, num_classes["humour"])
            self.fc_sar = torch.nn.Linear(hidden_dim * 2, num_classes["sarcasm"])
            self.fc_off = torch.nn.Linear(hidden_dim * 2, num_classes["offensive"])
            self.fc_mot = torch.nn.Linear(hidden_dim * 2, num_classes["motivational"])
        elif isinstance(num_classes, int):
            self.fc = torch.nn.Linear(hidden_dim * 2, num_classes)
        else:
            raise Exception(
                f"num_classes can only be either a dict or an int, {num_classes} is found"
            )

    def forward(self, x):
        y = torch.tanh(self.fc_img1(self.img_fe(x["image"].to(self.device))))
        if isinstance(self.num_classes, dict):
            return [
                self.fc_hum(y),
                self.fc_sar(y),
                self.fc_off(y),
                self.fc_mot(y),
            ]
        elif isinstance(self.num_classes, int):
            return self.fc(y)


class TextOnly(torch.nn.ModuleList):
    def __init__(self, max_words, hidden_dim, lstm_layers, num_classes, device):
        super(TextOnly, self).__init__()
        self.device = device
        self.num_classes = num_classes

        # Hyperparameters
        self.input_size = max_words
        self.hidden_dim = hidden_dim
        self.LSTM_layers = lstm_layers

        self.dropout = torch.nn.Dropout(0.5)
        self.embedding = torch.nn.Embedding(
            self.input_size, self.hidden_dim, padding_idx=0
        )
        self.lstm = torch.nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.LSTM_layers,
            batch_first=True,
        )
        self.fc1 = torch.nn.Linear(
            in_features=self.hidden_dim, out_features=self.hidden_dim * 2
        )
        if isinstance(num_classes, dict):
            self.fc_hum = torch.nn.Linear(self.hidden_dim * 2, num_classes["humour"])
            self.fc_sar = torch.nn.Linear(self.hidden_dim * 2, num_classes["sarcasm"])
            self.fc_off = torch.nn.Linear(self.hidden_dim * 2, num_classes["offensive"])
            self.fc_mot = torch.nn.Linear(
                self.hidden_dim * 2, num_classes["motivational"]
            )
        elif isinstance(num_classes, int):
            self.fc2 = torch.nn.Linear(self.hidden_dim * 2, num_classes)
        else:
            raise Exception(
                f"num_classes can only be either a dict or an int, {num_classes} is found"
            )

    def forward(self, x):
        # Hidden and cell state definion
        h = torch.zeros(
            (self.LSTM_layers, x["text_index"].to(self.device).size(0), self.hidden_dim)
        ).to(self.device)
        c = torch.zeros(
            (self.LSTM_layers, x["text_index"].to(self.device).size(0), self.hidden_dim)
        ).to(self.device)

        # Initialization of hidden and cell states
        torch.nn.init.xavier_normal_(h)
        torch.nn.init.xavier_normal_(c)

        out = self.embedding(x["text_index"].to(self.device))
        out, (hidden, cell) = self.lstm(out, (h, c))
        out = self.dropout(out)
        out = torch.relu_(self.fc1(out[:, -1, :]))
        out = self.dropout(out)
        if isinstance(self.num_classes, dict):
            return [
                self.fc_hum(out),
                self.fc_sar(out),
                self.fc_off(out),
                self.fc_mot(out),
            ]
        elif isinstance(self.num_classes, int):
            return self.fc2(out)


class MultiModal(torch.nn.Module):
    def __init__(
        self, pretrained, max_words, hidden_dim, lstm_layers, num_classes, device
    ):
        super().__init__()
        self.device = device
        self.num_classes = num_classes

        img_fe = ImageOnly(pretrained, hidden_dim, 1, device)
        img_fe.fc = torch.nn.Identity()
        self.img_fe = img_fe

        txt_fe = TextOnly(max_words, hidden_dim, lstm_layers, 1, device)
        txt_fe.fc2 = torch.nn.Identity()
        self.txt_fe = txt_fe

        self.fc_img1 = torch.nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.fc_img2 = torch.nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc_txt1 = torch.nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.fc_txt2 = torch.nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc_multi1 = torch.nn.Linear(hidden_dim * 2, hidden_dim)
        if isinstance(num_classes, dict):
            self.fc_hum = torch.nn.Linear(hidden_dim, num_classes["humour"])
            self.fc_sar = torch.nn.Linear(hidden_dim, num_classes["sarcasm"])
            self.fc_off = torch.nn.Linear(hidden_dim, num_classes["offensive"])
            self.fc_mot = torch.nn.Linear(hidden_dim, num_classes["motivational"])
        elif isinstance(num_classes, int):
            self.fc_multi2 = torch.nn.Linear(hidden_dim, num_classes)
        else:
            raise Exception(
                f"num_classes can only be either a dict or an int, {num_classes} is found"
            )

    def forward(self, x):
        y_img = F.relu(self.fc_img2(torch.tanh(self.fc_img1(self.img_fe(x)))))
        y_txt = F.relu(self.fc_txt2(torch.tanh(self.fc_txt1(self.txt_fe(x)))))
        y = F.relu(self.fc_multi1(torch.concat([y_img, y_txt], dim=-1)))
        if isinstance(self.num_classes, dict):
            return [
                self.fc_hum(y),
                self.fc_sar(y),
                self.fc_off(y),
                self.fc_mot(y),
            ]
        return self.fc_multi2(y)
