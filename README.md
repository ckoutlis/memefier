# MemeFier: Dual-stage Modality Fusion for Image Meme Classification

Implementation of the corresponding ICMR 2023 paper which is available here https://dl.acm.org/doi/abs/10.1145/3591106.3592254.

## Abstract
> Hate speech is a societal problem that has significantly grown through the Internet. New forms of digital content such as image memes have given rise to spread of hate using multimodal means, being far more difficult to analyse and detect compared to the unimodal case. Accurate automatic processing, analysis and understanding of this kind of content will facilitate the endeavor of hindering hate speech proliferation through the digital world. To this end, we propose MemeFier, a deep learning-based architecture for fine-grained classification of Internet image memes, utilizing a dual-stage modality fusion module. The first fusion stage produces feature vectors containing modality alignment information that captures non-trivial connections between the text and image of a meme. The second fusion stage leverages the power of a Transformer encoder to learn inter-modality correlations at the token level and yield an informative representation. Additionally, we consider external knowledge as an additional input, and background image caption supervision as a regularizing component. Extensive experiments on three widely adopted benchmarks, i.e., Facebook Hateful Memes, Memotion7k and MultiOFF, indicate that our approach competes and in some cases surpasses state-of-the-art.

![](https://github.com/ckoutlis/memefier/blob/master/docs/fig1.png)

## Datasets
* [Facebook Hatehul Memes](https://ai.facebook.com/blog/hateful-memes-challenge-and-data-set/ "Title")
* [Memotion7k](https://www.kaggle.com/datasets/williamscott701/memotion-dataset-7k "Title")
* [MultiOFF](https://drive.google.com/drive/folders/1hKLOtpVmF45IoBmJPwojgq6XraLtHmV6 "Title")

## Baselines
* (image only) Fine-tuning a pre-trained ResNet18
* (text only) LSTM based
* (multi-modal) Combination of the two above

## Results
Performance on Facebokk Hateful Memes in terms of accuracy and AUC:

| method   | accuracy  | AUC       |
|----------|-----------|-----------|
| image    | 0.530     | 0.573     |
| text     | 0.544     | 0.622     |
| multi    | 0.554     | 0.613     |
| MemeFier | **0.736** | **0.801** |

Performance on Memotion7k's three tasks (a,b, and c) in terms of average macro 
F1 score:

|            | a         | b         | c         |
|------------|-----------|-----------|-----------|
| image      | 0.333     | 0.502     | 0.315     |
| text       | 0.350     | 0.481     | 0.279     |
| multimodal | 0.346     | 0.493     | 0.310     |
| MemeFier   | **0.396** | **0.519** | **0.343** |

Performance on MultiOFF in terms of accuracy and F1 score:

| method     | accuracy  | F1        |
|------------|-----------|-----------|
| image      | 0.638     | 0.619     |
| text       | 0.571     | 0.508     |
| multimodal | 0.671     | **0.626** |
| MemeFier   | **0.685** | 0.625     |

## Citation
If you use this code for your research, please cite our paper.
```
@inproceedings{koutlis2023memefier,
  title={MemeFier: Dual-stage Modality Fusion for Image Meme Classification},
  author={Koutlis, Christos and Schinas, Manos and Papadopoulos, Symeon},
  booktitle={Proceedings of the 2023 ACM International Conference on Multimedia Retrieval},
  pages={586--591},
  year={2023}
}
```

## Contact
Christos Koutlis (ckoutlis@iti.gr)
