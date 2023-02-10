# MemeFier: Dual-stage modality fusion for image meme classification
The purpose of this work is to produce a model that can classify image memes in a
fine-grained manner to categories such as hateful, offensive, humorous, motivational etc.
We perform several experiments with baseline neural networks on standard benchmarks
to assess the minimum performance level. Additionally, we propose a novel 
deep-learning based architecture involving dual-stage modality fusion, 
external knowledge and caption supervision as regularisation factor.
The proposed model outperforms the baselines in most of the experiments.

## Datasets
* [Facebook Hatehul Memes](https://ai.facebook.com/blog/hateful-memes-challenge-and-data-set/ "Title")
* [Memotion7k](https://www.kaggle.com/datasets/williamscott701/memotion-dataset-7k "Title")
* [MultiOFF](https://drive.google.com/drive/folders/1hKLOtpVmF45IoBmJPwojgq6XraLtHmV6 "Title")

## Baselines
* (image only)Fine-tuning a pre-trained ResNet18
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