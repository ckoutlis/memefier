# Reproduce meme-classification environment
```
conda create -n meme-classification python=3.9
conda activate meme-classification
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install pandas
pip install matplotlib
pip install sklearn
pip install torchtext
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
pip install huggingface-hub
pip install transformers
pip install tqdm boto3 requests regex sentencepiece sacremoses
pip install textdistance
pip install nltk
```
