To reproduce the memefier environment run the following:
```
conda create -n memefier python=3.9
conda activate memefier
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch
pip install pandas matplotlib scikit-learn torchtext==0.13.0 ftfy regex tqdm huggingface-hub transformers boto3 requests sentencepiece sacremoses textdistance nltk git+https://github.com/openai/CLIP.git
```