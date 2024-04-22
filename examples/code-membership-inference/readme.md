# Does Your Neural Code Completion Model Use My Code? A Membership Inference Approach
## Requirements
- python3.7 or higher.
- Pytorch 1.10.1
- transformers 2.5.0
## Dataset
We use the py150 dataset from Raychev's OOPSLA 2016 paper [ Probabilistic Model for Code with Decision Trees](https://files.sri.inf.ethz.ch/website/papers/oopsla16-dt.pdf).
To download the dataset:
```shell
bash download_and_extract.sh
```
## Membership Inference
### CodeGPT Model
1. Data split and preprocess
```shell
cd CodeGPT/dataset/py150/
python load_py150.py
```
2. Model training
```shell
cd CodeGPT/code/
bash train.sh
```
3. Load the rank set
```shell
cd CodeGPT/code/
bash load_rnak.sh
```
### LSTM Model
1. Data preprocess(data has been splited when preprocess the CodeGPT dataset)
```shell
cd LSTM/seq/
bash generate_data.sh
cd ..
bash generate_vocab.sh
```
2. Model training
```shell
cd LSTM
bash train.sh
```
3. Load the rank set
```shell
cd LSTM
bash load_rank.sh
```
### Membership Inference Model
```shell
cd inference
bash inference.sh
```

