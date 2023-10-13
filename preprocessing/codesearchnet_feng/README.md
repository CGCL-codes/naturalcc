# [CodeBert, Feng et. al.](https://arxiv.org/pdf/2002.08155.pdf)

Data statistic about the cleaned dataset for code document generation is shown in this Table. We release the cleaned dataset in this [website](https://drive.google.com/open?id=1rd2Tc6oUWBo7JouwexW3ksQ0PaOhUr6h).

| PL         | Training |  Dev   |  Test  |
| :--------- | :------: | :----: | :----: |
| Python     | 251,820  | 13,914 | 14,918 |
| PHP        | 241,241  | 12,982 | 14,014 |
| Go         | 167,288  | 7,325  | 8,122  |
| Java       | 164,923  | 5,183  | 10,955 |
| JavaScript |  58,025  | 3,885  | 3,291  |
| Ruby       |  24,927  | 1,400  | 1,261  |

The results on CodeSearchNet are shown in this Table:

| Model       |   Ruby    | Javascript |    Go     |  Python   |   Java    |    PHP    |  Overall  |
| ----------- | :-------: | :--------: | :-------: | :-------: | :-------: | :-------: | :-------: |
| Seq2Seq     |   9.64    |   10.21    |   13.98   |   15.93   |   15.09   |   21.08   |   14.32   |
| Transformer |   11.18   |   11.59    |   16.38   |   15.81   |   16.26   |   22.12   |   15.56   |
| RoBERTa     |   11.17   |   11.90    |   17.72   |   18.14   |   16.47   |   24.02   |   16.57   |
| CodeBERT    | **12.16** | **14.90**  | **18.07** | **19.06** | **17.65** | **25.16** | **17.83** |



----------------------------------------------------------------------------------------------------

**we recommend to run this repository on linux/macOS**

### step 1. download codesearchnet_feng raw dataset (```~/raw```)
```
bash dataset/codesearchnet_feng/download.sh
```

### step 2. flatten attributes of code snippets into different files. For instance, flatten ruby's code_tokens into 
```train/valid/test.code_tokens```.
```
python -m dataset.codesearchnet_feng.flatten -l [language] -d [raw data directory] -f [flatten data directory] -a [data attributes] -c [cpu cores]
```

### step 3(optional). extract features of data attributes. For instance, AST, binary-AST etc. of code.
```
python -m dataset.codesearchnet_feng.feature_extract -l [language] -f [flatten data directory] -s [parse file] -a [data attributes] -c [cpu cores]
```
 
