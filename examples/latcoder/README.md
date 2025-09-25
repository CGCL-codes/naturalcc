# [LaTCoder](https://github.io.latcoder/) &middot; ![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)

LaTCoder is a method for converting webpage designs into code (Design-to-Code) using **Layout-as-Thought** (LaT):

* **Performance:** LaTCoder significantly outperforms state-of-the-art methods, including [direct prompting](https://github.com/NoviScl/Design2Code), [text-augmented](https://github.com/NoviScl/Design2Code), [self-revision](https://github.com/NoviScl/Design2Code), and [DCGen](https://github.com/WebPAI/DCGen).

* **Compatibilities:** This method is compatible with both strong MLLMs (e.g., GPT-4o and Gemini), weaker MLLMs (e.g., DeepSeek-VL2), and task-specific VLMs (e.g., WebSight-8B and Design2Code-18B).

## Installation

To use LaTCoder, you need to set up a Python environment (python=3.10) as follows:
```bash
pip install -r requirements.txt
```

## Generate Webpages from Datasets

Currently, we support two datasets: [Design2Code-HARD](https://huggingface.co/datasets/SALT-NLP/Design2Code-HARD) and CC-HARD.  
For the Design2Code-HARD dataset, download and unzip it into the `data/` directory.  
You can add your own datasets by modifying `my_datasets.py`.

To generate webpages from the Design2Code-HARD dataset using GPT-4o:
```python
python main.py --dataset 0 --backbone 0
```

**Note:**  
You need to configure the API key for GPT-4o in `vendors/openai__.py` first. Additionally, you can define other backbone MLLMs in the `vendors` directory.  
For using other datasets, backbone MLLMs, or additional command-line parameters, please refer to `main.py`.

## Run Baselines

We include three baseline methods from [Design2Code](https://github.com/NoviScl/Design2Code):
- Direct-prompting
- Text-augmented
- Self-revision

These methods use exactly the same prompts and settings as in the original repo, with some modifications (`baselines`) to ensure compatibility with all MLLMs in `vendors`.

We also include another baseline method from [DCGen](https://github.com/WebPAI/DCGen):
- DCGen

For this, use `max_depth=1` as specified in the original paper. Please clone and run the code from its repository.

Given that Design2CodeHF and WebCode2M have demonstrated GPT-4o’s superior performance over task-specific models like WebSight-8B, Design2Code-18B, and WebCoder, we have excluded these models from our baseline comparisons.

## Run Evaluation Using Automatic Metrics

To evaluate the results using automatic metrics, run the following:
```python
python evaluate.py -i input_dir -o output_dir
```

We include various automatic metrics for evaluation, including MAE, visual score, CLP, and TreeBLEU (`tree_rouge_1`).

### License

This project is [MIT licensed](./LICENSE).


### Citation

@inproceedings{yi2025latoder,
author = {Yi Gui, Zhen Li, Zhongyi Zhang, Guohao Wang, Tianpeng Lv, Gaoyang Jiang, Yi Liu, Dongping Chen, Yao Wan, Hongyu Zhang, Wenbin Jiang, Xuanhua Shi, Hai Jin},
title = {LaTCoder: Converting Webpage Design to Code with Layout-as-Thought},
year = {2025},
isbn = {9798400714542},
publisher = {Association for Computing Machinery},
doi = {10.1145/3711896.3737016},
booktitle = {Proceedings of the 31st ACM SIGKDD Conference on Knowledge
Discovery and Data Mining V.2 (KDD '25)},
keywords = {UI Automation, Code Generation, Design to Code},
location = {Toronto, ON, Canada},
series = {KDD '25}
}
