# Dataset: Java(hu)

The authors of [A Transformer-based Approach for Source Code Summarizatio
n](https://arxiv.org/pdf/2005.00653.pdf) shared their [code and dataset](https://github.com/wasiahmad/NeuralCodeSum). 
In this repo., it offers original and runnable codes of Java dataset and therefore we can generate AST with Tree-Sitter.

However, as for Python dataset, its original codes are not runnable. An optional way to deal with such problem is that
  we can acquire runnable Python codes from [raw data](https://github.com/wanyao1992/code_summarization_public).

<hr>

# Step 1 
Download pre-processed and raw (java_hu) dataset.
```shell script
bash dataset/java_hu/download.sh
```

# Step 2
Move **code/code_tokens/docstring/docstring_tokens** to ```~/java_hu/flatten/*```.
```shell script
python -m dataset.java_hu.flatten
```

# Step 3
Generating raw/bin data with multi-processing. 
Before generating datasets, plz make sure [config file](./dataset/java_hu/config/preprocess.yml) is set correctly.
```shell script
# code_tokens/docstring_tokens
python -m dataset.java_hu.preprocess
```
