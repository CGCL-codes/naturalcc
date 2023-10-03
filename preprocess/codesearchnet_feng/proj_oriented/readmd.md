# Dataset: CodeSearchNet(feng)

The authors of [CodeBERT:
A Pre-Trained Model for Programming and Natural Languages](https://arxiv.org/pdf/2002.08155.pdf) shared their [code and dataset](https://github.com/microsoft/CodeBERT). 

<hr>

## Step 1: Download pre-processed and raw (code_search_net_feng) dataset.
```shell script
bash dataset/csn_feng/download.sh
```

## Step 2: Build project-oriented data into ```~/code_search_net_feng/proj/raw/*```.
```shell script
python -m dataset.csn_feng.proj_oriented.build
```

## Step 3: Flatten attributes of code into ```~/code_search_net_feng/attributes_cast/*```.
```shell script
python -m dataset.csn_feng.attributes_cast -d ~/code_search_net_feng/proj/raw -f ~/code_search_net_feng/proj/attributes_cast -a code code_tokens docstring docstring_tokens repo
```

### Step 4
Generating raw/bin data with multi-processing. 
Before generating datasets, plz make sure [config file](./config/ruby.yml) is set correctly.  Here we use csn_feng_ruby as exmaple.
```shell script
# code_tokens/docstring_tokens
python -m dataset.csn_feng.proj_oriented.preprocess -f config/ruby
```