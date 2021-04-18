# Preprocess for CodeBert

### Step 1: get special symbols
```
python -m dataset.csn.codebert.get_special_symbols
```

### Step 2: generate BPE dictionary
```
python -m dataset.csn.codebert.run_sentencepiece --src-dir ~/.ncc/CodeSearchNet/flatten --tgt-dir ~/.ncc/CodeSearchNet/codebert/data-raw --vocab-size 50000 --model-type bpe --language ruby --model-prefix codesearchnet
```

### Step 3: generate mmap data
```
python -m dataset.csn.codebert.preprocess_codebert.py # from data-raw to data-mmap
```