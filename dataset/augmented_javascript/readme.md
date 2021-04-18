
## Step 0: Download augmented_javascript dataset realeased in https://github.com/parasj/contracode. 
```
bash download.sh 
```

## Step 1: Cast the downloaded target_dict to support our scenario
```
python -m dataset.augmented_javascript.cast_target_dict 
```

## Step 1: Cast the downloaded type_prediction_data to support our scenario
```
python -m dataset.augmented_javascript.cast_type_prediction_data
```

## Step 1: Cast downloaded `.pkl` file to `.json` for data binarization (mmap) (~1min).
```
python -m dataset.augmented_javascript.cast_pkl2json
```

## Step 2: before you run sentencepiece, you have gunzip raw data
```
cd ~/.ncc/augmented_javascript/raw
gunzip javascript_dedupe_definitions_nonoverlap_v2_train.jsonl.gz
```

## Step 3: Run sentencepiece to obtain the vocabulary and corresponding model (~20min).
Run the sentencepiece
```
python -m dataset.augmented_javascript.run_sentencepiece
```
Cast the sentencepiece vocab to the format of NCC Dictionary.
```
cd ~/.ncc/augmented_javascript/codebert/code_roberta/data-mmap 
cut -f1 csnjs_8k_9995p_unigram_url.vocab | tail -n +10 | sed "s/$/ 100/g" > csnjs_8k_9995p_unigram_url.dict.txt
```

## Step 4: Preprocessing (< 1 h).
> Note: currently only 100 samples are preprocessed for debugging. Modify around line 123 of ```preprocess.py```.

If we want to pretrain the codebert, we will use this data. Check the paths in `preprocess.yml` before running this command.
```
python -m dataset.augmented_javascript.preprocess
```

(Not for current) If we want to pretrain via contrastive learning, we should use this dataset with augmentation.
```
python -m dataset.augmented_javascript.preprocess_augmented
```