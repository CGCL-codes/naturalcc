# Code Summarization

### Step 1: Download the dataset
```
bash download.sh
```

### Step 2: Flatten the dataset

```
python -m preprocessing.codesearchnet.attributes_cast --languages ruby \
 --raw-dataset-dir /home/wanyao/codesearchnet/raw \
 --attributes-dir /home/wanyao/codesearchnet/attributes \
 --attrs code_tokens docstring_tokens \
 --cores 4
```
### Step 3: Binarize the dataset
```
ncc-preprocess --source-lang code_tokens --target-lang docstring_tokens --trainpref /home/wanyao/codesearchnet/attributes/ruby/train --testpref /home/wanyao/codesearchnet/attributes/ruby/test --validpref /home/wanyao/codesearchnet/attributes/ruby/valid --destdir /mnt/silver/yanrunbang/naturalcc/data-bin/codesearchnet --workers 4
```

### Step 4: Postprocess the dataset for summarization task
python -m preprocessing.codesearchnet.summarization.postprocess --dataset-dir /mnt/silver/yanrunbang/naturalcc/data-bin/codesearchnet --source-lang code_tokens --target-lang docstring_tokens

### Step 5: Train your model
```
ncc-train --task summarization \
  /mnt/silver/yanrunbang/naturalcc/data-bin/codesearchnet \
  --save-dir checkpoints/seq2seq_codesearchnet \
  --dataset-impl mmap \
  --arch seq2seq_summarization \
  --dropout 0.1 \
  --optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay 0.01 --clip-norm 0.0 \
  --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07 \
  --max-tokens 2048 --update-freq 16 \
  --fp16 \
  --max-update 50000
```