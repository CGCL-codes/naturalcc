#!/usr/bin/env bash

if [ -z $NCC ]; then
  CACHE_DIR=~/ncc_data
else
  CACHE_DIR=$NCC/ncc_data
fi
CACHE_DIR=/mnt/wanyao/zsj/ncc_data
DATASET_NAME=codesearchnet
echo "Downloading CodeSearchNet dataset"
RAW_DIR=$CACHE_DIR/$DATASET_NAME/raw
mkdir -p $RAW_DIR

langs=(
  "ruby"
  "java"
  "javascript"
  "go"
  "php"
  "python"
)

for ((idx = 0; idx < ${#langs[@]}; idx++)); do
  FILE=$RAW_DIR/${langs[idx]}.zip
  echo $FILE
  if [ -f $FILE ]; then
    echo "$FILE exists"
    rm -fr $RAW_DIR/${langs[idx]}
    rm $RAW_DIR/${langs[idx]}*.pkl
  else
    echo "Downloading ${DATASET_NAME} dataset at ${FILE}"
    gdown "https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/${langs[idx]}.zip" -O $FILE --no-cookies
  fi
  unzip $FILE -d $RAW_DIR
  # rm $FILE
  rm $RAW_DIR/${langs[idx]}_licenses.pkl
  mv $RAW_DIR/${langs[idx]}/final/jsonl/* $RAW_DIR/${langs[idx]}
  rm -fr $RAW_DIR/${langs[idx]}/final
done
