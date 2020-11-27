#!/usr/bin/env bash

data_names=(
  "java"
  "javascript"
  "go"
  "php"
  "python"
  "ruby"
)

echo "Downloading CodeSearchNet dataset"
DIR=~/.ncc/code_search_net/raw/
mkdir -p ${DIR}

for (( idx = 0 ; idx < ${#data_names[@]} ; idx++ )); do

FILE=${DIR}${data_names[idx]}.zip
echo "Downloading CodeSearchNet dataset file from ${data_names[idx]}"
wget https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/${data_names[idx]}.zip -P ${FILE}
gunzip ${FILE} -d ${DIR} # && rm ${FILE}

done

