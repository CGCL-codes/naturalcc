#!/usr/bin/env bash

if [ -z $NCC ]; then
  CACHE_DIR=~/ncc_data
else
  CACHE_DIR=$NCC/ncc_data
fi

data_names=python
data_urls="https://drive.google.com/uc?id=1kpzGHybDmw-PwYU8v8Kh86Xjh1cwEaDb"
echo "Downloading python-wan dataset"
DIR=$CACHE_DIR/python_wan/raw
mkdir -p ${DIR}

FILE=${DIR}/${data_names}.zip
gdown ${data_urls} -O ${FILE} --no-cookies
unzip ${FILE} -d ${DIR} # && rm ${FILE}

# rename dev to valid
mv ${DIR}dev ${DIR}valid

# raw file from
wget -P ${DIR} https://raw.githubusercontent.com/wanyao1992/code_summarization_public/master/dataset/original/data_ps.declbodies