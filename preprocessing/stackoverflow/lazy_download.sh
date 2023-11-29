#!/usr/bin/env bash

if [ -z $NCC ]; then
  CACHE_DIR=~
else
  CACHE_DIR=$NCC
fi

data_names=stackoverflow
data_urls="https://drive.google.com/uc?id=1KpKepv4VI5VJmb4YDeV2X2pIAsuzw0fZ"
echo "Downloading processed StackOverflow dataset"
DIR=$CACHE_DIR/${data_names}

FILE=${DIR}/stackoverflow.zip
gdown ${data_urls} -O ${FILE} --no-cookies
unzip ${FILE} -d ${DIR}

