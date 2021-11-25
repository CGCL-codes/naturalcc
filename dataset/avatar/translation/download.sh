#!/usr/bin/env bash

if [ -z $NCC ]; then
  CACHE_DIR=~
else
  CACHE_DIR=$NCC
fi

data_urls="https://github.com/wasiahmad/AVATAR/blob/main/data/data.zip"
echo "Downloading AVATAR dataset"
DIR=$CACHE_DIR/avatar/raw
mkdir -p ${DIR}

FILE=${DIR}/data.zip
wget ${data_urls} -O ${FILE}
unzip ${FILE} -d ${DIR} # && rm ${FILE}
