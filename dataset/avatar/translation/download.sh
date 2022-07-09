#!/usr/bin/env bash

if [ -z $NCC ]; then
  # CACHE_DIR=~
  echo "The environment variable '\$NCC' is not set, please read the docs."
  exit 1
else
  CACHE_DIR=$NCC
fi

data_urls="https://raw.githubusercontent.com/wasiahmad/AVATAR/main/data/data.zip"
echo "Downloading AVATAR dataset"
DIR=$CACHE_DIR/avatar/raw
mkdir -p ${DIR}

FILE=${DIR}/data.zip
wget ${data_urls} -O ${FILE}
unzip ${FILE} -d ${DIR} # && rm ${FILE}
rm ${FILE}
echo "AVATAR dataset download finished"
