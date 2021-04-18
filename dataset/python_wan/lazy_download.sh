#!/usr/bin/env bash

if [ -z $NCC ]; then
  CACHE_DIR=~/ncc_data
else
  CACHE_DIR=$NCC/ncc_data
fi

data_names=python_wan
data_urls="https://drive.google.com/uc?id=1hhTM7Inx-90j-TwcgPZByWcFl7CoPXj2"
echo "Downloading our processed python-wan dataset"
DIR=$CACHE_DIR/python_wan
mkdir -p ${DIR}

FILE=${DIR}/${data_names}.tar.gz
gdown ${data_urls} -O ${FILE} --no-cookies

tar -zxvf ${FILE} -C ${DIR} # && rm ${FILE}