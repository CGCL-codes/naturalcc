#!/usr/bin/env bash

if [ -z $NCC ]; then
  CACHE_DIR=~
else
  CACHE_DIR=$NCC
fi

data_names=py150
data_urls='http://files.srl.inf.ethz.ch/data/py150.tar.gz'
echo "Downloading py150 dataset"
DIR=${CACHE_DIR}/${data_names}/raw
mkdir -p ${DIR}
FILE=$DIR/py150.tar.gz

# download
if [ -f $FILE ]; then
  echo "$FILE exists"
else
  wget -P ${DIR} ${data_urls}
fi

# decompress
tar -zxvf ${DIR}/${data_names}.tar.gz -C ${DIR}
