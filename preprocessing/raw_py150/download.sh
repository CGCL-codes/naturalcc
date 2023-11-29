#!/usr/bin/env bash

if [ -z $NCC ]; then
  CACHE_DIR=~
else
  CACHE_DIR=$NCC
fi

data_names=raw_py150
data_urls='http://files.srl.inf.ethz.ch/data/py150_files.tar.gz'
echo "Downloading raw_py150 dataset"
DIR=${CACHE_DIR}/${data_names}/raw
mkdir -p ${DIR}
FILE=${DIR}/py150_files.tar.gz

# download
if [ -f $FILE ]; then
  echo "$FILE exists"
else
  wget -P ${DIR} ${data_urls}
fi

# decompress
tar -zxvf $FILE -C ${DIR}
tar -zxvf ${DIR}/data.tar.gz -C ${DIR}

echo "raw_py150 dataset downloaded to ${FILE}"
