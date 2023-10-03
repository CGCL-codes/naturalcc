#!/usr/bin/env bash

if [ -z $NCC ]; then
  CACHE_DIR=~
else
  CACHE_DIR=$NCC
fi

DATASET_NAME=codesearchnet_feng
echo "Downloading ${DATASET_NAME} dataset for CodeXGlue"
DIR=$CACHE_DIR/codexglue/code_to_text/raw
mkdir -p ${DIR}

FILE=${DIR}/Cleaned_CodeSearchNet.zip

if [ -f $FILE ]; then
  echo "$FILE exists"
else
  url="https://drive.google.com/uc?id=1rd2Tc6oUWBo7JouwexW3ksQ0PaOhUr6h"
  gdown ${url} -O $FILE --no-cookies
fi

unzip $FILE -d ${DIR}
mv ${DIR}/CodeSearchNet/* ${DIR}
rm -rf ${DIR}/CodeSearchNet
