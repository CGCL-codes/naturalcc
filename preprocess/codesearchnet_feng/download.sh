#!/usr/bin/env bash

if [ -z $NCC ]; then
  CACHE_DIR=~
else
  CACHE_DIR=$NCC
fi

DATASET_NAME=codesearchnet_feng
echo "Downloading Code Search Net(feng) dataset"
RAW_DIR=$CACHE_DIR/$DATASET_NAME/raw
mkdir -p $RAW_DIR

FILE=$RAW_DIR/Cleaned_CodeSearchNet.zip
echo $FILE
if [ -f $FILE ]; then
  echo "$FILE exists"
else
  echo "Downloading ${DATASET_NAME} dataset at ${FILE}"
  gdown https://drive.google.com/uc?id=1rd2Tc6oUWBo7JouwexW3ksQ0PaOhUr6h -O ${FILE} --no-cookies
fi

unzip ${FILE} -d ${RAW_DIR} # && rm ${FILE}

# remove dataset/lang at raw dir
mv $RAW_DIR/CodeSearchNet/* $RAW_DIR && rm -fr $RAW_DIR/CodeSearchNet

echo "Done"
