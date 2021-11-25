#!/usr/bin/env bash


echo "Downloading Code Search Net(feng) dataset"
DIR=~/code_search_net_feng/raw/retrieval
mkdir -p ${DIR}
FILE=${DIR}codesearch_data.zip

gdown https://drive.google.com/uc?id=1xgSR34XO8xXZg4cZScDYj2eGerBE9iGo -O ${FILE} --no-cookies
unzip ${FILE} -d ${DIR} # && rm ${FILE}

