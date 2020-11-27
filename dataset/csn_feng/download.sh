#!/usr/bin/env bash


echo "Downloading Code Search Net(feng) dataset"
DIR=~/.ncc/code_search_net_feng/raw/
mkdir -p ${DIR}
FILE=${DIR}code_search_net_feng.zip

gdown https://drive.google.com/uc?id=1rd2Tc6oUWBo7JouwexW3ksQ0PaOhUr6h -O ${FILE} --no-cookies
unzip ${FILE} -d ${DIR} # && rm ${FILE}

