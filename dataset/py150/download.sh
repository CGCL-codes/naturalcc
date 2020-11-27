#!/usr/bin/env bash

data_names="py150"
data_urls="http://files.srl.inf.ethz.ch/data/py150.tar.gz"

echo "Downloading py150 dataset"
DIR=~/.ncc/py150/raw/
mkdir -p ${DIR}
# download
wget -P ${DIR} ${data_urls}
# decompress
cd ${DIR}
tar -zxvf ${DIR}/${data_names}.tar.gz
