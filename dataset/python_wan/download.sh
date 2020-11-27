#!/usr/bin/env bash

data_names=(
  "python.zip"
)
data_urls=(
  "https://drive.google.com/uc?id=1XPE1txk9VI0aOT_TdqbAeI58Q8puKVl2"
)

echo "Downloading python-wan dataset"
DIR=~/.ncc/python_wan/raw/
mkdir -p ${DIR}

for (( idx = 0 ; idx < ${#data_urls[@]} ; idx++ )); do

FILE=${DIR}${data_names[idx]}
echo "Downloading python_wan dataset file from ${data_urls[idx]}"
gdown ${data_urls[idx]} -O ${FILE} --no-cookies
unzip ${FILE} -d ${DIR} # && rm ${FILE}

done

# rename dev to valid
mv ${DIR}dev ${DIR}valid

# raw file from
wget -P ${DIR} https://raw.githubusercontent.com/wanyao1992/code_summarization_public/master/dataset/original/data_ps.declbodies

