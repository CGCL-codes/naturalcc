#!/usr/bin/env bash

data_names=(
  "javascript_dedupe_definitions_nonoverlap_v2_train.jsonl.gz"
  "javascript_augmented.pickle.gz"
)
data_urls=(
  "https://drive.google.com/uc?id=1YfHvacsAn9ngfjiJYbdo8LiFUqkbroxj" # 841M
  "https://drive.google.com/uc?id=1YfPTPPOv4evldpN-n_4QBDWDWFImv7xO" # 3.4G
)

#DIR=~/.ncc/augmented_javascript/raw/
DIR=/export/share/jianguo/scodebert/augmented_javascript/raw/
mkdir -p ${DIR}

for (( idx = 0 ; idx < ${#data_urls[@]} ; idx++ )); do

FILE=${DIR}${data_names[idx]}
echo "Downloading augmented_javascript dataset file from ${data_urls[idx]}"
gdown ${data_urls[idx]} -O ${FILE} --no-cookies
gunzip ${FILE} -d ${DIR} # && rm ${FILE}

done


# type inference data
data_urls=(
  "https://drive.google.com/uc?id=1YtLVoMUsxU6HTpu5Qvs0xldm_SC_FZRz" #82K
  "https://drive.google.com/uc?id=1YvoM6rxcaX1wsyQu0HbGurQdaV6fsmym" #7.0M
  "https://drive.google.com/uc?id=1YsoSKGhuOUw3CNAAzZ3j9Fm5C8itc8Jt" #6.0M
  "https://drive.google.com/uc?id=1YunIabuWqd3V9kZssloXrOUvyfLcM7FH" #53M
  "https://drive.google.com/uc?id=1YvN5UQuijRgUAL3aF1MzGLn1NiVMTAEo" #3.6M
)

#DIR=~/.ncc/augmented_javascript/type_prediction/raw/
DIR=/export/share/jianguo/scodebert/augmented_javascript/type_prediction/raw/
mkdir -p ${DIR}

for (( idx = 0 ; idx < ${#data_urls[@]} ; idx++ )); do

echo "Downloading augmented_javascript dataset file from ${data_urls[idx]}"
gdown ${data_urls[idx]} -O ${DIR} --no-cookies

done

