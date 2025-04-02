#!/usr/bin/env bash


data_names=(
  "java.zip"
)
data_urls=(
  "https://drive.google.com/uc?id=13o4MiELiQoomlly2TCTpbtGee_HdQZxl"
)

echo "Downloading java-hu dataset"
DIR=~/java_hu/raw/
mkdir -p ${DIR}

for (( idx = 0 ; idx < ${#data_urls[@]} ; idx++ )); do

FILE=${DIR}${data_names[idx]}
echo "Downloading augmented_javascript dataset file from ${data_urls[idx]}"
gdown ${data_urls[idx]} -O ${FILE} --no-cookies
unzip ${FILE} -d ${DIR} # && rm ${FILE}

done

# rename dev to valid
mv ${DIR}dev ${DIR}valid