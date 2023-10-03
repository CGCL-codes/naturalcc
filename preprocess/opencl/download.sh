#!/usr/bin/env bash

if [ -z $NCC ]; then
  CACHE_DIR=~
else
  CACHE_DIR=$NCC
fi

data_names=opencl
data_urls="https://drive.google.com/uc?id=1ZZ8cpBoeD9PQsw7M-mWMJOVSyhaT0ygc"

echo "Downloading opencl dataset"
DIR=$CACHE_DIR
mkdir -p ${DIR}

FILE=${DIR}/${data_names}.tar.gz
gdown ${data_urls} -O ${FILE} --no-cookies
tar -zxvf ${FILE} -C ${DIR} && rm ${FILE}

# download XFG (conteXtual Flow Graph) for inst2vec model
DAW_DIR=$CACHE_DIR/$data_names/raw
XFG_FILE=$DAW_DIR/devmap_data.zip
wget https://polybox.ethz.ch/index.php/s/U08Z3xLhvbLk8io/download -O ${XFG_FILE}
unzip $XFG_FILE -d $DAW_DIR
rm -fr $DAW_DIR/__MACOSX
rm $DAW_DIR/cgo17-amd.csv $DAW_DIR/cgo17-nvidia.csv

# download inst2vec embedding
INST2VEC_EMBEDDING=$DAW_DIR/inst2vec.pkl
wget https://raw.githubusercontent.com/spcl/ncc/master/published_results/emb.p -O $INST2VEC_EMBEDDING

# download inst2vec vocabulary
INST2VEC_VOCABULARY=$DAW_DIR/vocabulary.zip
wget https://polybox.ethz.ch/index.php/s/AWKd60qR63yViH8/download -O $INST2VEC_VOCABULARY
unzip $INST2VEC_VOCABULARY -d $DAW_DIR
rm -fr $DAW_DIR/__MACOSX