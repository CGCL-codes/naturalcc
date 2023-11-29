# Check NCC
if [ -z $NCC ];
then
    echo 'The dataset environment variable $NCC is not set'
    exit 1
fi


HOME_DIR=$NCC/manytypes4py
mkdir -p $HOME_DIR

# Dataset Downloading

ZENODO_GET=""
if [ -z `which zenodo_get` ];
then
    if [ -z `which zenodo_get.py` ];
    then
           echo 'Please install zenodo-get by `pip install zenodo-get`'
           exit 1
    else
        ZENODO_GET=zenodo_get.py
    fi
else
    ZENODO_GET=zenodo_get
fi

cd $HOME_DIR && ${ZENODO_GET} 10.5281/zenodo.5244636

# Dataset Extraction
if [ ! -d $HOME_DIR/ManyTypes4PyDataset-v0.7 ];
then
    echo "Extracting the dataset"
    tar xzvf ManyTypes4PyDataset-v0.7.tar.gz
else
    echo "The ManyTypes4PyDataset is already extracted."
fi
