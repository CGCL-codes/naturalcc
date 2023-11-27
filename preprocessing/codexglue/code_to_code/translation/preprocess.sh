#!/usr/bin/env bash

if [ -z $NCC ]; then
  CACHE_DIR=~
else
  CACHE_DIR=$NCC
fi

DATASET_NAME=codexglue/code_to_code/translation
echo "Preprocess CodeXGlue/Code-to-Code/translation dataset"
RAW_DIR=$CACHE_DIR/$DATASET_NAME/raw
mkdir -p $RAW_DIR

mv $CACHE_DIR/$DATASET_NAME/train.java-cs.txt.java $RAW_DIR/train.java
mv $CACHE_DIR/$DATASET_NAME/train.java-cs.txt.cs $RAW_DIR/train.csharp
mv $CACHE_DIR/$DATASET_NAME/valid.java-cs.txt.java $RAW_DIR/valid.java
mv $CACHE_DIR/$DATASET_NAME/valid.java-cs.txt.cs $RAW_DIR/valid.csharp
mv $CACHE_DIR/$DATASET_NAME/test.java-cs.txt.java $RAW_DIR/test.java
mv $CACHE_DIR/$DATASET_NAME/test.java-cs.txt.cs $RAW_DIR/test.csharp

mv *.csharp *.java $RAW_DIR
