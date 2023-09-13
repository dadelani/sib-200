#!/bin/bash

mkdir -p data
mkdir -p data/raw

# download flores200
wget --trust-server-names https://tinyurl.com/flores200dataset

# unzip downloaded files.
tar -xvzf flores200_dataset.tar.gz

# move flores200 dev and devtest to data/raw
for split in dev devtest
do
  dir_name=flores200_dataset/$split
    for filename in "$dir_name"/*
    do
      cp -r $filename data/raw/
      done
  done
cp -r flores200_dataset/dev/deu_Latn.dev data/raw/
cp -r flores200_dataset/devtest/deu_Latn.devtest data/raw/


# delete archive and unused folders
rm -rf flores*

python create_sib_data.py

rm -rf data/raw
