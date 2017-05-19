#!/bin/sh

wget https://skymindacademy.blob.core.windows.net/physionet2012/physionet2012.tar.gz
tar xzvf physionet2012.tar.gz
mv -i physionet2012 src/main/resources/physionet2012
