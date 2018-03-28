#!/usr/bin/env bash

export FILENAME=checkpoints_cityscapes.tar.gz
export URL=http://www.cs.toronto.edu/polyrnn/models/$FILENAME

wget $URL
tar -xvf $FILENAME ./models/
