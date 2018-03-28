#!/usr/bin/env bash

python src/inference.py \
    --PolyRNN_metagraph='models/poly/polygonplusplus.ckpt.meta' \
    --PolyRNN_checkpoint='models/poly/polygonplusplus.ckpt' \
    --EvalNet_checkpoint='models/evalnet/evalnet.ckpt' \
    --InputFolder='imgs/' \
    --GGNN_checkpoint='models/ggnn/ggnn.ckpt' \
    --GGNN_metagraph='models/ggnn/ggnn.ckpt.meta' \
    --OutputFolder='output/' \
    --Use_ggnn=True

python src/vis_predictions.py \
    -pred_dir='output/' \
    --show_ggnn
