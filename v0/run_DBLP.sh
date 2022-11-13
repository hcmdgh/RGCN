#!/bin/bash

set -eux

if [ ! -f main.py ]; then 
    cd v0 
fi 

python main.py \
    --hg_path "/Dataset/PyG/DBLP/Processed/DBLP.dglhg.pkl" \
    --infer_ntype author \
    --hidden_dim 256 \
    --num_layers 4 \
    --activation prelu \
    --num_epochs 200 \
    --lr 0.001 \
    --weight_decay 0.
