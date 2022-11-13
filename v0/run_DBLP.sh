#!/bin/bash

# 【实验结果】
# best_val_f1_micro 0.94     best_val_f1_micro_epoch 7
# best_val_f1_macro 0.93948  best_val_f1_macro_epoch 13
# best_test_f1_micro 0.94443 best_test_f1_micro_epoch 12
# best_test_f1_macro 0.9394  best_test_f1_macro_epoch 12

set -eux

if [ ! -f main.py ]; then 
    cd v0 
fi 

python main.py \
    --hg_path "/Dataset/PyG/DBLP/Processed/DBLP.dglhg.pkl" \
    --infer_ntype author \
    --hidden_dim 400 \
    --num_layers 4 \
    --activation prelu \
    --num_epochs 200 \
    --lr 0.001 \
    --weight_decay 0.
