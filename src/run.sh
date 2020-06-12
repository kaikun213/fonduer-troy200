#!/bin/bash

exec python3 main_troy200_fonduer.py --docs 200 --exp "norm" --clear_db 1 --cls_methods "lstm" >> results/troy200_norm_lstm.txt &
exec python3 main_troy200_fonduer.py --docs 200 --exp "pred" --clear_db 1 --cls_methods "lstm" >> results/troy200_pred_lstm.txt &
exec python3 main_troy200_fonduer.py --docs 200 --exp "gold" --clear_db 1 --cls_methods "lstm" >> results/troy200_gold_lstm.txt
