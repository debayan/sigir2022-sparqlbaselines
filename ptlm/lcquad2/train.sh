# Training T5-base, T5-small and BART-base

echo "Training T5 base on LCQUAD 2.0"
python3 Train_T5.py --train_file train_new_mix.pickle --model_name t5-base

echo "Training T5 small on LCQUAD 2.0"
python3 Train_T5.py --train_file train_new_mix.pickle --model_name t5-small

echo "Training BART base on LCQUAD 2.0"
python3 Train_BART.py --train_file train_new_mix.pickle --model_name facebook/bart-base

