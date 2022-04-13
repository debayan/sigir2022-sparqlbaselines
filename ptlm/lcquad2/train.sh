# Training T5-base, T5-small and BART-base

python3 Train_T5.py --train_file train_new_mix.pickle --model_name t5-base
python3 Train_T5.py --train_file train_new_mix.pickle --model_name t5-small
python3 Train_BART.py --train_file train_new_mix.pickle --model_name facebook/bart-base

