# Train T5-base

mkdir base

echo "Training T5 base on the first split for LCQUAD 1.0"
python3 Train_T5.py --split_file mix/split_mix1.pickle --model_name t5-base --save_dir base

echo "Training T5 base on the second split for LCQUAD 1.0"
python3 Train_T5.py --split_file mix/split_mix2.pickle --model_name t5-base --save_dir base

echo "Training T5 base on the third split for LCQUAD 1.0"
python3 Train_T5.py --split_file mix/split_mix3.pickle --model_name t5-base --save_dir base

echo "Training T5 base on the fourth split for LCQUAD 1.0"
python3 Train_T5.py --split_file mix/split_mix4.pickle --model_name t5-base --save_dir base

echo "Training T5 base on the fifth split for LCQUAD 1.0"
python3 Train_T5.py --split_file mix/split_mix5.pickle --model_name t5-base --save_dir base

# Train T5-small

mkdir small

echo "Training T5 small on the first split for LCQUAD 1.0"
python3 Train_T5.py --split_file mix/split_mix1.pickle --model_name t5-small --save_dir small

echo "Training T5 small on the second split for LCQUAD 1.0"
python3 Train_T5.py --split_file mix/split_mix2.pickle --model_name t5-small --save_dir small

echo "Training T5 small on the third split for LCQUAD 1.0"
python3 Train_T5.py --split_file mix/split_mix3.pickle --model_name t5-small --save_dir small

echo "Training T5 small on the fourth split for LCQUAD 1.0"
python3 Train_T5.py --split_file mix/split_mix4.pickle --model_name t5-small --save_dir small

echo "Training T5 small on the fifth split for LCQUAD 1.0"
python3 Train_T5.py --split_file mix/split_mix5.pickle --model_name t5-small --save_dir small

# Train BART-base

mkdir bart

echo "Training BART base on the first split for LCQUAD 1.0"
python3 Train_BART.py --split_file mix/split_mix1.pickle --model_name facebook/bart-base --save_dir bart

echo "Training BART base on the second split for LCQUAD 1.0"
python3 Train_BART.py --split_file mix/split_mix2.pickle --model_name facebook/bart-base --save_dir bart

echo "Training BART base on the third split for LCQUAD 1.0"
python3 Train_BART.py --split_file mix/split_mix3.pickle --model_name facebook/bart-base --save_dir bart

echo "Training BART base on the fourth split for LCQUAD 1.0"
python3 Train_BART.py --split_file mix/split_mix4.pickle --model_name facebook/bart-base --save_dir bart

echo "Training BART base on the fifth split for LCQUAD 1.0"
python3 Train_BART.py --split_file mix/split_mix5.pickle --model_name facebook/bart-base --save_dir bart
