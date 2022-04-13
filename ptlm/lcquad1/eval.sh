# Test T5-base

python3 Test_T5.py --split_file mix/split_mix1.pickle --model_name t5-base --save_dir base --beam_length 10 --checkpoint split_mix1_checkpoint11000.pth
python3 Test_T5.py --split_file mix/split_mix2.pickle --model_name t5-base --save_dir base --beam_length 10 --checkpoint split_mix2_checkpoint12000.pth
python3 Test_T5.py --split_file mix/split_mix3.pickle --model_name t5-base --save_dir base --beam_length 10 --checkpoint split_mix3_checkpoint12000.pth
python3 Test_T5.py --split_file mix/split_mix4.pickle --model_name t5-base --save_dir base --beam_length 10 --checkpoint split_mix4_checkpoint11000.pth
python3 Test_T5.py --split_file mix/split_mix5.pickle --model_name t5-base --save_dir base --beam_length 10 --checkpoint split_mix5_checkpoint11000.pth

python3 temp_F1.py --save_dir base > base/test_output_base.txt

# Test T5-small

python3 Test_T5.py --split_file mix/split_mix1.pickle --model_name t5-small --save_dir small --beam_length 10 --checkpoint split_mix1_checkpoint12000.pth
python3 Test_T5.py --split_file mix/split_mix2.pickle --model_name t5-small --save_dir small --beam_length 10 --checkpoint split_mix2_checkpoint12000.pth
python3 Test_T5.py --split_file mix/split_mix3.pickle --model_name t5-small --save_dir small --beam_length 10 --checkpoint split_mix3_checkpoint9000.pth
python3 Test_T5.py --split_file mix/split_mix4.pickle --model_name t5-small --save_dir small --beam_length 10 --checkpoint split_mix4_checkpoint11000.pth
python3 Test_T5.py --split_file mix/split_mix5.pickle --model_name t5-small --save_dir small --beam_length 10 --checkpoint split_mix5_checkpoint12000.pth

python3 temp_F1.py --save_dir small > small/test_output_small.txt

# Test BART-base

python3 Test_BART.py --split_file mix/split_mix1.pickle --model_name facebook/bart-base --save_dir bart --beam_length 10 --checkpoint split_mix1_checkpoint20000.pth
python3 Test_BART.py --split_file mix/split_mix2.pickle --model_name facebook/bart-base --save_dir bart --beam_length 10 --checkpoint split_mix2_checkpoint30000.pth
python3 Test_BART.py --split_file mix/split_mix3.pickle --model_name facebook/bart-base --save_dir bart --beam_length 10 --checkpoint split_mix3_checkpoint30000.pth
python3 Test_BART.py --split_file mix/split_mix4.pickle --model_name facebook/bart-base --save_dir bart --beam_length 10 --checkpoint split_mix4_checkpoint30000.pth
python3 Test_BART.py --split_file mix/split_mix5.pickle --model_name facebook/bart-base --save_dir bart --beam_length 10 --checkpoint split_mix5_checkpoint30000.pth

python3 temp_F1.py --save_dir bart > bart/test_output_bart.txt

