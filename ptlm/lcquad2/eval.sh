# Testing T5-base, T5-small and BART-base

python3 Test_T5.py --test_file test_new_mix.pickle --model_name t5-base --checkpoint T5_train_new_mix_checkpoint40000.pth --beam_length 10
python3 Test_T5.py --test_file test_new_mix.pickle --model_name t5-small --checkpoint small_train_new_mix_checkpoint56000.pth --beam_length 10
python3 Test_BART.py --test_file test_new_mix.pickle --model_name facebook/bart-base --checkpoint BART_train_new_mix_checkpoint40000.pth --beam_length 10

python3 get_F1.py --test_file T5_mix_test_result.json > T5_mix_test.txt
python3 get_F1.py --test_file small_mix_test_result.json > small_mix_test.txt
python3 get_F1.py --test_file BART_mix_test_result.json > BART_mix_test.txt

