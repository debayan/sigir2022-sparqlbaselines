
# SIGIR2022 Short Paper : Modern Baselines for SPARQL Semantic Parsing

### Create virtualenv
```
git clone https://github.com/debayan/sigir2022-sparqlbaselines.git
cd sigir2022-sparqlbaselines/
python3.8 -m venv .
source bin/activate
pip3 install -r requirements.txt
```

## Reproducing the Results from the paper:

**NOTE**: You need a working copy of Wikidata and DBPedia KGs to replicate the experiment results. Specify the URL of DBPedia KG endpoint in pgn-bert/lcq1/Pointer-Generator-Networks/kbstats3253_corrctor.py and Wikidata endpoint in pgn-bert/lcq2/Pointer-Generator-Networks/kbstats3253_corrctor.py  in the hitkg() function.

### PGN-BERT LC-QuAD 1.0
Download pre-trained models from https://ltdata1.informatik.uni-hamburg.de/debayansigir2022-sparqlbaselines/pgn-bert-lcq1-models.tgz. Download input vectors from https://ltdata1.informatik.uni-hamburg.de/debayansigir2022-sparqlbaselines/bertlcq1.tgz. Unpack both inside pgn-bert/lcq1/Pointer-Generator-Networks. To run inference and evaluation on the 5 folds:

```
cd pgn-bert/lcq1/Pointer-Generator-Networks
CUDA_VISIBLE_DEVICES=0 python eval.py 1 bert_classify_results_1/checkpoint-19000/
CUDA_VISIBLE_DEVICES=0 python eval.py 2 bert_classify_results_2/checkpoint-13000/
CUDA_VISIBLE_DEVICES=0 python eval.py 3 bert_classify_results_3/checkpoint-15000/
CUDA_VISIBLE_DEVICES=4 python eval.py 4 bert_classify_results_4/checkpoint-16000/
CUDA_VISIBLE_DEVICES=0 python eval.py 5 bert_classify_results_5/checkpoint-16500/
```
Logs for the inference and evaluation for the 5 folds are available at https://github.com/debayan/sigir2022-sparqlbaselines/tree/main/pgn-bert/lcq1/Pointer-Generator-Networks/logbertbert. 

### PGN-BERT LC-QuAD 2.0
Download pre-trained model and input vectors from https://ltdata1.informatik.uni-hamburg.de/debayansigir2022-sparqlbaselines/pgn-bert-lcq2-models.tgz and https://ltdata1.informatik.uni-hamburg.de/debayansigir2022-sparqlbaselines/bertlcq2.tar respectively. Unpack them into pgn-bert/lcq2/Pointer-Generator-Networks. To run inference and evaluation:

```
cd pgn-bert/lcq2/Pointer-Generator-Networks/
CUDA_VISIBLE_DEVICES=0 python eval.py shuf_saved_bert_lcq2/Fire-At-Feb-14-2022_00-37-46.pth bert_bert_classify_results_1/checkpoint-5000/
```
Logs for the inference and evaluation are available at https://github.com/debayan/sigir2022-sparqlbaselines/blob/main/pgn-bert/lcq2/Pointer-Generator-Networks/logberteval1.tgz.

### T5-BART LC-QuAD 1.0
Download the pretrained models for LC-QuAD 1.0 from https://ltdata1.informatik.uni-hamburg.de/debayansigir2022-sparqlbaselines/checkpoints_SIGIR_PTLM.zip. Place the BART-base checkpoint in `ptlm/lcquad1/bart`, place the T5-base checkpoint in `ptlm/lcquad1/base`, and place the T5-small checkpoint in `ptlm/lcquad1/small`. Now run the following:
```
cd ptlm/lcquad1
bash eval.sh
```
To train the models run the following:
```
cd ptlm/lcquad1
bash train.sh
```

### T5-BART LC-QuAD 2.0
Download the pretrained models for LC-QuAD 2.0 from https://ltdata1.informatik.uni-hamburg.de/debayansigir2022-sparqlbaselines/checkpoints_SIGIR_PTLM.zip. Place the downloaded checkpoints in `ptlm/lcquad2` and run:
```
cd ptlm/lcquad2
bash eval.sh
```
To train the models run the following:
```
cd ptlm/lcquad1
bash train.sh
```

