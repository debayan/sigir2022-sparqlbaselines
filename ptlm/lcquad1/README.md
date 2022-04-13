# LCQUAD1-T5

To train (Split 1) on T5-base, device 1:

```
python3 Train_T5.py --model t5-base --device 1 --split_file split1.pickle
```

To test (Split 1) on T5-base, device 1:

```
python3 Train_T5.py --model t5-base --device 1 --split_file split1.pickle --test True --checkpoint split1_results/split1_checkpoint11000.pth
```

Enter the correct path to the checkpoint for testing in the ``--checkpoint`` argument.
