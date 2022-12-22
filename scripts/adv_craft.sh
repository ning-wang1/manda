#!/bin/sh
cd ..

python craft_adv_samples.py --attack 'fgsm' --change_threshold 0.05 --dataset nsl_kdd

