#!/bin/sh
cd ..

python craft_adv_samples.py --attack 'cw' --dataset 'mnist' --craft_num 100 --i '1'
python craft_adv_samples.py --attack 'cw' --dataset 'mnist' --craft_num 100 --i '2'
python craft_adv_samples.py --attack 'cw' --dataset 'mnist' --craft_num 100 --i '3'
python craft_adv_samples.py --attack 'cw' --dataset 'mnist' --craft_num 100 --i '4'

