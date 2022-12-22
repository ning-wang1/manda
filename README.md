# manda
Code for infocom 2021 paper 'MANDA: On Adversarial Example Detection for Network Intrusion Detection System'

## Prepare the environment 
1. Implement annoconda following the instruction in https://www.anaconda.com/. Anaconda will help to manage the learning environement.
2. reate your env using the environment.yml file included in the repository. All the required libraries (including tensorflow and python) will be implemented. The default environment name is tf1_python3 as specified in the yml file. If you would like to use another name, edit the first line of the environment.yml file.

Create your env
> conda env create --file environment.yml

Activate your environment.
> conda activate tf1_python3
 

## Running the code
1. run 'craft_adv_sample.py' for crafting adversarial examples. There are several configurable params for selecting the specific attack.
2. run 'detect_ae_cv.py' for detecting the crafted AEs. There are also several configurable params.

Please cite

> @article{wang2022manda,
  title={Manda: On adversarial example detection for network intrusion detection system},
  author={Wang, Ning and Chen, Yimin and Xiao, Yang and Hu, Yang and Lou, Wenjing and Hou, Thomas},
  journal={IEEE Transactions on Dependable and Secure Computing},
  year={early access, 2022},
}

or

> @INPROCEEDINGS{9488874,
  author={Wang, Ning and Chen, Yimin and Hu, Yang and Lou, Wenjing and Hou, Y. Thomas},
  booktitle={IEEE INFOCOM 2021 - IEEE Conference on Computer Communications}, 
  title={MANDA: On Adversarial Example Detection for Network Intrusion Detection System}, 
  year={2021},
  volume={},
  number={},
  pages={1-10},
  doi={10.1109/INFOCOM42981.2021.9488874}}
