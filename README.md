# manda
code for infocom 2021 paper MANDA

Prepare the environment 
Implement annoconda following the instruction in https://www.anaconda.com/. Anaconda will help to manage the learning environement.
Create your env using the environment.yml file included in the repository. All the required libraries (including pytorch and python) will be implemented. The default environment name is myenv as specified in the yml file. If you would like to use another name, edit the first line of the environment.yml file.
 conda env create --file environment.yml
activate your environment.
 conda activate envname
 

Running the code
1. run 'craft_adv_sample.py' for crafting adversarial examples. There are several configurable params for selecting the specific attack.
2. run 'detect_ae_cv.py' for detecting the crafted AEs. There are also several configurable params.
