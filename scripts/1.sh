#!/bin/sh
cd ..

# Activate the relevant virtual environment:

# python main2.py --save_filepath 'result/uncertainty2.csv'
# python main2.py --save_filepath 'result/uncertainty3.csv'
# python main2.py --save_filepath 'result/uncertainty4.csv'
# python main2.py --save_filepath 'result/uncertainty5.csv'

python main2.py --save_filepath 'result/uncertainty1_0.csv' --original_lab 0 --target_lab 1
python main2.py --save_filepath 'result/uncertainty2_0.csv' --original_lab 0 --target_lab 1
python main2.py --save_filepath 'result/uncertainty3_0.csv' --original_lab 0 --target_lab 1
python main2.py --save_filepath 'result/uncertainty4_0.csv' --original_lab 0 --target_lab 1
python main2.py --save_filepath 'result/uncertainty5_0.csv' --original_lab 0 --target_lab 1

# python main2.py --save_filepath 'result/uncertainty6.csv'
# python main2.py --save_filepath 'result/uncertainty7.csv'
# python main2.py --save_filepath 'result/uncertainty8.csv'
# python main2.py --save_filepath 'result/uncertainty9.csv'
# python main2.py --save_filepath 'result/uncertainty10.csv'


# python main2.py --save_filepath 'result/uncertainty6_0.csv' --original_lab 0 --target_lab 1
# python main2.py --save_filepath 'result/uncertainty7_0.csv' --original_lab 0 --target_lab 1
# python main2.py --save_filepath 'result/uncertainty8_0.csv' --original_lab 0 --target_lab 1
# python main2.py --save_filepath 'result/uncertainty9_0.csv' --original_lab 0 --target_lab 1
# python main2.py --save_filepath 'result/uncertainty10_0.csv' --original_lab 0 --target_lab 1