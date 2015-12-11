#! /bin/bash

# TODO: specify threshold here!
thr_1="15"
thr_2="15"
thr_3="20"
thr_4="3000"
thr_5="1"

python ./main.py 1 ${thr_1}
python ./main.py 2 ${thr_2}
python ./main.py 3 ${thr_3}
python ./main.py 4 ${thr_4}
python ./main.py 5 ${thr_5}
