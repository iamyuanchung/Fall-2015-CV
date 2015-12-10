#! /bin/bash

# TODO: you can change threshold here!
thr_roberts="12"
thr_prewitt="24"
thr_sobel="38"
thr_frei="30"
thr_kirsch="135"
thr_robinson="43"
thr_nevatia="12500"

python ./main.py roberts ${thr_roberts}
python ./main.py prewitt ${thr_prewitt}
python ./main.py sobel ${thr_sobel}
python ./main.py frei ${thr_frei}
python ./main.py kirsch ${thr_kirsch}
python ./main.py robinson ${thr_robinson}
python ./main.py nevatia ${thr_nevatia}
