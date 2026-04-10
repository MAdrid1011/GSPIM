#!/bin/bash
# 写一个脚本，接受一个参数，然后运行benchmark_gspim.py
NAME=$1
python benchmark_gspim.py --model_path output/N3V/$NAME --num_frames 300 --source_path data/N3V/$NAME --window_size 5
