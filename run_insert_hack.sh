#!/bin/bash

for i in $(seq 1 20)
do
    echo $i
    ./build/fast_skip_list_host -c -t -d --top_level_threads 2 -f /scratch/pim_tree_data/init.in /scratch/pim_tree_data/test_100000000_1_5.in > /scratch/result_debug.txt
done