#!/bin/bash

for type in 3 5
do
    echo
    echo
    echo "Start Testing Type $type"
    for top_level_threads in 1
    do
        echo
        echo "Top Level Threads $top_level_threads"
        for zipf_factor in 0 1
        do
            echo "./build/fast_skip_list_host -c -t -d --top_level_threads ${top_level_threads} -f /scratch/pim_tree_data/init.insorted /scratch/pim_tree_data/test_100000000_${zipf_factor}_${type}.in > results9/result_noshadow_${top_level_threads}_${zipf_factor}_${type}.txt"
            # ./build/fast_skip_list_host -c -t -d --top_level_threads ${top_level_threads} -f /scratch/pim_tree_data/init.insorted /scratch/pim_tree_data/test_100000000_${zipf_factor}_${type}.in > results9/result_${top_level_threads}_${zipf_factor}_${type}.txt
            # echo "$zipf_factor"
        done
    done
done