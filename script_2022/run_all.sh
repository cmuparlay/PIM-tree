#!/bin/bash


#for type in 1 3 4 5 6
for type in 1 3 5 6
do
    echo
    echo
    echo "Start Testing Type $type"
    for top_level_threads in 2
    do
        echo
        echo "Top Level Threads $top_level_threads"
        for zipf_factor in 0 0.2 0.4 0.6 0.8 1 1.2
        do
            echo "./build/fast_skip_list_host -c -t -d --top_level_threads ${top_level_threads} -f /scratch/pim_tree_data/init.insorted /scratch/pim_tree_data/test_100000000_${zipf_factor}_${type}.in > results9/result_${top_level_threads}_${zipf_factor}_${type}.txt"
            # ./build/fast_skip_list_host -c -t -d --top_level_threads ${top_level_threads} -f /scratch/pim_tree_data/init.insorted /scratch/pim_tree_data/test_100000000_${zipf_factor}_${type}.in > results9/result_${top_level_threads}_${zipf_factor}_${type}.txt
            # echo "$zipf_factor"
        done
    done
done