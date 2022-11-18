#!/bin/bash


#for type in 1 3 4 5 6
for type in "workloada" "workloadb" "workloade"
do
    echo
    echo
    echo "Start Testing Type $type"
    #for zipf_factor in 0 0.2 0.4 0.6 0.8
    for zipf_factor in 0 1
    do
        echo "./build/fast_skip_list_host -c -t -d -f /scratch/pim_tree_data/init.in /scratch/ycsb/test_100000000_${type}_${zipf_factor}.in > results9/result_${zipf_factor}_${type}.txt"
        # ./build/fast_skip_list_host -c -t -d -f /scratch/pim_tree_data/init.in /scratch/ycsb/test_100000000_${type}_${zipf_factor}.in > results9/result_${zipf_factor}_${type}.txt
        # echo "$zipf_factor"
    done
done