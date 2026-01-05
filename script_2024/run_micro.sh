#!/bin/bash


#for type in 1 3 4 5 6


    # echo
    # echo "Start Testing Type $type"
    # for top_level_threads in 2
    # do
    #     echo
    #     echo "Top Level Threads $top_level_threads"
prefix_str="_key=[-scratch-pim_tree_data-kv_init]"
key_length_str="keylength=500000000"
p_str="P=2048"
init_str="[----]"
for batch in 1000000
do
    batch_str="batch=${batch}"
    for alpha in 0.0 0.2 0.4 0.6 0.8 1 1.2
    do
        alpha_str="alpha=${alpha}"
        for size in 100000000
        do
            size_str="size=${size}"
            for type in 1 3 4 5 6
            do
                arr=(0.0 0.0 0.0 0.0 0.0 0.0 0.0)
                arr[type]=1.0
                possibility_str=$(IFS=_; echo "${arr[*]}")
            
                op_file_name="'${prefix_str}_${key_length_str}_${p_str}_${init_str}_${batch_str}_${alpha_str}_${size_str}_${possibility_str}'"
                echo ${op_file_name}
                # echo "./build/fast_skip_list_host -c -t -d --top_level_threads ${top_level_threads} -f /scratch/pim_tree_data/init.insorted /scratch/pim_tree_data/test_100000000_${zipf_factor}_${type}.in > results9/result_${top_level_threads}_${zipf_factor}_${type}.txt"
                # ./build/fast_skip_list_host -c -t -d --top_level_threads ${top_level_threads} -f /scratch/pim_tree_data/init.insorted /scratch/pim_tree_data/test_100000000_${zipf_factor}_${type}.in > results9/result_${top_level_threads}_${zipf_factor}_${type}.txt
                # echo "$zipf_factor"
            done
        done
    done
    # done
done