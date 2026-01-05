#!/bin/bash

mkdir results

top_level_threads=1

scratch_data_dir="/scratch/pim_tree_data"
scratch_ops_dir="${scratch_data_dir}/ops"
init_file_str="${scratch_data_dir}/kv_init_key=[kv_init]_keylength=500000000_P=64_[INIT]_batch=1000000_alpha=0.0_size=500000000_0.0_0.0_0.0_0.0_0.0_0.0_0.0"

micro_prefix_str="micro_key=[kv_init]"
ycsb_prefix_str="ycsb_key=[kv_init]"
batch_prefix_str="batch_key=[kv_init]"

key_length_str="keylength=500000000"
p_str="P=2048"
init_str="[----]"
test_size_str="size=100000000"

wiki_init_file_str="${scratch_data_dir}/wiki_1000M_init.binary"
# wiki datasets
for repeat in "1" "2" "3"
do
    for sdk_type in "upmem" "direct"
    do
        for type in "insert"
        do
            wiki_ops_file_str="${scratch_data_dir}/ops/wiki_200M_${type}.binary"
            wiki_result_str="results/${sdk_type}_${repeat}/result_wiki_${top_level_threads}_${type}.txt"
            

            if [ "$sdk_type" = "upmem" ]; then
                exec_str="pim_tree_host_upmem"
            else
                exec_str="pim_tree_host"
            fi

            ls ${wiki_init_file_str}
            ls ${wiki_ops_file_str}
            ls ${wiki_result_str}
            ls ./build/${exec_str}

            echo "numactl --interleave=all ./build/${exec_str} -f ${wiki_init_file_str} ${wiki_ops_file_str} -l 1000000000 200000000 --top_level_threads ${top_level_threads} -c -t -d --test_batch_size 2000000 > ${wiki_result_str}"
            numactl --interleave=all ./build/${exec_str} -f ${wiki_init_file_str} ${wiki_ops_file_str} -l 1000000000 200000000 --top_level_threads ${top_level_threads} -c -t -d --test_batch_size 2000000 > ${wiki_result_str}
        done
    done
done