#!/bin/bash

mkdir results

top_level_threads=2

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

micro_file_str="${scratch_ops_dir}/${micro_prefix_str}_${key_length_str}_${p_str}_${init_str}"
ycsb_file_str="${scratch_ops_dir}/${ycsb_prefix_str}_${key_length_str}_${p_str}_${init_str}"
batch_file_str="${scratch_ops_dir}/${batch_prefix_str}_${key_length_str}_${p_str}_${init_str}"

# micro benchmarks
for batch in 1000000
do
    batch_str="batch=${batch}"
    for alpha in 0.0 0.2 0.4 0.6 0.8 1.0 1.2
    do
        alpha_str="alpha=${alpha}"
        for type in 1 3 4 5 6
        do
            arr=(0.0 0.0 0.0 0.0 0.0 0.0 0.0)
            arr[type]=1.0
            possibility_str=$(IFS=_; echo "${arr[*]}")
            
            op_file_name="'${micro_file_str}_${batch_str}_${alpha_str}_${test_size_str}_${possibility_str}'"
            result_str="results/result_micro_${top_level_threads}_${zipf_factor}_${type}.txt"

            echo ${op_file_name}
            echo "numactl --interleave=all ./build/fast_skip_list_host -f ${init_file_str} ${op_file_name} --top_level_threads ${top_level_threads} -c -t -d > ${result_str}"
            # numactl --interleave=all ./build/fast_skip_list_host -f ${init_file_str} ${op_file_name} --top_level_threads ${top_level_threads} -c -t -d > ${result_str}
        done
    done
done

# # YCSB
# for batch in 1000000
# do
#     batch_str="batch=${batch}"
#     for alpha in 0.0 1.0
#     do
#         alpha_str="alpha=${alpha}"
#         for type in {1..3}
#         do
#             arr=(0.0 0.0 0.0 0.0 0.0 0.0 0.0)
#             if [ $type -eq 1 ]; then
#                 arr[3]=0.5
#                 arr[4]=0.5
#             elif [ $type -eq 2 ]; then
#                 arr[3]=0.9
#                 arr[4]=0.1
#             else
#                 arr[6]=0.9
#                 arr[4]=0.1
#             fi
#             possibility_str=$(IFS=_; echo "${arr[*]}")
            
#             op_file_name="'${ycsb_file_str}_${batch_str}_${alpha_str}_${test_size_str}_${possibility_str}'"
#             result_str="results/result_ycsb_${top_level_threads}_${zipf_factor}_${type}.txt"

#             echo ${op_file_name}
#             echo "numactl --interleave=all ./build/fast_skip_list_host -f ${init_file_str} ${op_file_name} --top_level_threads ${top_level_threads} -c -t -d > ${result_str}"
#             numactl --interleave=all ./build/fast_skip_list_host -f ${init_file_str} ${op_file_name} --top_level_threads ${top_level_threads} -c -t -d > ${result_str}
#         done
#     done
# done

# for batch in 1000000 500000 200000 100000 50000 20000
# do
#     batch_str="batch=${batch}"
#     for alpha in 0.0
#     do
#         alpha_str="alpha=${alpha}"
#         for type in 3
#         do
#             arr=(0.0 0.0 0.0 0.0 0.0 0.0 0.0)
#             arr[type]=1.0
#             possibility_str=$(IFS=_; echo "${arr[*]}")
            
#             op_file_name="'${batch_file_str}_${batch_str}_${alpha_str}_${test_size_str}_${possibility_str}'"
#             result_str="results/result_batch_${top_level_threads}_${batch}.txt"

#             echo ${op_file_name}
#             echo "numactl --interleave=all ./build/fast_skip_list_host -f ${init_file_str} ${op_file_name} --top_level_threads ${top_level_threads} -c -t -d > ${result_str}"
#             numactl --interleave=all ./build/fast_skip_list_host -f ${init_file_str} ${op_file_name} --top_level_threads ${top_level_threads} -c -t -d > ${result_str}
#         done
#     done
# done
