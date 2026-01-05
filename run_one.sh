#!/bin/bash

generate_op_filename() {
    local file_prefix=${10}

    local kv_file=$1
    local kv_file_replace=$(echo "${kv_file}" | tr '/' '-')
    local kv_file_str="_key=[${kv_file_replace}]"

    local key_length=$2
    local key_length_str="keylength=${key_length}"

    local p=$3
    local p_str="P=${p}"

    local init=$4
    local init_str="[${init}]"

    local batch=$5
    local batch_str="batch=${batch}"

    local alpha=$6
    local alpha_str="alpha=${alpha}"

    local size=$7
    local size_str="size=${size}"

    local possibility_str=$8

    local base_folder=$9

    echo "${base_folder}/${file_prefix}${kv_file_str}_${key_length_str}_${p_str}_${init_str}_${batch_str}_${alpha_str}_${size_str}_${possibility_str}"
}

generate_possibility() {
    local possibility_str=""
    if [ $1 = "INIT" ]; then
        possibility_str="0.0_0.0_0.0_0.0_0.0_0.0_0.0"
    elif [ $1 = "GET" ]; then
        possibility_str="0.0_1.0_0.0_0.0_0.0_0.0_0.0"
    elif [ $1 = "PREDECESSOR" ]; then
        possibility_str="0.0_0.0_0.0_1.0_0.0_0.0_0.0"
    elif [ $1 = "INSERT" ]; then
        possibility_str="0.0_0.0_0.0_0.0_1.0_0.0_0.0"
    elif [ $1 = "REMOVE" ]; then
        possibility_str="0.0_0.0_0.0_0.0_0.0_1.0_0.0"
    elif [ $1 = "SCAN" ]; then
        possibility_str="0.0_0.0_0.0_0.0_0.0_0.0_1.0"
    elif [ $1 = "YCSB_A" ]; then
        possibility_str="0.0_0.0_0.0_0.5_0.5_0.0_0.0"
    elif [ $1 = "YCSB_B" ]; then
        possibility_str="0.0_0.0_0.0_0.95_0.05_0.0_0.0"
    elif [ $1 = "YCSB_E" ]; then
        possibility_str="0.0_0.0_0.0_0.0_0.05_0.0_0.05"
    else
        echo "Invalid type"
        exit 1
    fi
    echo ${possibility_str}
}

folder="/scratch/pim_tree_data"

kv_length="500000000"
kv_file="${folder}/kv_init_${kv_length}"
ls ${kv_file}

batch_size="1000000"
init_folder="${folder}"
init_possibility=$(generate_possibility "INIT")
init_file_prefix="kv_init"
echo "init_possibility=${init_possibility}"
echo "init_folder=${init_folder}"
echo "file_prefix=${init_file_prefix}"
init_file=$(generate_op_filename ${kv_file} ${kv_length} "2048" "INIT" ${batch_size} "0.0" ${kv_length} ${init_possibility} ${init_folder} ${init_file_prefix})

ops_length="100000000"
op_folder="${folder}/micro"
op_possibility=$(generate_possibility "PREDECESSOR")
op_file_prefix="micro"
echo "op_possibility=${op_possibility}"
echo "op_folder=${op_folder}"
echo "file_prefix=${op_file_prefix}"
op_file=$(generate_op_filename ${kv_file} ${kv_length} "2048" "----" ${batch_size} "0.0" ${ops_length} ${op_possibility} ${op_folder} ${op_file_prefix})

ls ${init_file}
echo "init_file=${init_file}"
ls ${op_file}
echo "op_file=${op_file}"

command="./build/pim_tree_host --file ${init_file} ${op_file} -l ${kv_length} ${ops_length}"
echo ${command}
# ${command}

# folder="/scratch/pim_tree_data_2024"
# kv_length="500000000"
# kv_file="${folder}/kv_init_${kv_length}"
# prefix_str="_key=[-scratch-pim_tree_data-kv_init]"
# key_length_str="keylength=500000000"
# p_str="P=2048"
# init_str="[----]"
# for batch in 1000000
# do
#     batch_str="batch=${batch}"
#     for alpha in 0.0 0.2 0.4 0.6 0.8 1 1.2
#     do
#         alpha_str="alpha=${alpha}"
#         for size in 100000000
#         do
#             size_str="size=${size}"
#             for type in 1 3 4 5 6
#             do
#                 arr=(0.0 0.0 0.0 0.0 0.0 0.0 0.0)
#                 arr[type]=1.0
#                 possibility_str=$(IFS=_; echo "${arr[*]}")
            
#                 op_file_name="'${prefix_str}_${key_length_str}_${p_str}_${init_str}_${batch_str}_${alpha_str}_${size_str}_${possibility_str}'"
#                 echo ${op_file_name}
#                 # echo "./build/fast_skip_list_host -c -t -d --top_level_threads ${top_level_threads} -f /scratch/pim_tree_data/init.insorted /scratch/pim_tree_data/test_100000000_${zipf_factor}_${type}.in > results9/result_${top_level_threads}_${zipf_factor}_${type}.txt"
#                 # ./build/fast_skip_list_host -c -t -d --top_level_threads ${top_level_threads} -f /scratch/pim_tree_data/init.insorted /scratch/pim_tree_data/test_100000000_${zipf_factor}_${type}.in > results9/result_${top_level_threads}_${zipf_factor}_${type}.txt
#                 # echo "$zipf_factor"
#             done
#         done
#     done
#     # done
# done