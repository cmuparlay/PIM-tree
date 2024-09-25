#!/bin/bash

init_folder="/scratch/pim_tree_data_2024_uint64"
init_file="${init_folder}/kv_init_key=[-home-upmem0037-zhaoyw-pim_tree_data_2024-kv_init]_keylength=500000000_P=2048_[INIT]_batch=1000000_alpha=0.0_size=500000000_0.0_0.0_0.0_0.0_0.0_0.0_0.0"

micro_folder="/scratch/pim_tree_data_2024_uint64/micro"
ops_file="${micro_folder}/_key=[-scratch-pim_tree_data-kv_init]_keylength=500000000_P=2048_[----]_batch=1000000_alpha=0.0_size=100000000_0.0_0.0_0.0_1.0_0.0_0.0_0.0"

ls ${init_file}
ls ${ops_file}

command="./build/pim_tree_host --file ${init_file} ${ops_file} -l 500000000 100000000"
echo ${command}

${command}