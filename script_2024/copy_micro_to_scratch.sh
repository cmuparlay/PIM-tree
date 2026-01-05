#!/bin/bash

data_dir="/home/upmem0037/zhaoyw/pim_tree_data_2024"
scratch_data_dir="/scratch/pim_tree_data"
scratch_ops_dir="${scratch_data_dir}/ops"

init_file_name="kv_init_key=[kv_init]_keylength=500000000_P=64_[INIT]_batch=1000000_alpha=0.0_size=500000000_0.0_0.0_0.0_0.0_0.0_0.0_0.0"

mkdir ${scratch_data_dir}
mkdir ${scratch_ops_dir}

cp ${data_dir}/${init_file_name} ${scratch_data_dir}
cp -a ${data_dir}/micro/* ${scratch_ops_dir}
cp -a ${data_dir}/ycsb/* ${scratch_ops_dir}


# wiki data
cp ${data_dir}/wiki/wiki_1000M_init.binary ${scratch_data_dir}
cp ${data_dir}/wiki/wiki_200M_insert.binary ${scratch_ops_dir}
cp ${data_dir}/wiki/wiki_200M_predecessor.binary ${scratch_ops_dir}
