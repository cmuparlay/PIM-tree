# PIM-tree

PIM-tree is a theoretically and practically efficient comparison-based ordered index for Processing-In-Memory (PIM). It achieves high performance against data skew by providing a sweet spot for both data locality and load-balance of PIM modules. This repository contains the implementation of PIM-tree index used in our paper.

If you use PIM-tree, please cite our paper:

[1] **PIM-tree: A Skew-resistant Index for Processing-in-Memory**. Hongbo Kang, Yiwei Zhao, Guy E. Blelloch, Laxman Dhulipala, Yan Gu, Charles McGuffey, and Phillip B. Gibbons. Proceedings of the VLDB Endowment (PVLDB), 16(4): 946-958, 2022. *doi:10.14778/3574245.3574275*. *arxiv Preprint: 2211.10516*. [[Paper](https://dl.acm.org/doi/10.14778/3574245.3574275)][[arXiv](https://arxiv.org/abs/2211.10516)].

[2] **The Processing-in-Memory Model.** Hongbo Kang, Phillip B Gibbons, Guy E Blelloch, Laxman Dhulipala, Yan Gu, Charles McGuffey. 2021. In Proceedings of the 33rd ACM Symposium on Parallelism in Algorithms and Architectures. 295–306. [[doi](https://dl.acm.org/doi/10.1145/3409964.3461816)].

### Related Repositories:
1. The codes for range-partitioning PIM-based indexes serving as PIM-tree's competitors can be found in this repo at branch `range_partition`.
2. The codes for baseline "Jump-Push Search" and "Push-Pull Search" used in the study of the impact of different optimizations can be found in this repo at branch `jumppush_pushpull`.
3. Implementation of shared-memory competitors is referred to [SetBench](https://bitbucket.org/trbot86/setbench/src/master/).

## Requirements

This implementation was created to run the experiments in the paper. Current implementation of PIM-tree can only run on [UPMEM](https://www.upmem.com/) machines. This codeset is built on [UPMEM SDK](https://sdk.upmem.com/).

## Building

To build everything, enter the root of the cloned repository. You will need to change `NR_DPUS` in `Makefile` to the number of DPU modules on your machine before you start building. Then run your desired command listed below.

| Command | Description |
|---------|-------------|
|make | Compiles and links|
|make test | Compiles and start a test case|
|make energy | Compiles a version used for energy data collection|
|make debug | Compiles a version for debugging|
|make clean | Cleans the previous compiled files|

The build produces files in `build` directory, which can be classfied into two types:
- Host program(`pim_tree_host`): The host application used to drive the system.
- DPU program(`pim_tree_dpu`): Application run on the DPUs.

DPU applications ending with query types (`insert`, `delete`, `scan`, `predecessor`, `get_update`, `build`, `init`) are produced due to limited space of DPUs' instruction memories. A single DPU application handling all the query requests are too large to fit in the IRAM, and thus has to split.

Host and DPU applications ending with `no_shadow_subtree` serve as a competitor of Jump-Push Baseline introduced in the paper.

Host and DPU applications ending with `energy` are used to collect energy consumption data.

## Running

The basic structure for a running command is:

```
./build/pim_tree_host [--arguments]
```

Please refer to the following list to set the arguments:

| Argument | Abbreviation | Used Scenario* | Usage Description |
|---------|-------------|-------------|-------------|
| `--file` | `-f` | F |Use `--file [init_file] [test_file]` to include the path of dataset files|
| `--length` | `-l` | TG |Use `-l [init_length] [test_length]` to set the number of initializing and testing queries|
| `--init_batch_size` | NA | TG |Use `--init_batch_size [init batch size]` to set the batch size of initializing queries|
| `--test_batch_size` | NA | TG |Use `--test_batch_size [test batch size]` to set the batch size of testing queries|
| `--output_batch_size` | NA | G |Use `--output_batch_size [batch size for output file]` to set the batch size of the output file|
| `--get` | `-g` | TG |Use `-g [get_ratio]` to set the ratio of Get queries|
| `--predecessor` | `-p` | TG |Use `-p [predecessor_ratio]` to set the ratio of Predecessor queries|
| `--insert` | `-i` | TG |Use `-i [insert_ratio]` to set the ratio of Insert queries|
| `--remove` | `-r` | TG |Use `-r [remove_ratio]` to set the ratio of Delete queries|
| `--scan` | `-s` | TG |Use `-s [scan_ratio]` to set the ratio of Scan queries|
| `--alpha` | NA | TG |Use `--alpha [?x]` to set the skew of testing queries|
| `--nocheck` | `-c` | A |Stop checking the correctness of the tested data structure|
| `--noprint` | `-t` | A |Do not print timer name when timing|
| `--nodetail` | `-d` | A |Do not print detail|
| `--top_level_threads` | NA | FT |Use `--top_level_threads [#threads]` to set the number of threads for CPU side pipelining|
| `--push_pull_limit_dynamic` | NA | FT |Use `--push_pull_limit_dynamic [limit]` to set the push-pull limit|
| `--output` | `-o` | G |Use `--output [init_file] [test_file]` to set paths of output datasets|
| `--generate_all_test_cases` | NA | G |Generate the output dataset with function `generate_all_test()`. Ignore other arguments.|
| `--init_state` | NA | FG |Enables sequential writing of initializing dataset|

Please refer to the following list for used scenarios*:

| Used Scenario Abbreviation | Scenario Description |
|---------|-------------|
| F | Testing with existing dataset files |
| T | Directly testing with self-set Zipfian workloads |
| G | Generating new dataset files with Zipfian workloads |
| A | Any time |

For detailed information about workload generation and test execution, check function `generate_all_test` and `exec` in `./pim_base/include/host/driver.hpp` for more detail.

## Code Structure

This section will introduce the contents in each code file and how the codes are organized.

### ```/common```

This directory contains the configurations of the hardware used and the data structure.

```common.h```: Numeric configurations of the data structure.

```settings.h```: Configuration for program modes. (whether to turn on (1) debugging (2) tracking of various metrics)

```task_base.h```: Macros defining the interface of DPU function calls (tasks sent between CPU and DPUs).

### ```/dpu```

This directory contains source codes used to build the DPU applications.

```bnode.h```: Functions related to "distributed chunked skip list nodes" in layer 2 and 1.

```cache.h```: Definitions and functions related to "cache_init_record", a request generated when building
shadow subtrees for new layer 2 nodes.

```data_block.h```: Definitions and functions related to "data_block", a variable length vector on MRAM, used as the
storage unit for Bnodes.

```dpu_buffer.h```: Auxiliary function for procession of variable length data on DPUs.

```dpu.c```: Main function for the DPU applications. Handling function calls (tasks) from the CPU host application.

```gc.h```: DPU garbage collection.

```hashtable_l3size.h```: Definition and implementation of the local linear-probing hash table on each DPU, used in GET, INSERT, and DELETE operations.

```l3_ab_tree.h```: Implementation of replicated upper part (L3 in our paper) with ab-tree.

```l3_skip_list.h```: Implementation of replicated upper part (L3 in our paper) with skip list. (abandoned)

```node_dpu.h```: Defining node classes used in Layer 3, 2, 1, and data nodes.

```pnode.h```: Implementation of data nodes.

```statistics.h```: Stats collection.

```storage.h```: Maintaining consistency of WRAM heap.

### ```/host```

This directory contains source codes used to build the CPU host application.

```compile.hpp```: Handling the potential cases where DPU programs overflow the IRAM of the DPUs.

```host.cpp```: Main function for the CPU host application.

```operation.hpp```: Functions for database queries on the CPU host side.

```papi_counters.hpp```: Experiment stats collection.

### ```/pim_base/include```

This submodule contains efficient low-level function calls we implemented, including efficient CPU-DPU communication, pipelining, performance stats collection, database correctness testing, argument parsing, random distribution generators and other miscs.

We wrap this up into high-level function calls, so that readers could get a clarified overview of our data structure design without getting too involved in the details of UPMEM programming.
