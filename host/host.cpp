/*
 * Copyright (c) 2014-2019 - UPMEM
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @file host.c
 * @brief Template for a Host Application Source File.
 */

#define __mram_ptr 

bool print_debug = false;

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdbool>
#include <iostream>
#include <set>
#include <parlay/parallel.h>
#include <atomic>

#include "task_framework_host.hpp"
#include "task.hpp"
#include "driver.hpp"

#ifndef DPU_BINARY
#define DPU_BINARY "build/jumppush_pushpull_skip_list_dpu"
#endif

// #define ANSI_COLOR_RED "\x1b[31m"
// #define ANSI_COLOR_GREEN "\x1b[32m"
// #define ANSI_COLOR_RESET "\x1b[0m"
using namespace std;

// dpu_set_t dpu_set;
// struct dpu_set_t dpu;
// uint32_t each_dpu;
// int nr_of_dpus;
// int64_t epoch_number;

void init_dpus() {
    printf("\n********** INIT DPUS **********\n");
    auto io = alloc_io_manager();
    ASSERT(io == io_managers[0]);
    io->init();
    IO_Task_Batch* batch = io->alloc<dpu_init_task, empty_task_reply>(direct);

    parlay::parallel_for(0, nr_of_dpus, [&](size_t i) {
        auto it = (dpu_init_task*)batch->push_task_zero_copy(i, -1, false);
        *it = (dpu_init_task){.dpu_id = (int64_t)i};
    });
    io->finish_task_batch();
    ASSERT(!io->exec());
}

/**
 * @brief Main of the Host Application.
 */
int main(int argc, char* argv[]) {
    timer::default_detail = false;
    driver::init();
    dpu_control::alloc(DPU_ALLOCATE_ALL);
    dpu_control::load(DPU_BINARY);
    init_dpus();
    // dpu_control::print_all_log();
    driver::exec(argc, argv);

    dpu_control::free();
    return 0;
}