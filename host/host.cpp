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

#include "task.hpp"
#include "task_framework_host.hpp"
#include "operation.hpp"
#include "compile.hpp"
#include "driver.hpp"

using namespace std;

void init_dpus() {
    printf("\n********** INIT DPUS **********\n");
    auto io = alloc_io_manager();
    ASSERT(io == io_managers[0]);
    io->init();
    IO_Task_Batch* batch = io->alloc<dpu_init_task, dpu_init_reply>(direct);

    parlay::parallel_for(0, nr_of_dpus, [&](size_t i) {
        auto it = (dpu_init_task*)batch->push_task_zero_copy(i, -1, false);
        *it = (dpu_init_task){.dpu_id = (int64_t)i};
    });
    io->finish_task_batch();
    ASSERT(io->exec());

    dpu_init_reply *ir = (dpu_init_reply *)batch->ith(0, 0);
    pim_skip_list::dpu_memory_regions& dmr = pim_skip_list::dmr;
    dmr.bbuffer_start = ir->bbuffer_start;
    dmr.bbuffer_end = ir->bbuffer_end;
    dmr.pbuffer_start = ir->pbuffer_start;
    dmr.pbuffer_end = ir->pbuffer_end;
#ifdef KHB_CPU_DEBUG
    printf("bbuffer:\nstart=%x\nend=%x\n\n", dmr.bbuffer_start,
           dmr.bbuffer_end);
    printf("pbuffer:\nstart=%x\nend=%x\n\n", dmr.pbuffer_start,
           dmr.pbuffer_end);
    for (int id = 1; id < nr_of_dpus; id++) {
        dpu_init_reply *ir = (dpu_init_reply *)batch->ith(id, 0);
        ASSERT(dmr.bbuffer_start == ir->bbuffer_start);
        ASSERT(dmr.bbuffer_end == ir->bbuffer_end);
        ASSERT(dmr.pbuffer_start == ir->pbuffer_start);
        ASSERT(dmr.pbuffer_end == ir->pbuffer_end);
    }
#endif
    io->reset();
}

/**
 * @brief Main of the Host Application.
 */
int main(int argc, char *argv[]) {
    driver::init();
    dpu_control::alloc(DPU_ALLOCATE_ALL);
    {
        cpu_coverage_timer->start();
        dpu_binary_switch_to(dpu_binary::init_binary);
        init_wram_save_pos();
        init_dpus();
        cpu_coverage_timer->end();
        cpu_coverage_timer->reset();
        pim_coverage_timer->reset();
    }
    
    driver::exec(argc, argv);
    // l3counters.print();
    // l2counters.print();
    // l1counters.print();
    // datacounters.print();
    // datacounters1.print();
    // datacounters2.print();
    // datacounters3.print();
    dpu_control::free();
    return 0;
}