#define __mram_ptr 

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdbool>
#include <iostream>

#include "task.hpp"
#include "task_framework_host.hpp"
#include "driver.hpp"

#ifndef DPU_BINARY
#ifdef DPU_ENERGY
#define DPU_BINARY "build/range_partitioning_skip_list_dpu_energy"
#else
#define DPU_BINARY "build/range_partitioning_skip_list_dpu"
#endif
#endif

using namespace std;

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

int main(int argc, char* argv[]) {
    driver::init();
    dpu_control::alloc(DPU_ALLOCATE_ALL);
    dpu_control::load(DPU_BINARY);
    init_dpus();
    driver::exec(argc, argv);

    dpu_control::free();
    return 0;
}