#pragma once

#include <defs.h>
#include <mram.h>
#include <perfcounter.h>
#include <string.h>
#include "pptr.h"
#include "macro.h"

typedef struct L3node {
    int64_t key;
    int64_t height;
    int64_t value;
    mppptr left __attribute__((aligned (8)));
    mppptr right __attribute__((aligned (8)));
} L3node;

typedef __mram_ptr struct L3node* mL3ptr;

extern mL3ptr root;

#ifdef DPU_ENERGY
extern uint64_t op_count;
extern uint64_t db_size_count;
extern uint64_t cycle_count;
#endif
