#pragma once
#include "common.h"
#include "node_dpu.h"

extern __mram_ptr uint8_t l3buffer[];
extern int l3cnt;

typedef struct garbage {
    uint64_t nxt; // 8B aligned
    int64_t height; // same place as height in nodes
} garbage;

__host __mram_ptr uint64_t L3_garbage[MAX_L3_HEIGHT];

void L3_gc_init() {
    memset(L3_garbage, -1, sizeof(L3_garbage));
}

void L3_gc(mL3ptr space, int height) {
    IN_DPU_ASSERT(height < MAX_L3_HEIGHT, "L3_garbage_collect: recycling a node too high");
    __mram_ptr garbage* g = (__mram_ptr garbage*)space;
    g->nxt = L3_garbage[height];
    g->height = -1;
    L3_garbage[height] = (uint64_t)g;
}

mL3ptr L3_allocate(int height) {
    int actual_size = sizeof(L3node) + sizeof(pptr) * height * 2;
    if (L3_garbage[height] == (uint64_t)-1) {
        mL3ptr addr = (mL3ptr)(l3buffer + l3cnt);
        l3cnt += actual_size;
        return addr;
    } else {
        IN_DPU_ASSERT(false, "wrong");
        __mram_ptr garbage* addr = (__mram_ptr garbage*)L3_garbage[height];
        L3_garbage[height] = addr->nxt;
        return (mL3ptr)addr;
    }
}