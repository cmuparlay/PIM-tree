#pragma once

#include <stdlib.h>
#include <mutex.h>
#include "common.h"
#include "node_dpu.h"

#define GC_ACTIVE

typedef struct gcnode {
    pptr size_addr;
    int64_t content[];
} gcnode;

typedef __mram_ptr struct gcnode* mgcptr;

gcnode free_list_l3node[MAX_L3_HEIGHT]; // for l3node

void gc_init() {
    #ifdef GC_ACTIVE
    for (int i = 0; i < MAX_L3_HEIGHT; i ++) {
        free_list_l3node[i].size_addr = PPTR(0, INVALID_DPU_ADDR);
    }
    #endif
}

void free_node(mL3ptr nn, int size) {
    #ifdef GC_ACTIVE
    IN_DPU_ASSERT(size < MAX_L3_HEIGHT, "free node invalid size");
    mgcptr gcnn = (mgcptr)nn;
    pptr addr = free_list_l3node[size].size_addr;
    gcnn->size_addr = addr;
    free_list_l3node[size].size_addr = PPTR(addr.id + 1, gcnn);
    #endif
}


pptr alloc_node(int height, uint32_t n) { // return (num_node_get, gcnodeaddr)
    #ifndef GC_ACTIVE
    return PPTR(0, INVALID_DPU_ADDR);
    #else
    IN_DPU_ASSERT(height < MAX_L3_HEIGHT, "alloc node invalid size");
    IN_DPU_ASSERT(n > 0, "alloc node invalid n");
    pptr addr = free_list_l3node[height].size_addr;
    if (addr.id >= n) { // have more nodes than requied
        mgcptr nxt = (mgcptr)free_list_l3node[height].size_addr.addr;
        pptr ret = PPTR(n, nxt);
        for (uint32_t i = 0; i < n; i ++) {
            gcnode nxtnode = *nxt;
            IN_DPU_ASSERT((uint32_t)nxt != INVALID_DPU_ADDR, "alloc invalid nxt");
            nxt = (mgcptr)nxtnode.size_addr.addr;
        }
        free_list_l3node[height].size_addr = PPTR(addr.id - n, nxt);
        return ret;
    } else { // run out of recycled nodes
        pptr ret = free_list_l3node[height].size_addr;
        free_list_l3node[height].size_addr = PPTR(0, INVALID_DPU_ADDR);
        return ret;
    }
    #endif
}