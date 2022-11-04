#pragma once

#include <stdlib.h>
#include <mutex.h>
#include "common.h"
#include "node_dpu.h"

#define GC_ACTIVE
// NOTE: all functions need lock
// Executed in serial

typedef struct gcnode {
    pptr size_addr;
    int64_t content[];
} gcnode;

typedef __mram_ptr struct gcnode* mgcptr;

MUTEX_INIT(free_lock);

gcnode free_list_l3bnode;
gcnode free_list_pnode;
gcnode free_list_bnode;
gcnode free_list_bnode_tmp;
// temporarily hold bnodes from being recycled until epoch finish.
// a easy way to solve the problem: "cache removing tasks" may remove cache from a removed node
// this invalid case currently detected and avoided by checking node.len?=-1
gcnode free_list_data_block;

void gc_init_single(gcnode* x) {
    x->size_addr = PPTR(0, INVALID_DPU_ADDR);
}

void gc_init() {
    #ifdef GC_ACTIVE
    #ifdef SKIP_LIST
    assert(false);
    #endif
    gc_init_single(&free_list_l3bnode);
    gc_init_single(&free_list_pnode);
    gc_init_single(&free_list_bnode);
    gc_init_single(&free_list_bnode_tmp);
    gc_init_single(&free_list_data_block);
    #endif
}

void free_node(gcnode* gcn, mpvoid ad) {
    #ifdef GC_ACTIVE
    mutex_lock(free_lock);
    mgcptr gcad = (mgcptr)ad;
    pptr addr = gcn->size_addr;
    gcad->size_addr = addr;
    gcn->size_addr = PPTR(addr.id + 1, gcad);
    mutex_unlock(free_lock);
    #endif
}

// used to move free_list_bnode_tmp to free_list_bnode
void move_free_list(gcnode* src, gcnode* dst) {
    #ifdef GC_ACTIVE
    if (me() != 0) {
        return;
    }
    int size = src->size_addr.id;
    {
        mgcptr cur = (mgcptr)src->size_addr.addr;
        for (int i = 0; i < size; i ++) {
            mgcptr nxt = (mgcptr)cur->size_addr.addr;
            free_node(dst, (mpvoid)cur);
            cur = nxt;
        }
        IN_DPU_ASSERT(cur == INVALID_DPU_ADDR, "mfl! error\n");
    }
    gc_init_single(src);
    #endif
}

pptr alloc_node(gcnode* gcn, int n) { // return (num_node_get, gcnodeaddr)
    #ifndef GC_ACTIVE
    return PPTR(0, INVALID_DPU_ADDR);
    #else
    pptr addr = gcn->size_addr;
    if (addr.id >= n) { // have more nodes than requied
        mgcptr nxt = (mgcptr)gcn->size_addr.addr;
        pptr ret = PPTR(n, nxt);
        for (int i = 0; i < n; i ++) {
            gcnode nxtnode = *nxt;
            IN_DPU_ASSERT(nxt != INVALID_DPU_ADDR, "alloc invalid nxt");
            nxt = (mgcptr)nxtnode.size_addr.addr;
        }
        gcn->size_addr = PPTR(addr.id - n, nxt);
        return ret;
    } else { // run out of recycled nodes
        pptr ret = gcn->size_addr;
        gcn->size_addr = PPTR(0, INVALID_DPU_ADDR);
        return ret;
    }
    #endif
}