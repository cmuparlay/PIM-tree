#pragma once

#include <defs.h>
#include <mram.h>
#include <perfcounter.h>
#include <string.h>
#include "data_block.h"
#include "macro.h"

/* -------------------------- Make Sure! sizeof(Everything) = 8x -------------------------- */
typedef __mram_ptr pptr* mppptr;
typedef __mram_ptr int64_t* mpint64_t;
typedef __mram_ptr uint8_t* mpuint8_t;
typedef __mram_ptr struct L3node* mL3ptr;
typedef __mram_ptr struct L3Bnode* mL3Bptr;
typedef __mram_ptr struct L2node* mL2ptr;
typedef __mram_ptr struct L1node* mL1ptr;
typedef __mram_ptr struct Bnode* mBptr;
typedef __mram_ptr struct Pnode* mPptr;

typedef struct L3node {
    int64_t key;
    int64_t height;
    pptr down;
    mppptr left __attribute__((aligned (8)));
    mppptr right __attribute__((aligned (8)));
} L3node;

typedef struct L3Bnode {
    pptr up, right;
    int64_t height;
    int64_t size;
    int64_t keys[DB_SIZE];
    pptr addrs[DB_SIZE];
} L3Bnode;

// bug fix by making len not at the first position to avoid
// overwritten when doing garbage collection.
// check "gcnode" for position information.
typedef struct Bnode {
    int64_t height;
    int64_t len;
    pptr up, left, right;
    mdbptr keys, addrs;
    mdbptr caddrs, padding;
} Bnode;

typedef struct Pnode {
    int64_t key;
    int64_t value;
    int64_t height;
} Pnode;

void b_node_print(pptr ptr, varlen_buffer* buf) {
    printf("Bnode=%llx\n", PPTR_TO_I64(ptr));
    mBptr addr = (mBptr)ptr.addr;
    printf("len=%lld height=%lld\n", addr->len, addr->height);
    data_block_print(addr->keys, buf, true);
    data_block_print(addr->addrs, buf, true);
    data_block_print(addr->caddrs, buf, true);
}

extern int64_t DPU_ID;

static pptr ml3bptr_to_pptr(const mL3Bptr addr) {
    return (pptr){.id = DPU_ID, .addr = (uint32_t)addr};
}
static inline mL3Bptr pptr_to_ml3bptr(pptr x) { return (mL3Bptr)x.addr; }
