#pragma once

#include "common.h"
#include "node_dpu.h"
#include "stdlib.h"
#include <mutex.h>
// #include "task_dpu.h"
// #include "garbage_collection.h"

typedef struct ht_slot {
    uint32_t pos;  // ideal position in the hash table
    uint32_t v;    // value
} ht_slot;

// #define null_ht_slot ((ht_slot){.pos = 0, .v = 0})

const ht_slot null_ht_slot = ((ht_slot){.pos = 0, .v = 0});

// #define LX_HASHTABLE_SIZE (1 << 10)
// L3
MUTEX_INIT(get_new_L3_lock);
MUTEX_INIT(ht_lock);
extern __mram_ptr ht_slot l3ht[];  // must be 8 bytes aligned. 0 as null.
extern int l3htcnt;
extern __mram_ptr uint8_t l3buffer[];
extern int l3cnt;

// L2
MUTEX_INIT(get_new_L2_lock);
extern __mram_ptr ht_slot l2ht[];  // must be 8 bytes aligned. 0 as null.
extern int l2htcnt;
extern __mram_ptr uint8_t l2buffer[];
extern int l2cnt;

extern mL3ptr root;

// __mram_noinit uint8_t l2buffer[LX_BUFFER_SIZE];
// int l2cnt;

// __mram_noinit uint8_t l1buffer[LX_BUFFER_SIZE];
// int l1cnt;

// __mram_noinit uint8_t l0buffer[LX_BUFFER_SIZE];
// int l0cnt;

// __mram_ptr uint8_t* l2ht[LX_HASHTABLE_SIZE];
// __mram_ptr uint8_t* l1ht[LX_HASHTABLE_SIZE];
// __mram_ptr uint8_t* l0ht[LX_HASHTABLE_SIZE];


__host bool storage_inited = false;
static inline void storage_init() {
    if (storage_inited) {
        return;
    }
    storage_inited = true;
    ht_slot hs = null_ht_slot;
    for (int i = 0; i < LX_HASHTABLE_SIZE; i++) {
        l3ht[i] = hs;
        l2ht[i] = hs;
    }
    // L3_gc_init();
}
static inline void ht_insert(__mram_ptr ht_slot* ht, int* cnt, int32_t pos,
                             uint32_t val) {
    mutex_lock(ht_lock);
    int ipos = pos;
    ht_slot hs = ht[pos];
    while (hs.v != 0) {  // find slot
        pos = (pos + 1) & (LX_HASHTABLE_SIZE - 1);
        hs = ht[pos];
        IN_DPU_ASSERT(pos != ipos, "htisnert: full\n");
    }
    ht[pos] = (ht_slot){.pos = ipos, .v = val};
    *cnt = *cnt + 1;
    mutex_unlock(ht_lock);
}

static inline bool ht_no_greater_than(int a, int b) {  // a <= b with wrapping
    int delta = b - a;
    if (delta < 0) {
        delta += LX_HASHTABLE_SIZE;
    }
    return delta < (LX_HASHTABLE_SIZE >> 1);
}

static inline void ht_delete(__mram_ptr ht_slot* ht, int* cnt, int32_t pos,
                             uint32_t val) {
    mutex_lock(ht_lock);
    int ipos = pos;  // initial position
    ht_slot hs = ht[pos];
    while (hs.v != val) {  // find slot
        pos = (pos + 1) & (LX_HASHTABLE_SIZE - 1);
        hs = ht[pos];
        IN_DPU_ASSERT(pos != ipos, "htisnert: full\n");
    }
    ipos = pos;  // position to delete
    pos = (pos + 1) & (LX_HASHTABLE_SIZE - 1);

    while (true) {
        hs = ht[pos];
        if (hs.v == 0) {
            ht[ipos] = null_ht_slot;
            break;
        } else if (ht_no_greater_than(hs.pos, ipos)) {
            ht[ipos] = hs;
            ipos = pos;
        } else {
        }
        pos = (pos + 1) & (LX_HASHTABLE_SIZE - 1);
        IN_DPU_ASSERT(pos != ipos, "htisnert: full\n");
    }
    *cnt = *cnt - 1;
    mutex_unlock(ht_lock);
}

static inline uint32_t ht_search(__mram_ptr ht_slot* ht, int64_t key,
                                 int (*filter)(ht_slot, int64_t)) {
    int ipos = hash_to_addr(key, 0, LX_HASHTABLE_SIZE);
    int pos = ipos;
    while (true) {
        ht_slot hs = ht[pos];  // pull to wram
        int v = filter(hs, key);
        if (v == -1) {  // empty slot
            return INVALID_DPU_ADDR;
            // continue;
        } else if (v == 0) {  // incorrect value
            pos = (pos + 1) & (LX_HASHTABLE_SIZE - 1);
        } else if (v == 1) {  // correct value;
            return (uint32_t)hs.v;
        }
        IN_DPU_ASSERT(pos != ipos, "htisnert: full\n");
    }
}

// L3
static inline uint32_t L3_node_size(int height) {
    return sizeof(L3node) + sizeof(pptr) * height * 2;
}

static inline L3node* init_L3(int64_t key, int height, pptr down,
                              uint8_t* buffer, __mram_ptr void* maddr) {
    L3node* nn = (L3node*)buffer;
    nn->key = key;
    nn->height = height;
    nn->down = down;
    nn->left = (mppptr)(maddr + sizeof(L3node));
    nn->right = (mppptr)(maddr + sizeof(L3node) + sizeof(pptr) * height);
    // for (int i = 0; i < sizeof(pptr) * height * 2; i ++) {
    //     buffer[sizeof(L3node) + i] = (uint8_t)-1;
    // }
    memset(buffer + sizeof(L3node), -1, sizeof(pptr) * height * 2);
    return nn;
}

static inline __mram_ptr void* reserve_space_L3(uint32_t size) {
    // mutex_lock(get_new_L3_lock);
    __mram_ptr void* ret = l3buffer + l3cnt;
    l3cnt += size;
    IN_DPU_ASSERT(l3cnt < L3_BUFFER_SIZE, "rs3! of\n");
    // mutex_unlock(get_new_L3_lock);
    return ret;
}

static inline mL3ptr get_new_L3(int64_t key, int height, pptr down, __mram_ptr void* maddr) {
    int size = L3_node_size(height);
    // __mram_ptr void* maddr = reserve_space_L3(size);
    uint8_t buffer[sizeof(L3node) + sizeof(pptr) * 2 * MAX_L3_HEIGHT];
    L3node* nn = init_L3(key, height, down, buffer, maddr);
    mram_write((void*)nn, maddr, size);
    // ht_insert(l3ht, &l3htcnt, hash_to_addr(key, 0, LX_HASHTABLE_SIZE),
    //           (uint32_t)maddr);
    return maddr;
}

// L2
static inline uint32_t L2_node_size(int height) {
    return sizeof(L2node) + sizeof(int64_t) * height + sizeof(pptr) * height * 2;
}

static inline L2node* init_L2(int64_t key, int height, pptr down,
                              uint8_t* buffer, __mram_ptr void* maddr) {
    IN_DPU_ASSERT(height <= LOWER_PART_HEIGHT, "init L2: wrong height");
    L2node* nn = (L2node*)buffer;
    nn->key = key;
    nn->height = height;
    nn->down = down;
    nn->chk = (mpint64_t)(maddr + sizeof(L2node));
    nn->left = (mppptr)(maddr + sizeof(L2node) + sizeof(int64_t) * height);
    nn->right = (mppptr)(maddr + sizeof(L2node) + sizeof(int64_t) * height + sizeof(pptr) * height);
    memset(buffer + sizeof(L2node), -1, sizeof(int64_t) * height + sizeof(pptr) * height * 2);
    return nn;
}

static inline __mram_ptr void* reserve_space_L2(uint32_t size) {
    __mram_ptr void* ret = l2buffer + l2cnt;
    l2cnt += size;
    IN_DPU_ASSERT(l2cnt < L2_BUFFER_SIZE, "rs2! of\n");
    return ret;
}

static inline mL2ptr get_new_L2(int64_t key, int height, pptr down,
                                __mram_ptr void* maddr) {
    int l2height = (height > LOWER_PART_HEIGHT) ? LOWER_PART_HEIGHT : height;
    // IN_DPU_ASSERT(height <= LOWER_PART_HEIGHT, "get new L2: height too big");
    int size = L2_node_size(l2height);
    uint8_t buffer[sizeof(L2node) +
                   (sizeof(int64_t) + sizeof(pptr) * 2) * LOWER_PART_HEIGHT];
    L2node* nn = init_L2(key, l2height, down, buffer, maddr);
    nn->height = height;
    mram_write((void*)nn, maddr, size);
    // ht_insert(l2ht, &l2htcnt, hash_to_addr(key, 0, LX_HASHTABLE_SIZE),
    //           (uint32_t)maddr);
    return maddr;
}
