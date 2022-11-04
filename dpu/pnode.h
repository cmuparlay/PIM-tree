#pragma once

#include <alloc.h>
#include "hashtable_l3size.h"
#include "common.h"
#include "gc.h"

MUTEX_INIT(p_lock);

// Bnode
#ifdef IRAM_FRIENDLY
mPptr pbuffer;
#else
__mram_noinit Pnode pbuffer[P_BUFFER_SIZE / sizeof(Pnode)];
#endif
__host uint32_t pcnt = 1;
mPptr pbuffer_start, pbuffer_end;

#ifdef IRAM_FRIENDLY
extern __mram_ptr ht_slot * ht;
#else
extern __mram_ptr ht_slot ht[LX_HASHTABLE_SIZE];  // must be 8 bytes aligned
#endif
extern int htcnt;

static inline mPptr alloc_pn() {
    mutex_lock(p_lock);
    pptr recycle = alloc_node(&free_list_pnode, 1);
    mPptr ret;
    if (recycle.id == 0) {
        ret = pbuffer + pcnt;
        pcnt ++;
    } else {
        ret = (mPptr)recycle.addr;
    }
    mutex_unlock(p_lock);
    SPACE_IN_DPU_ASSERT(pcnt < (P_BUFFER_SIZE / sizeof(Pnode)), "rsp! of\n");
    return ret;
}

static inline bool in_pbuffer(mPptr addr) {
    return addr >= pbuffer_start && addr < pbuffer_end;
}

static inline void p_newnode(int64_t _key, int64_t _value, int64_t height, mPptr newnode) {
    Pnode nn;
    nn.key = _key;
    nn.height = height;
    nn.value = _value;

    m_write(&nn, newnode, sizeof(Pnode));
    // IN_DPU_ASSERT(LX_HASHTABLE_SIZE == lb(LX_HASHTABLE_SIZE),
    //               "hh_dpu! not 2^x\n");
    int ret = ht_insert(ht, &htcnt, hash_to_addr(_key, LX_HASHTABLE_SIZE),
              (uint32_t)newnode);
    (void)ret;
}

static inline int p_ht_get(ht_slot v, int64_t key) {
    if (v.v == 0) {
        return -1;
    }
    mPptr addr = (mPptr)v.v;
    if (addr->key == key) {
        return 1;
    }
    return 0;
}

static inline pptr p_get(int64_t key) {
    uint32_t htv = ht_search(ht, key, p_ht_get);
    if (htv == INVALID_DPU_ADDR) {
        return (pptr){.id = INVALID_DPU_ID, .addr = INVALID_DPU_ADDR};
    } else {
        return (pptr){.id = DPU_ID, .addr = htv};
    }
}

static inline int64_t p_get_key(pptr addr) {
    mPptr ptr = (mPptr)addr.addr;
    // IN_DPU_ASSERT((ptr >= pbuffer) && (ptr < (pbuffer + (B_BUFFER_SIZE /
    // sizeof(Pnode))), "pgt! inv\n");
    return ptr->key;
}

static inline int64_t p_get_value(pptr addr) {
    mPptr ptr = (mPptr)addr.addr;
    // IN_DPU_ASSERT((ptr >= pbuffer) && (ptr < (pbuffer + (B_BUFFER_SIZE /
    // sizeof(Pnode))), "pgt! inv\n");
    return ptr->value;
}

static inline int64_t p_get_height(int64_t key) {
    uint32_t htv = ht_search(ht, key, p_ht_get);
    if (htv == INVALID_DPU_ADDR) {
        return -1ll;
    } else {
        ht_delete(ht, &htcnt, hash_to_addr(key, LX_HASHTABLE_SIZE), htv);
        free_node(&free_list_pnode, (mpvoid)htv);
        mPptr nn = (mPptr)htv;
        return nn->height;
    }
}
