#pragma once

// local linear probing hash table
// parallel read
// serial insert / delete

#include <mutex.h>
#include "debug.h"
#include "node_dpu.h"
#include "common.h"

typedef struct ht_slot {
    uint32_t pos;  // ideal position in the hash table
    uint32_t v;    // value
} ht_slot;

#define null_ht_slot ((ht_slot){.pos = 0, .v = 0})

// hash table
MUTEX_INIT(ht_lock);
#ifdef IRAM_FRIENDLY
__mram_ptr ht_slot * ht;
#else
__mram_noinit ht_slot ht[LX_HASHTABLE_SIZE]; // must be 8 bytes aligned
#endif
__host int htcnt = 0;

static inline void storage_init() {
    ht_slot hs = null_ht_slot;
    for (int i = 0; i < LX_HASHTABLE_SIZE; i++) {
        m_write(&hs, ht + i, sizeof(ht_slot));
    }
}

static inline int32_t ht_insert(__mram_ptr ht_slot* ht, int* cnt, int32_t pos,
                             uint32_t val) {
    mutex_lock(ht_lock);
    int ipos = pos;
    ht_slot hs;
    m_read_single(ht + pos, &hs, sizeof(ht_slot));
    while (hs.v != 0) {  // find slot
        pos = (pos + 1) & (LX_HASHTABLE_SIZE - 1);
        m_read_single(ht + pos, &hs, sizeof(ht_slot));
        IN_DPU_ASSERT(pos != ipos, "htisnert: full\n");
    }
    ht_slot hh = (ht_slot){.pos = ipos, .v = val};
    m_write_single(&hh, ht + pos, sizeof(ht_slot));
    *cnt = *cnt + 1;
    mutex_unlock(ht_lock);
    return pos;
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
    int ipos = hash_to_addr(key, LX_HASHTABLE_SIZE);
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
