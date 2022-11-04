#pragma once

#define NR_OF_DRIVERS (3)

/* Size of the buffer on which the checksum will be performed */
#define BATCH_SIZE (2100000)

#define MAX_L3_HEIGHT (20)
// 12 levels
#define L2_HEIGHT (3)

#define L2_SIZE_LOG (4)
#define L2_SIZE (1 << L2_SIZE_LOG)
#define DB_SIZE (16)  // size of the data block

// HASH TABLE 8MB. should be power of 2
#define LX_HASHTABLE_SIZE ((4 << 20) >> 3)
// cache init record size 300KB
#define CACHE_INIT_RECORD_SIZE (300 << 10)

#define CACHE_HEIGHT (2)
#define L3_BUFFER_SIZE (6 << 20) // 6 MB
#define B_BUFFER_SIZE (3 << 19) // 1.5 MB
#define DB_BUFFER_SIZE (17 << 20) // 17 MB
#define P_BUFFER_SIZE (15 << 19) // 7.5MB

#define MAX_TASK_BUFFER_SIZE_PER_DPU (800 << 13) // 6.4 MB
#define MAX_TASK_COUNT_PER_DPU_PER_BLOCK ((100 << 10) >> 3) // 100 KB = 12.5 K

// L3_SKIP_LIST macro obsolete.
// Do not use L3_SKIP_LIST. The program may crash with it.
// #define L3_SKIP_LIST

/* Structure used by both the host and the dpu to communicate information */

static inline int lb(int64_t x) { return x & (-x); }

static inline int hh_dpu(int64_t key, uint64_t M) { return key & (M - 1); }

static inline int hash_to_addr(int64_t key, uint64_t M) {
    return hh_dpu(key, M);
}

// static inline bool valid_pptr(pptr x, int nr_of_dpus) {
//     return (x.id < (uint32_t)nr_of_dpus);
// }
