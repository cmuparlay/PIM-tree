#pragma once

// #define KHB_DEBUG
// #define KHB_CPU_DEBUG

// #ifdef KHB_DEBUG
// #define ASSERT(x) assert(x)
// #else
// #define ASSERT(x) x
// #endif

/* Size of the buffer on which the checksum will be performed */
// #define BUFFER_SIZE (200)
// #define LOWER_PART_HEIGHT (6)
#define BATCH_SIZE (2100000)

// 12 levels
#define LOGP (12)
#define LOWER_PART_HEIGHT (12)

// L0,1,2,3 25MB
// #define LX_BUFFER_SIZE (12 << 20)
#define L3_BUFFER_SIZE (12 << 20)
#define L2_BUFFER_SIZE (36 << 20)

// HASH TABLE 2MB. should be power of 2
#define LX_HASHTABLE_SIZE ((1 << 12) >> 3)

#define MAX_TASK_BUFFER_SIZE_PER_DPU (800 << 10) // 800 KB
#define MAX_TASK_COUNT_PER_DPU_PER_BLOCK ((100 << 10) >> 3) // 100 KB = 12.5 K

#define MAX_L3_HEIGHT (20)

static inline int hh(int64_t key, uint64_t height, uint64_t M) {
    assert(height == 0);
    // printf("KEY: %lld\n", key);
    key = (key % M);
    // printf("KEY: %lld\n", key);
    key = (key < 0) ? (key + M) : key;
    // printf("KEY: %lld\n", key);
    key = key * 47 + height;
    // printf("KEY: %lld\n", key);
    return (key * 23 + 17) % M;
}

static inline int hash_to_dpu(int64_t key, uint64_t height, uint64_t M) {
    return hh(key, height, M);
}

static inline int hash_to_addr(int64_t key, uint64_t height, uint64_t M) {
    return hh(key, height, M);
}