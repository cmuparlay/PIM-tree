#pragma once
#include "node_dpu.h"

typedef __mram_ptr struct cache_init_record* mcirptr;

typedef struct cache_init_record {
    pptr source;
    pptr addr;
    pptr request;
} cache_init_record;
mcirptr cirbuffer_start, cirbuffer_end;

#ifdef IRAM_FRIENDLY
mcirptr cirbuffer;
#else
__mram_noinit cache_init_record cirbuffer[CACHE_INIT_RECORD_SIZE / sizeof(cache_init_record)];
#endif
__host uint32_t circnt = 1;
MUTEX_INIT(cir_lock);

static inline bool in_cirbuffer(mcirptr addr) {
    // IN_DPU_ASSERT(
    //     cirbuffer_start == cirbuffer + 1 &&
    //         cirbuffer_end == (cirbuffer + (CACHE_INIT_RECORD_SIZE / sizeof(cache_init_record))),
    //     "ic! inv\n");
    return addr >= cirbuffer_start && addr < cirbuffer_end;
}

static inline mcirptr reserve_space_cache_init_record(int len) {
    mutex_lock(cir_lock);
    mcirptr ret = cirbuffer + circnt;
    circnt += len;
    mutex_unlock(cir_lock);
    SPACE_IN_DPU_ASSERT(circnt < (CACHE_INIT_RECORD_SIZE / sizeof(cache_init_record)), "rscir! of\n");
    return ret;
}