#pragma once
#include "common.h"
#include "task_framework_dpu.h"
#include "node_dpu.h"
#include <defs.h>
#include <mram.h>
#include <mutex.h>
#include <alloc.h>

// Range Scan

#ifdef DPU_SCAN
// MRAM Buffer Size
#define M_BUFFER_SIZE (L3_BUFFER_SIZE >> 10)
#define VARLEN_BUFFER_SIZE (64)
__mram_noinit int64_t mrambuffer[M_BUFFER_SIZE];

typedef struct varlen_buffer_dpu {
    int64_t len;
    int llen;
    int capacity;
    int64_t* ptr;
    mpint64_t ptr_mram;
} varlen_buffer_dpu;

static inline void varlen_buffer_init_dpu(varlen_buffer_dpu* buf, int capacity, mpint64_t mptr) {
    buf->len = 0;
    buf->llen = 0;
    buf->capacity = capacity; 
    buf->ptr = (int64_t*)mem_alloc(S64(capacity));
    buf->ptr_mram = mptr;
}

static inline varlen_buffer_dpu* varlen_buffer_new_dpu(int capacity, mpint64_t mptr) {
    varlen_buffer_dpu* buf = (varlen_buffer_dpu*)mem_alloc(sizeof(varlen_buffer_dpu));
    varlen_buffer_init_dpu(buf, capacity, mptr);
    return buf;
}

static inline void varlen_buffer_push_dpu(varlen_buffer_dpu* buf, int64_t v) {
    if(buf->llen == buf->capacity) {
        m_write(buf->ptr, (buf->ptr_mram + buf->len - buf->capacity), S64(buf->capacity));
        buf->llen = 1;
        buf->ptr[0] = v;
    }
    else{
        buf->ptr[buf->llen] = v;
        buf->llen++;
    }
    buf->len++;
}

static inline void varlen_buffer_to_mram_dpu(varlen_buffer_dpu* buf, mpint64_t mptr) {
    m_write(buf->ptr, (mptr + buf->len - buf->llen), S64(buf->llen));
    mram_to_mram(mptr, buf->ptr_mram, S64(buf->len - buf->llen));
}

static inline void varlen_buffer_reset_dpu(varlen_buffer_dpu* buf) {
    buf->len = 0;
    buf->llen = 0;
}

static inline int64_t varlen_buffer_element_dpu(varlen_buffer_dpu* buf, int64_t idx) {
    if(idx >= buf->len - buf->llen) {
        return buf->ptr[idx - buf->len + buf->llen];
    }
    else {
        int64_t res;
        m_read(buf->ptr_mram + idx, &res, S64(1));
        return res;
    }
}

static inline void varlen_buffer_set_element_dpu
    (varlen_buffer_dpu* buf, int64_t idx, int64_t value) {
    if(idx >= buf->len - buf->llen) {
        buf->ptr[idx - buf->len + buf->llen] = value;
    }
    else {
        m_write(&value, buf->ptr_mram + idx, S64(1));
    }
}
#endif
