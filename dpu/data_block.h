#pragma once

#include <defs.h>
#include <mram.h>
#include <mutex.h>
#include "macro.h"
#include "gc.h"

MUTEX_INIT(db_lock);

typedef struct varlen_buffer {
    int len;
    int capacity;
    int64_t* ptr;
} varlen_buffer;

static inline void varlen_buffer_init(varlen_buffer* buf, int capacity) {
    buf->len = 0;
    buf->capacity = capacity; 
    buf->ptr = mem_alloc(S64(capacity));
}

static inline varlen_buffer* varlen_buffer_new(int capacity) {
    varlen_buffer* buf = mem_alloc(sizeof(varlen_buffer));
    varlen_buffer_init(buf, capacity);
    return buf;
}

static inline bool varlen_buffer_set_capacity(varlen_buffer* buf, int capacity) {
    if (buf->capacity < capacity) {
        while (buf->capacity < capacity) {
            buf->capacity *= 2;
        }
        buf->ptr = mem_alloc(S64(buf->capacity));
        return true;
    }
    return false;
}

static inline bool varlen_buffer_expand(varlen_buffer* buf, int capacity) {
    if (capacity <= buf->capacity) {
        return false;
    }
    varlen_buffer new_buf = *buf;
    bool succ = varlen_buffer_set_capacity(&new_buf, capacity);
    IN_DPU_ASSERT(succ, "vbe: fail\n");
    memcpy(new_buf.ptr, buf->ptr, S64(buf->len));
    *buf = new_buf;
    return true;
}

static inline void varlen_buffer_from_arr(varlen_buffer* buf, int len, int64_t* arr) {
    buf->len = buf->capacity = len;
    buf->ptr = arr;
}

#ifdef DPU_SCAN
static inline bool varlen_buffer_expand_scan(varlen_buffer* buf, int capacity) {
    if (capacity <= buf->capacity) {
        return false;
    }
    int64_t *old_buf_ptr = buf->ptr;
    bool succ = varlen_buffer_set_capacity(buf, capacity);
    IN_DPU_ASSERT(succ, "vbe: fail\n");
    memcpy(buf->ptr, old_buf_ptr, S64(buf->len));
    return true;
}

static inline void varlen_buffer_push(varlen_buffer* buf, int64_t v) {
    int len = buf->len;
    if(len >= buf->capacity) varlen_buffer_expand_scan(buf, len + 1);
    buf->ptr[len] = v;
    buf->len = len + 1;
}
#endif

typedef __mram_ptr struct data_block* mdbptr;

typedef struct len_addr {
    int len;
    mdbptr nxt;
} len_addr;

typedef struct data_block {
    struct len_addr la;
    int64_t data[DB_SIZE];
} data_block;

#ifdef IRAM_FRIENDLY
mdbptr dbbuffer;
#else
__mram_noinit data_block dbbuffer[DB_BUFFER_SIZE / sizeof(data_block)];
#endif
__host uint32_t dbcnt = 1;

mdbptr dbbuffer_start, dbbuffer_end;

static inline bool in_dbbuffer(mdbptr addr) {
    // IN_DPU_ASSERT(dbbuffer_start == dbbuffer + 1&&
    //                   dbbuffer_end == (dbbuffer_start +
    //                                    (DB_BUFFER_SIZE / sizeof(data_block))
    //                                    - 1),
    //               "idb! inv\n");
    return addr >= dbbuffer_start && addr < dbbuffer_end;
}

static inline void data_block_init(mdbptr db) {
    db->la = (len_addr){.len = 0, .nxt = (mdbptr)INVALID_DPU_ADDR};
}

static inline mdbptr alloc_db() {
    mutex_lock(db_lock);
    pptr recycle = alloc_node(&free_list_data_block, 1);
    mdbptr ret;
    if (recycle.id == 0) {
        ret = dbbuffer + dbcnt;
        dbcnt ++;
    } else {
        ret = (mdbptr)recycle.addr;
    }
    data_block_init(ret);
    mutex_unlock(db_lock);
    SPACE_IN_DPU_ASSERT(dbcnt < (DB_BUFFER_SIZE / sizeof(data_block)),
                  "rsdb! of\n");
    return ret;
}

static inline mdbptr data_block_allocate() {
    return alloc_db();
}

int data_block_to_mram(mdbptr db, mpint64_t vals) {
    mdbptr tmp_db = db;
    // IN_DPU_ASSERT((uint32_t)tmp_db != INVALID_DPU_ADDR, "dbtb! inv\n");
    struct data_block actual_db;
    m_read_single(tmp_db, &actual_db, sizeof(data_block));

    int len = actual_db.la.len;

    for (int inslen = 0; inslen < len; inslen += DB_SIZE) {
        int remlen = len - inslen;
        int curlen = (DB_SIZE < remlen) ? DB_SIZE : remlen;
        m_write_single(actual_db.data, vals + inslen, S64(curlen));
        if ((inslen + curlen) < len) {
            tmp_db = actual_db.la.nxt;
            m_read_single(tmp_db, &actual_db, sizeof(data_block));
        }
    }
    return len;
}

void data_block_to_buffer(mdbptr db, varlen_buffer* buf) {
    mdbptr tmp_db = db;
    IN_DPU_ASSERT((uint32_t)tmp_db != INVALID_DPU_ADDR, "dbtb! inv\n");
    struct data_block actual_db = *db;

    int len = actual_db.la.len;
    varlen_buffer_set_capacity(buf, len);
    int64_t* vals = buf->ptr;

    for (int inslen = 0; inslen < len; inslen += DB_SIZE) {
        int remlen = len - inslen;
        int curlen = (DB_SIZE < remlen) ? DB_SIZE : remlen;
        memcpy(vals + inslen, actual_db.data, S64(curlen));
        if ((inslen + curlen) < len) {
            tmp_db = actual_db.la.nxt;
            IN_DPU_ASSERT((uint32_t)tmp_db != INVALID_DPU_ADDR, "dbtb! inv\n");
            actual_db = *tmp_db;
        }
    }
    buf->len = len;
}

void data_block_print(mdbptr db, varlen_buffer* buf, bool x16) {
    data_block_to_buffer(db, buf);
    printf("addr=%x len=%d\n", (uint32_t)db, buf->len);
    for (int i = 0; i < buf->len; i++) {
        if (x16) {
            printf("i=%d v=%llx ", i, buf->ptr[i]);
        } else {
            printf("i=%d v=%lld ", i, buf->ptr[i]);
        }
    }
    printf("\n");
}

void remove_data_blocks(mdbptr db) {
    mdbptr cur = db;
    while ((uint32_t)cur != INVALID_DPU_ADDR) {
        mdbptr nxt = cur->la.nxt;
        free_node(&free_list_data_block, (mpvoid)cur);
        cur = nxt;
    }
}

mdbptr data_block_from_mram(mdbptr db, mpint64_t bufkeys, int len) {
    // IN_DPU_ASSERT(in_dbbuffer(db), "dbfb! inv\n");
    mdbptr ret = db;
    data_block tmp;
    db->la = (len_addr){.len = 0, .nxt = (mdbptr)INVALID_DPU_ADDR};
    for (int inslen = 0; inslen <= len; inslen += DB_SIZE) {
        int curlen = MIN(DB_SIZE, len - inslen);
        if (curlen > 0) {
            m_read_single(bufkeys + inslen, tmp.data, S64(curlen));
        }
        tmp.la = db->la;
        tmp.la.len = len - inslen;
        if (tmp.la.len >= DB_SIZE && (uint32_t)tmp.la.nxt == INVALID_DPU_ADDR) {
            tmp.la.nxt = data_block_allocate();
        }
        m_write_single(&tmp, db, sizeof(data_block));
        db = tmp.la.nxt;
    }
    remove_data_blocks(tmp.la.nxt);
    return ret;
}

// #define MIN(a, b) (((a) < (b)) ? (a) : (b))
mdbptr data_block_from_buffer(mdbptr db, varlen_buffer* buf) {
    // IN_DPU_ASSERT(in_dbbuffer(db), "dbfb! inv\n");
    mdbptr ret = db;
    int64_t* bufkeys = buf->ptr;
    int len = buf->len;
    data_block tmp;
    for (int inslen = 0; inslen <= len; inslen += DB_SIZE) {
        int curlen = MIN(DB_SIZE, len - inslen);
        if (curlen > 0) {
            memcpy(tmp.data, bufkeys + inslen, S64(curlen));
        }
        tmp.la = db->la;
        tmp.la.len = len - inslen;
        if (tmp.la.len >= DB_SIZE && (uint32_t)tmp.la.nxt == INVALID_DPU_ADDR) {
            tmp.la.nxt = data_block_allocate();
        }
        m_write_single(&tmp, db, sizeof(data_block));
        db = tmp.la.nxt;
    }
    return ret;
}