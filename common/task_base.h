#pragma once

/* -------------------------- Task Type -------------------------- */

#define EMPTY 0

#include <stdint.h>
#include <assert.h>
#include "pptr.h"
#include "macro_common.h"
#include "common.h"

/* -------------------------- Level 3 -------------------------- */

// name
// id
// fixed : true for fixed
// length (expected)
// content
#ifndef TASK
#define TASK(NAME, ID, FIXED, LENGTH, CONTENT)
#endif

TASK(empty_task_reply, 0, true, 0, {})

#define L3_INIT_TSK 1
TASK(L3_init_task, 1, true, sizeof(L3_init_task), {
    int64_t key;
    int64_t height;
    pptr down;
})

// #define L3_INIT_REP 2
// TASK(L3_init_reply, 2, true, sizeof(L3_init_reply), {
//     pptr addr; // addr of the upper part node
// })

#define L3_INSERT_TSK 3
TASK(L3_insert_task, 3, true, sizeof(L3_insert_task), {
    int64_t key;
    int64_t height;
    pptr down;
})

// #define L3_INSERT_REP 4
// TASK(L3_insert_reply, 4, true, sizeof(L3_insert_reply), {
//     pptr addr; // addr of the upper part node
// })

#define L3_REMOVE_TSK 5
TASK(L3_remove_task, 5, true, sizeof(L3_remove_task), {
    int64_t key;
    // pptr addr;
})

#define L3_SEARCH_TSK 7
TASK(L3_search_task, 7, true, sizeof(L3_search_task), {
    int64_t key;
})

#define L3_SEARCH_REP 8
TASK(L3_search_reply, 8, true, sizeof(L3_search_reply), {
    pptr addr;
})

// #define L3_SANCHECK_TSK 9
// typedef struct {
//     int64_t nothing;
// } L3_sancheck_task;

// #define L3_GET_TSK 10
// typedef struct {
//     int64_t key;
// } L3_get_task;

// #define L3_GET_REP 11
// typedef struct {
//     int64_t available;
// } L3_get_reply;

// #define L3_BUILD_D_TSK 12
// TASK(L3_build_d_task, 12, true, sizeof(L3_build_d_task), {
//     pptr addr;
//     pptr down;
// })

// Range Scan
#define L3_SCAN_TSK 13
TASK(L3_scan_task, 13, true, sizeof(L3_scan_task), {
    int64_t lkey;
    int64_t rkey;
})

#define L3_SCAN_REP 14
TASK(L3_scan_reply, 14, false, sizeof(L3_scan_reply), {
    int64_t len;
    pptr addr[];
})

/* -------------------------- B node -------------------------- */

#define B_NEWNODE_TSK 301
TASK(b_newnode_task, 301, true, sizeof(b_newnode_task), {
    int64_t height;
})

#define B_NEWNODE_REP 302
TASK(b_newnode_reply, 302, true, sizeof(b_newnode_reply), {
    pptr addr;
})

#define B_GET_NODE_TSK 303
TASK(b_get_node_task, 303, true, sizeof(b_get_node_task), {
    pptr addr;
})

#define B_GET_NODE_REP 304
TASK(b_get_node_reply, 304, false, S64(L2_SIZE * 2 + 1), {
    int64_t len;
    int64_t vals[];
    // int64_t keys[];
    // pptr addrs[];
})

#define B_SEARCH_TSK 305 // wrong expect length
// TASK(b_search_task, 305, true, sizeof(b_search_task), {
//     pptr addr;
//     int64_t key;
// })
TASK(b_search_task, 305, false, sizeof(b_search_task), {
    pptr addr;
    int64_t len;
    int64_t keys[];
})

#define B_SEARCH_REP 306
// TASK(b_search_reply, 306, true, sizeof(b_search_reply), {
//     pptr addr;
// })
TASK(b_search_reply, 306, false, sizeof(b_search_reply), {
    int64_t len;
    int64_t addrs[];
})

#define B_INSERT_TSK 307
TASK(b_insert_task, 307, false, S64(2 + 2), {
    pptr addr;
    int64_t len;
    int64_t vals[];
    // int64_t keys[];
    // pptr addrs[];
})

#define B_TRUNCATE_TSK 308
TASK(b_truncate_task, 308, true, sizeof(b_truncate_task), {
    pptr addr;
    int64_t key;
})

#define B_TRUNCATE_REP 309
TASK(b_truncate_reply, 309, false, S64(L2_SIZE * 2 + 2), {
    int64_t len;
    pptr right;
    int64_t vals[];
    // int64_t keys[];
    // pptr addrs[];
})

#define B_SEARCH_WITH_PATH_TSK 310
// TASK(b_search_with_path_task, 310, true, sizeof(b_search_with_path_task), {
//     pptr addr;
//     int64_t key;
//     int64_t height;
// })
TASK(b_search_with_path_task, 310, false, sizeof(b_search_with_path_task), {
    pptr addr;
    int64_t len;
    int64_t vals[];
    // int64_t keys[];
    // int64_t heights[];
})
static inline int b_search_with_path_task_siz(int len) {
    return ((len + 1) << 1);
}

#define B_SEARCH_WITH_PATH_REP 311
// TASK(b_search_with_path_reply, 311, true, sizeof(b_search_with_path_reply), {
//     pptr addr2;
//     pptr addr1;
//     pptr addr0;
// })
TASK(b_search_with_path_reply, 311, false, sizeof(b_search_with_path_reply), {
    int64_t len;
    offset_pptr ops[];
    // int64_t addrs[];
})

#define B_FIXED_SEARCH_TSK 312
TASK(b_fixed_search_task, 312, true, sizeof(b_fixed_search_task), {
    pptr addr;
    int64_t key;
})

#define B_FIXED_SEARCH_REP 313
TASK(b_fixed_search_reply, 313, true, sizeof(b_fixed_search_reply), {
    pptr addr;
})

#define B_SET_LR_TSK 314
TASK(b_set_lr_task, 314, true, sizeof(b_set_lr_task), {
    pptr addr;
    pptr left;  // null_pptr for not setting
    pptr right;
})

#define B_REMOVE_TSK 315
TASK(b_remove_task, 315, false, S64(2 + 1), {
    pptr addr;
    int64_t len;
    int64_t keys[];
})

#define B_REMOVE_GET_NODE_TSK 316
TASK(b_remove_get_node_task, 316, true, sizeof(b_remove_get_node_task),
     { pptr addr; })

#define B_REMOVE_GET_NODE_REP 317
TASK(b_remove_get_node_reply, 317, false, S64(L2_SIZE * 2 + 3), {
    pptr left;
    pptr right;
    int64_t len;
    int64_t vals[];
})

// #define B_SET_U_TSK 318
// TASK(b_set_u_task, 318, true, sizeof(b_set_u_task), {
//     pptr addr;
//     pptr up;
// })

// #define B_GET_U_TSK 319
// TASK(b_get_u_task, 319, true, sizeof(b_get_u_task), { pptr addr; })

// #define B_GET_U_REP 320
// TASK(b_get_u_reply, 320, true, sizeof(b_get_u_reply), { pptr up; })

// #define B_REMOVE_REP 316
// typedef struct {
//     int64_t len;
//     int64_t vals[]; // S64(2+2*len)
//     // pptr left, right; len*(key,down);
// } b_remove_reply;

// Range Scan
#define B_FETCH_CHILD_TSK 321
TASK(b_fetch_child_task, 321, true, sizeof(b_fetch_child_task), {
    pptr addr;
})

#define B_FETCH_CHILD_REP 322
TASK(b_fetch_child_reply, 322, false, S64((L2_SIZE << 1) + 1), {
    int64_t len;
    pptr addr[];
})

#define B_SCAN_SEARCH_TSK 323
TASK(b_scan_search_task, 323, true, sizeof(b_scan_search_task), {
    pptr addr;
    int64_t lkey;
    int64_t rkey;
})

#define B_SCAN_SEARCH_REP 324
TASK(b_scan_search_reply, 324, false, S64((L2_SIZE << 1) + 1), {
    int64_t len;
    pptr addr[];
})

/* -------------------------- P node -------------------------- */
#define P_NEWNODE_TSK 401
TASK(p_newnode_task, 401, true, sizeof(p_newnode_task), {
    int64_t key;
    int64_t height;
    int64_t value;
})

#define P_NEWNODE_REP 402
TASK(p_newnode_reply, 402, true, sizeof(p_newnode_reply), { pptr addr; })

#define P_GET_KEY_TSK 403
TASK(p_get_key_task, 403, true, sizeof(p_get_key_task), { pptr addr; })

#define P_GET_KEY_REP 404
TASK(p_get_key_reply, 404, true, sizeof(p_get_key_reply), {
    int64_t key;
    int64_t value;
})

#define P_GET_TSK 405
TASK(p_get_task, 405, true, sizeof(p_get_task), { int64_t key; })

#define P_GET_REP 406
TASK(p_get_reply, 406, true, sizeof(p_get_reply), { int64_t key; int64_t value;})

#define P_GET_HEIGHT_TSK 407
TASK(p_get_height_task, 407, true, sizeof(p_get_height_task), { int64_t key; })

#define P_GET_HEIGHT_REP 408
TASK(p_get_height_reply, 408, true, sizeof(p_get_height_reply),
     { int64_t height; })

#define P_UPDATE_TSK 409
TASK(p_update_task, P_UPDATE_TSK, true, sizeof(p_update_task), { int64_t key; int64_t value;})

#define P_UPDATE_REP 410
TASK(p_update_reply, P_UPDATE_REP, true, sizeof(p_update_reply), { int64_t valid; })

/* -------------------------- Cache -------------------------- */
#define CACHE_NEWNODE_TSK 501
TASK(cache_newnode_task, 501, false, S64(L2_SIZE * 2 + 3), {
    pptr addr;
    pptr caddr;
    int64_t len;
    int64_t vals[];
    // int64_t keys[];
    // pptr addrs[];
})

#define CACHE_INSERT_TSK 502
TASK(cache_insert_task, 502, true, sizeof(cache_insert_task), {
    pptr addr;
    int64_t key;
    pptr t_addr;
    int64_t height;
})

#define CACHE_INIT_REQ_TSK 503
TASK(cache_init_request_task, 503, true, sizeof(cache_init_request_task),
     { int64_t nothing; })

#define CACHE_INIT_REQ_REP 504
TASK(cache_init_request_reply, 504, false, S64(1 + 1), {
    int64_t len;
    int64_t vals[];
})

#define CACHE_TRUNCATE_TSK 505
TASK(cache_truncate_task, 505, true, sizeof(cache_truncate_task), {
    pptr addr;  // source
    int64_t key;
    int64_t height;
})

#define CACHE_REMOVE_TSK 506
TASK(cache_remove_task, 506, true, sizeof(cache_remove_task), {
    pptr addr;
    int64_t key;
    int64_t height;
})

#define CACHE_MULTI_INSERT_TSK 507
TASK(cache_multi_insert_task, 507, false, sizeof(3 + 2), {
    pptr addr;
    int64_t len;
    int64_t height;
    int64_t vals[];
    // int64_t keys[];
    // int64_t addrs[];
})

/* -------------------------- Util -------------------------- */
#define STATISTICS_TSK 1001
TASK(statistic_task, 1001, true, sizeof(statistic_task), { int64_t dpu_id; })

#define INIT_TSK 1002
TASK(dpu_init_task, 1002, true, sizeof(dpu_init_task), { int64_t dpu_id; })

#define INIT_REP 1003
TASK(dpu_init_reply, 1003, true, sizeof(dpu_init_reply), {
    uint64_t bbuffer_start;
    uint64_t bbuffer_end;
    uint64_t pbuffer_start;
    uint64_t pbuffer_end;
})

// #define TICK_TSK 1004
// typedef struct {
//     int64_t nothing;
// } tick_task;
