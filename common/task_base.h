#pragma once

/* -------------------------- Task Type -------------------------- */

#define EMPTY 0

#include <stdint.h>
#include <assert.h>
#include "pptr.h"
#include "macro_common.h"
#include "common.h"

#ifndef TASK
#define TASK(NAME, ID, FIXED, LENGTH, CONTENT)
#endif

/* -------------------------- Level 3 -------------------------- */

#define EMPTY 0
TASK(empty_task_reply, 0, true, 0, {})

#define L3_INIT_TSK 1
TASK(L3_init_task, 1, true, sizeof(L3_init_task), {
    int64_t key;
    pptr addr;
    int64_t height;
})

#define L3_INIT_REP 2
TASK(L3_init_reply, 2, true, sizeof(L3_init_reply), {
    pptr addr;
})

#define L3_INSERT_TSK 3
TASK(L3_insert_task, 3, true, sizeof(L3_insert_task), {
    int64_t key;
    pptr addr;
    int64_t height;
})

#define L3_INSERT_REP 4
TASK(L3_insert_reply, 4, true, sizeof(L3_insert_reply), {
    pptr addr;
})

#define L3_SEARCH_TSK 7
TASK(L3_search_task, 7, true, sizeof(L3_search_task), {
    int64_t key;
})

#define L3_SEARCH_REP 8
TASK(L3_search_reply, 8, true, sizeof(L3_search_reply), {
    pptr addr;
})

#define L3_GET_TSK 10
TASK(L3_get_task, 10, true, sizeof(L3_get_task), {
    int64_t key;
})

#define L3_GET_REP 11
TASK(L3_get_reply, 11, true, sizeof(L3_get_reply), {
    int64_t available;
})

/* -------------------------- Level 2 -------------------------- */

#define L2_INIT_TSK 101
TASK(L2_init_task, 101, true, sizeof(L2_init_task), {
    int64_t key;
    pptr addr;
    int64_t height;
})

#define L2_INIT_REP 102
TASK(L2_init_reply, 102, true, sizeof(L2_init_reply), {
    pptr addr;
})

#define L2_INSERT_TSK 103
TASK(L2_insert_task, 103, true, sizeof(L2_insert_task), {
    int64_t key;
    pptr addr;
    int64_t height;
})

#define L2_INSERT_REP 104
TASK(L2_insert_reply, 104, true, sizeof(L2_insert_reply), {
    pptr addr;
})

#define L2_BUILD_LR_TSK 105
TASK(L2_build_lr_task, 105, true, sizeof(L2_build_lr_task), {
    pptr addr;
    int64_t chk;
    int64_t height; // positive for right, negative for left
    pptr val;
})

#define L2_BUILD_UP_TSK 106
TASK(L2_build_up_task, 106, true, sizeof(L2_build_up_task), {
    pptr addr;
    pptr up;
})

#define L2_GET_NODE_TSK 109
TASK(L2_get_node_task, 109, true, sizeof(L2_get_node_task), {
    pptr addr;
    int64_t height;
})

#define L2_GET_NODE_REP 110
TASK(L2_get_node_reply, 110, true, sizeof(L2_get_node_reply), {
    int64_t chk;
    pptr right;
})

#define L2_SEARCH_TSK 111
TASK(L2_search_task, 111, true, sizeof(L2_search_task), {
    int64_t key;
    pptr addr;
    int64_t height;
})

#define L2_SEARCH_REP 112
TASK(L2_search_reply, 112, true, sizeof(L2_search_reply), {
    pptr addr;
})

/* -------------------------- Util -------------------------- */
#define INIT_TSK 501
TASK(dpu_init_task, 501, true, sizeof(dpu_init_task), {
    int64_t dpu_id;
})
