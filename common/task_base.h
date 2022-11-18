#pragma once

/* -------------------------- Task Type -------------------------- */

#define EMPTY 0

#include <stdint.h>
#include <assert.h>

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

TASK(L3_insert_task, 1, true, sizeof(L3_insert_task), {
    int64_t key;
    int64_t value;
    int64_t height;
})

TASK(L3_init_task, 2, true, sizeof(L3_init_task), {
    int64_t key;
    int64_t value;
    int64_t height;
})

TASK(L3_remove_task, 3, true, sizeof(L3_remove_task), {
    int64_t key;
})

TASK(L3_search_task, 4, true, sizeof(L3_search_task), {
    int64_t key;
})

TASK(L3_search_reply, 5, true, sizeof(L3_search_reply), {
    int64_t key;
    int64_t value;
})

TASK(L3_get_task, 6, true, sizeof(L3_get_task), {
    int64_t key;
})

TASK(L3_get_reply, 7, true, sizeof(L3_get_reply), {
    int64_t valid;
    int64_t value;
})

TASK(L3_update_task, 8, true, sizeof(L3_update_task), {
    int64_t key;
    int64_t value;
})

TASK(L3_get_min_task, 9, true, sizeof(L3_get_min_task), {
    int64_t key;
})

TASK(L3_get_min_reply, 10, true, sizeof(L3_get_min_reply), {
    int64_t key;
})

TASK(dpu_init_task, 501, true, sizeof(dpu_init_task), {
    int64_t dpu_id;
})

// Range Scan
TASK(L3_scan_task, 20, true, sizeof(L3_scan_task), {
    int64_t lkey;
    int64_t rkey;
})

TASK(L3_scan_reply, 21, false, sizeof(L3_scan_reply), {
    int64_t length;
    int64_t vals[];
})
