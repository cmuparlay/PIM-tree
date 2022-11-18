/*
 * Copyright (c) 2014-2017 - uPmem
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * An example of checksum computation with multiple tasklets.
 *
 * Every tasklet processes specific areas of the MRAM, following the "rake"
 * strategy:
 *  - Tasklet number T is first processing block number TxN, where N is a
 *    constant block size
 *  - It then handles block number (TxN) + (NxM) where M is the number of
 *    scheduled tasklets
 *  - And so on...
 *
 * The host is in charge of computing the final checksum by adding all the
 * individual results.
 */
#include <stdint.h>
#include <stdio.h>
#include <defs.h>
#include <mram.h>
#include <alloc.h>
#include <perfcounter.h>
#include <barrier.h>
#include <string.h>
#include <seqread.h>
#include "driver.h"
#include "task.h"
#include "l3.h"
#include "l2.h"
// #include "node_dpu.h"
#include "task_framework_dpu.h"

/* -------------- Storage -------------- */

// BARRIER_INIT(init_barrier, NR_TASKLETS);
// BARRIER_INIT(end_barrier, NR_TASKLETS);

// __host volatile int host_barrier;

// __host int num_tasklets;
// __host uint64_t task_type;

// DPU ID
__host int64_t DPU_ID = -1;

__host int64_t dpu_epoch_number;
__host int64_t dpu_task_type;
__host int64_t dpu_task_count;

#define MRAM_BUFFER_SIZE (128)
void *bufferA_shared, *bufferB_shared;
int8_t *max_height_shared;
uint32_t *newnode_size;

// Node Buffers & Hash Tables
// L3
__mram_noinit ht_slot l3ht[LX_HASHTABLE_SIZE]; // must be 8 bytes aligned
__host int l3htcnt = 0;
__mram_noinit uint8_t l3buffer[L3_BUFFER_SIZE];
__host int l3cnt = 8;

// L2
__mram_noinit ht_slot l2ht[LX_HASHTABLE_SIZE]; // must be 8 bytes aligned
__host int l2htcnt = 0;
__mram_noinit uint8_t l2buffer[L2_BUFFER_SIZE];
__host int l2cnt = 8;

__host mL3ptr root;

static inline void dpu_init(dpu_init_task *it) {
    DPU_ID = it->dpu_id;
    storage_init();
}

void execute(int l, int r) {
    uint32_t tid = me();
    int length = r - l;
    switch(recv_block_task_type) {
        case INIT_TSK: {
            init_block_with_type(dpu_init_task, empty_task_reply);
            if (tid == 0) {
                init_task_reader(0);
                dpu_init_task* it = (dpu_init_task*)get_task_cached(0);
                dpu_init(it);
            }
            break;
        }

        case L3_INIT_TSK: {
            init_block_with_type(L3_init_task, empty_task_reply);
            if (tid == 0) {
                init_task_reader(0);
                L3_init_task* tit = (L3_init_task*)get_task_cached(0);
                L3_init(tit);
            }
            break;
        }

        case L3_INSERT_TSK: {
            init_block_with_type(L3_insert_task, empty_task_reply);

            int64_t* keys = mem_alloc(sizeof(int64_t) * length);
            pptr* addrs = mem_alloc(sizeof(pptr) * length);
            int8_t* heights = mem_alloc(sizeof(int8_t) * length);

            init_task_reader(l);

            newnode_size[tid] = 0;
            for (int i = 0; i < length; i++) {
                L3_insert_task* tit = (L3_insert_task*)get_task_cached(l + i);
                keys[i] = tit->key;
                addrs[i] = tit->addr;
                heights[i] = tit->height;
                newnode_size[tid] += L3_node_size(heights[i]);
                IN_DPU_ASSERT(heights[i] > 0 && heights[i] < MAX_L3_HEIGHT,
                              "execute: invalid height\n");
            }

            mL3ptr* right_predecessor_shared = bufferA_shared;
            mL3ptr* right_newnode_shared = bufferB_shared;
            L3_insert_parallel(length, l, keys, heights, addrs, newnode_size,
                               max_height_shared, right_predecessor_shared,
                               right_newnode_shared);
            break;
        }

        case L3_SEARCH_TSK: {
            init_block_with_type(L3_search_task, L3_search_reply);
            init_task_reader(l);
            for (int i = l; i < r; i++) {
                L3_search_task* tst = (L3_search_task*)get_task_cached(i);
                pptr val = null_pptr;
                L3_search(tst->key, 0, NULL, &val);
                L3_search_reply tsr =
                    (L3_search_reply){.addr = val};
                push_fixed_reply(i, &tsr);
            }
            break;
        }

        case L2_INIT_TSK: {
            init_block_with_type(L2_init_task, L2_init_reply);
            if (tid == 0) {
                init_task_reader(0);
                L2_init_task *sit = (L2_init_task*)get_task_cached(0);
                mL2ptr nn = L2_init(sit);
                L2_init_reply sir = (L2_init_reply){.addr = PPTR(DPU_ID, nn)};
                push_fixed_reply(0, &sir);
            }
            break;
        }

        case L2_INSERT_TSK: {
            init_block_with_type(L2_insert_task, L2_insert_reply);

            int64_t* keys = mem_alloc(sizeof(int64_t) * length);
            pptr* addrs = mem_alloc(sizeof(pptr) * length);
            int8_t* heights = mem_alloc(sizeof(int8_t) * length);

            init_task_reader(l);

            newnode_size[tid] = 0;
            for (int i = 0; i < length; i++) {
                if (i >= length) break;
                L2_insert_task* sit = (L2_insert_task*)get_task_cached(l + i);
                keys[i] = sit->key;
                addrs[i] = sit->addr;
                heights[i] = sit->height;
                newnode_size[tid] += L2_node_size(
                    (heights[i] > LOWER_PART_HEIGHT) ? LOWER_PART_HEIGHT
                                                     : heights[i]);
                IN_DPU_ASSERT(heights[i] > 0, "execute: invalid height\n");
            }
            L2_insert_parallel(l, length, keys, heights, addrs, newnode_size);
            break;
        }

        case L2_BUILD_LR_TSK: {
            init_block_with_type(L2_build_lr_task, empty_task_reply);
            init_task_reader(l);
            for (int i = l; i < r; i++) {
                L2_build_lr_task* sblrt = (L2_build_lr_task*)get_task_cached(i);
                L2_build_lr(sblrt->height, sblrt->addr, sblrt->val, sblrt->chk);
            }
            break;
        }

        case L2_GET_NODE_TSK: {
            init_block_with_type(L2_get_node_task, L2_get_node_reply);
            init_task_reader(l);
            for (int i = l; i < r; i++) {
                L2_get_node_task* sgnt = (L2_get_node_task*)get_task_cached(i);
                L2_get_node_reply sgnr = L2_get_node(sgnt->addr, sgnt->height);
                push_fixed_reply(i, &sgnr);
            }
            break;
        }

        case L2_SEARCH_TSK: {
            init_block_with_type(L2_search_task, L2_search_reply);
            init_task_reader(l);
            for (int i = l; i < r; i++) {
                L2_search_task* sst = (L2_search_task*)get_task_cached(i);
                IN_DPU_ASSERT(valid_pptr(sst->addr), "sst! invaddr\n");
                IN_DPU_ASSERT(sst->height >= 0, "sst! invheight\n");
                L2_search_reply ssr = (L2_search_reply){.addr = L2_search(sst->key, sst->addr, sst->height)};
                push_fixed_reply(i, &ssr);
            }
            break;
        }

        default: {
            IN_DPU_ASSERT(false, "Wrong Task Type\n");
            break;
        }
    }
    finish_reply(recv_block_task_cnt, tid);
}

void init() {
    bufferA_shared = mem_alloc(sizeof(mL3ptr) * NR_TASKLETS * MAX_L3_HEIGHT);
    bufferB_shared = mem_alloc(sizeof(mL3ptr) * NR_TASKLETS * MAX_L3_HEIGHT);
    max_height_shared = mem_alloc(sizeof(int8_t) * NR_TASKLETS);
    newnode_size = mem_alloc(sizeof(uint32_t) * NR_TASKLETS);
    for (int i = 0; i < NR_TASKLETS; i++) {
        max_height_shared[i] = 0;
    }
    if (DPU_ID == 0 && me() == 0) {
        printf("l3cnt=%d l2cnt=%d l3htcnt=%d l2htcnt=%d\n", l3cnt, l2cnt, l3htcnt, l2htcnt);
    }
}

void wram_heap_load() {

}

void wram_heap_save() {

}

int main() { run(); }