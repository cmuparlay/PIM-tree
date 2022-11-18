#include <stdint.h>
#include <stdio.h>
#include <defs.h>
#include <mram.h>
#include <alloc.h>
#include <perfcounter.h>
#include <barrier.h>
#include <string.h>
#include <seqread.h>
#include "common.h"
#include "task_dpu.h"
#include "l3.h"

/* -------------- Storage -------------- */

BARRIER_INIT(init_barrier, NR_TASKLETS);
BARRIER_INIT(end_barrier, NR_TASKLETS);

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
__host mL3ptr root;
__host __mram_ptr uint64_t* receive_buffer_offset;

// send
__host __mram_ptr uint8_t* receive_buffer = DPU_MRAM_HEAP_POINTER;
__host __mram_ptr uint8_t* receive_task_start = DPU_MRAM_HEAP_POINTER + sizeof(int64_t) * 3;

// receive
__host __mram_ptr uint8_t* send_buffer = DPU_MRAM_HEAP_POINTER + DPU_SEND_BUFFER_OFFSET;
__host __mram_ptr uint8_t* send_task_start = DPU_MRAM_HEAP_POINTER + DPU_SEND_BUFFER_OFFSET + sizeof(int64_t);

// fixed length
__host __mram_ptr int64_t* send_task_count = DPU_MRAM_HEAP_POINTER + DPU_SEND_BUFFER_OFFSET;
__host __mram_ptr int64_t* send_size = DPU_MRAM_HEAP_POINTER + DPU_SEND_BUFFER_OFFSET + sizeof(int64_t);


#define MRAM_TASK_BUFFER

static inline void init(init_task *it) {
    DPU_ID = it->id;
    storage_init();
}

void execute(int l, int r) {
    IN_DPU_ASSERT(dpu_task_type == INIT_TSK || DPU_ID != -1, "execute: id not initialized");
    uint32_t tasklet_id = me();
    int length = r - l;
    
    __mram_ptr int64_t* buffer_type = (__mram_ptr int64_t*)send_buffer;
    *buffer_type = BUFFER_FIXED_LENGTH; // default

    switch (dpu_task_type) {
        case INIT_TSK: {
            if (tasklet_id == 0) {
                init_task it;
                mram_read(receive_task_start, &it, sizeof(init_task));
                init(&it);
            }
            break;
        }

        case L3_INIT_TSK: {
            if (tasklet_id == 0) {
                L3_init_task tit;
                mram_read(receive_task_start, &tit, sizeof(L3_init_task));
                L3_init(&tit);
            }
            break;
        }

        case L3_INSERT_TSK: {
            __mram_ptr L3_insert_task* mram_tit =
                (__mram_ptr L3_insert_task*)receive_task_start;
            mram_tit += l;

            newnode_size[tasklet_id] = 0;
            for (int i = 0; i < length; i++) {
                int height = mram_tit[i].height;
                newnode_size[tasklet_id] += L3_node_size(height);
                IN_DPU_ASSERT(height > 0 && height < MAX_L3_HEIGHT,
                              "execute: invalid height\n");
            }

            mL3ptr* right_predecessor_shared = bufferA_shared;
            mL3ptr* right_newnode_shared = bufferB_shared;
            L3_insert_parallel(length, l, mram_tit, newnode_size, max_height_shared,
                               right_predecessor_shared, right_newnode_shared);
            break;
        }

        case L3_REMOVE_TSK: {
            __mram_ptr L3_remove_task* mram_trt =
                (__mram_ptr L3_remove_task*)receive_task_start;
            mram_trt += l;

            mL3ptr* left_node_shared = bufferA_shared;
            L3_remove_parallel(length, l, mram_trt, max_height_shared,
                               left_node_shared);
            break;
        }

        case L3_SEARCH_TSK: {
            __mram_ptr L3_search_task* tst =
                (__mram_ptr L3_search_task*)receive_task_start;
            tst += l;

            __mram_ptr L3_search_reply* dst =
                (__mram_ptr L3_search_reply*)send_task_start;

            for (int i = 0; i < length; i++) {
                int64_t value;
                int64_t key = L3_search(tst[i].key, 0, NULL, &value);
                L3_search_reply tsr = (L3_search_reply){.result_key = value};
                mram_write(&tsr, &dst[l + i], sizeof(int64_t));
            }
            break;
        }

        case L3_GET_TSK: {
            __mram_ptr L3_get_task* tgt =
                (__mram_ptr L3_get_task*)receive_task_start;
            for (int i = l; i < r; i++) {
                L3_get(tgt[i].key, i);
            }
            break;
        }

        case L3_SANCHECK_TSK: {
            if (tasklet_id == 0) {
                L3_sancheck();
            }
            break;
        }

        default: {
            IN_DPU_ASSERT(false, "Wrong Task Type\n");
            break;
        }
    }
}

void garbage_func();

int main() {
    uint32_t tasklet_id = me();

    if (tasklet_id == 0) {
        mem_reset();

        bufferA_shared =
            mem_alloc(sizeof(mL3ptr) * NR_TASKLETS * MAX_L3_HEIGHT);
        bufferB_shared =
            mem_alloc(sizeof(mL3ptr) * NR_TASKLETS * MAX_L3_HEIGHT);
        max_height_shared = mem_alloc(sizeof(int8_t) * NR_TASKLETS);
        newnode_size = mem_alloc(sizeof(uint32_t) * NR_TASKLETS);
        for (int i = 0; i < NR_TASKLETS; i++) {
            max_height_shared[i] = 0;
        }

        mram_read(receive_buffer, &dpu_epoch_number, sizeof(int64_t));
        mram_read(receive_buffer + sizeof(int64_t), &dpu_task_type,
                  sizeof(int64_t));
        mram_read(receive_buffer + sizeof(int64_t) * 2, &dpu_task_count,
                  sizeof(int64_t));
    }

    barrier_wait(&init_barrier);

    if (dpu_task_count == 0) {
        return 0;
    }

    uint32_t lft = dpu_task_count * tasklet_id / NR_TASKLETS;
    uint32_t rt = dpu_task_count * (tasklet_id + 1) / NR_TASKLETS;

    execute(lft, rt);

    if (tasklet_id == 0) {
        printf("epoch=%lld l3cnt=%d l3htcnt=%d\n", dpu_epoch_number, l3cnt,
               l3htcnt);
    }
    return 0;
}
