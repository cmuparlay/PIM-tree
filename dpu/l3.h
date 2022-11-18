#pragma once

#include "common.h"
// #include "task_dpu.h"
#include "storage.h"
#include <barrier.h>
#include <alloc.h>
// #include <profiling.h>

// PROFILING_INIT(prof_newnode);
// PROFILING_INIT(prof_internal);
// PROFILING_INIT(prof_external);
// PROFILING_INIT(prof_finish);


// BARRIER_INIT(L3_barrier, NR_TASKLETS);

// extern volatile int host_barrier;
BARRIER_INIT(L3_barrier, NR_TASKLETS);
MUTEX_INIT(L3_lock);
// extern const mutex_id_t L3_lock;

// extern int NR_TASKLETS;

extern int64_t DPU_ID, dpu_epoch_number;
extern __mram_ptr uint8_t* send_task_start;
extern __mram_ptr ht_slot l3ht[]; 

static inline void L3_init(L3_init_task *tit) {
    IN_DPU_ASSERT(l3cnt == 8, "L3init: Wrong l3cnt\n");
    __mram_ptr void* maddr = reserve_space_L3(L3_node_size(tit->height));
    root = get_new_L3(LLONG_MIN, tit->height, tit->addr, maddr);
}

static inline int64_t L3_search(int64_t key, int record_height,
                                mL3ptr *rightmost, pptr *value) {
    mL3ptr tmp = root;
    int64_t ht = root->height - 1;
    while (ht >= 0) {
        pptr r = tmp->right[ht];
        if (r.id != INVALID_DPU_ID && ((mL3ptr)r.addr)->key <= key) {
            tmp = (mL3ptr)r.addr;  // go right
            continue;
        }
        if (rightmost != NULL && ht < record_height) {
            rightmost[ht] = tmp;
        }
        ht--;
    }
    if (value != NULL) {
        *value = tmp->down;
    }
    return tmp->key;
} 

// changed !!! ??? !!!

static inline void print_nodes(int length, mL3ptr *newnode, bool quit,
                               bool lock) {
    uint32_t tasklet_id = me();
    if (lock) mutex_lock(L3_lock);
    printf("*** %d ***\n", tasklet_id);
    for (int i = 0; i < length; i++) {
        // printf("*%d %lld %x\n", i, newnode[i]->key, (uint32_t)newnode[i]);
        printf("*%d %lld %x\n", i, newnode[i]->key, (uint32_t)newnode[i]);
        for (int ht = 0; ht < newnode[i]->height; ht++) {
            printf("%x %x\n", newnode[i]->left[ht].addr,
                   newnode[i]->right[ht].addr);
        }
    }
    if (lock) mutex_unlock(L3_lock);
    if (quit) {
        EXIT();
    }
}

const uint32_t OLD_NODES_DPU_ID = (uint32_t)-2;
static inline void L3_insert_parallel(int length, int l, int64_t *keys,
                                      int8_t *heights, pptr *addrs,
                                      uint32_t *newnode_size,
                                      int8_t *max_height_shared,
                                      mL3ptr *right_predecessor_shared,
                                      mL3ptr *right_newnode_shared) {
    uint32_t tasklet_id = me();
    int8_t max_height = 0;
    mL3ptr *newnode = mem_alloc(sizeof(mL3ptr) * length);

    barrier_wait(&L3_barrier);

    if (tasklet_id == 0) {
        for (int i = 0; i < NR_TASKLETS; i++) {
            newnode_size[i] = (uint32_t)reserve_space_L3(newnode_size[i]);
        }
    }

    barrier_wait(&L3_barrier);

    __mram_ptr void *maddr = (__mram_ptr void *)newnode_size[tasklet_id];

    for (int i = 0; i < length; i++) {
        newnode[i] = get_new_L3(keys[i], heights[i], addrs[i], maddr);
        maddr += L3_node_size(heights[i]);
        if (heights[i] > max_height) {
            max_height = heights[i];
        }
    }

    // mutex_lock(L3_lock);
    mL3ptr *predecessor = mem_alloc(sizeof(mL3ptr) * max_height);
    mL3ptr *left_predecessor = mem_alloc(sizeof(mL3ptr) * max_height);
    mL3ptr *left_newnode = mem_alloc(sizeof(mL3ptr) * max_height);
    // mutex_unlock(L3_lock);

    mL3ptr *right_predecessor =
        right_predecessor_shared + tasklet_id * MAX_L3_HEIGHT;
    mL3ptr *right_newnode = right_newnode_shared + tasklet_id * MAX_L3_HEIGHT;
    max_height_shared[tasklet_id] = max_height;

    IN_DPU_ASSERT(max_height <= root->height,
                  "L3insert: Wrong newnode height\n");

    if (length > 0) {
        int i = 0;
        L3_search(keys[i], heights[i], predecessor, NULL);
        for (int ht = 0; ht < heights[i]; ht++) {
            left_predecessor[ht] = right_predecessor[ht] = predecessor[ht];
            left_newnode[ht] = right_newnode[ht] = newnode[i];
        }
        max_height = heights[i];
        // print_nodes(heights[0], predecessor, true);
    }

    for (int i = 1; i < length; i++) {
        L3_search(keys[i], heights[i], predecessor, NULL);
        int minheight = (max_height < heights[i]) ? max_height : heights[i];
        for (int ht = 0; ht < minheight; ht++) {
            if (right_predecessor[ht] == predecessor[ht]) {
                right_newnode[ht]->right[ht] =
                    (pptr){.id = DPU_ID, .addr = (uint32_t)newnode[i]};
                newnode[i]->left[ht] =
                    (pptr){.id = DPU_ID, .addr = (uint32_t)right_newnode[ht]};
            } else {
                // IN_DPU_ASSERT(false, "L3 insert parallel : P1 ERROR");
                right_newnode[ht]->right[ht] =
                    (pptr){.id = OLD_NODES_DPU_ID,
                           .addr = right_predecessor[ht]->right[ht].addr};
                IN_DPU_ASSERT(
                    right_predecessor[ht]->right[ht].id != INVALID_DPU_ID,
                    "L3 insert parallel: Wrong rp->right");
                // mL3ptr rn = (mL3ptr)right_predecessor[ht]->right[ht].addr;
                // rn->left[ht] =
                //     (pptr){.id = DPU_ID, .addr =
                //     (uint32_t)right_newnode[ht]};

                newnode[i]->left[ht] = (pptr){
                    .id = OLD_NODES_DPU_ID, .addr = (uint32_t)predecessor[ht]};
                // predecessor[ht]->right[ht] =
                //     (pptr){.id = DPU_ID, .addr = (uint32_t)newnode[i]};
            }
        }
        for (int ht = 0; ht < heights[i]; ht++) {
            right_predecessor[ht] = predecessor[ht];
            right_newnode[ht] = newnode[i];
        }
        if (heights[i] > max_height) {
            for (int ht = max_height; ht < heights[i]; ht++) {
                left_predecessor[ht] = predecessor[ht];
                left_newnode[ht] = newnode[i];
            }
            max_height = heights[i];
        }
    }

    barrier_wait(&L3_barrier);

    int max_height_r = 0;
    for (int r = tasklet_id + 1; r < NR_TASKLETS; r++) {
        max_height_r = (max_height_shared[r] > max_height_r)
                           ? max_height_shared[r]
                           : max_height_r;
    }
    for (int ht = max_height_r; ht < max_height; ht++) {
        // right_newnode[ht]->right[ht] = right_predecessor[ht]->right[ht];
        if (right_predecessor[ht]->right[ht].id != INVALID_DPU_ID) {
            right_newnode[ht]->right[ht] =
                (pptr){.id = OLD_NODES_DPU_ID,
                       .addr = right_predecessor[ht]->right[ht].addr};
        } else {
            right_newnode[ht]->right[ht] = null_pptr;
        }
    }

    for (int l = (int)tasklet_id - 1, ht = 0; ht < max_height; ht++) {
        while (l >= 0 && ht >= max_height_shared[l]) {
            l--;
            IN_DPU_ASSERT(l >= -1 && l <= NR_TASKLETS, "L3 insert: l overflow");
        }
        if (l >= 0 && ht < max_height_shared[l]) {
            mL3ptr *right_predecessor_l =
                right_predecessor_shared + l * MAX_L3_HEIGHT;
            mL3ptr *right_newnode_l = right_newnode_shared + l * MAX_L3_HEIGHT;
            if (right_predecessor_l[ht] == left_predecessor[ht]) {
                right_newnode_l[ht]->right[ht] =
                    (pptr){.id = DPU_ID, .addr = (uint32_t)left_newnode[ht]};
                left_newnode[ht]->left[ht] =
                    (pptr){.id = DPU_ID, .addr = (uint32_t)right_newnode_l[ht]};
            } else {
                // if (right_predecessor_l[ht]->right[ht].id == INVALID_DPU_ID) {
                //     // printf("%x\n", right_predecessor_l[ht]->right[ht].addr);
                //     // printf("%d %d\n", tasklet_id, l);
                //     // printf("%x %x\n", right_predecessor_l[ht], left_predecessor[ht]);
                //     // print_pptr(right_predecessor_l[ht]->right[ht], "\n");
                //     // print_pptr(left_predecessor[ht]->right[ht], "\n");
                //     for (int i = 0; i < NR_TASKLETS; i ++) {
                //         printf("%d %d\n", i, max_height_shared[i]);
                //     }
                // }
                IN_DPU_ASSERT(
                    right_predecessor_l[ht]->right[ht].id != INVALID_DPU_ID,
                    "L3 insert parallel: build l_newnode <-> right_successor "
                    "id error");
                IN_DPU_ASSERT(
                    right_predecessor_l[ht]->right[ht].addr != INVALID_DPU_ADDR,
                    "L3 insert parallel: build l_newnode <-> right_successor "
                    "addr error");
                right_newnode_l[ht]->right[ht] =
                    (pptr){.id = OLD_NODES_DPU_ID,
                           .addr = right_predecessor_l[ht]->right[ht].addr};

                left_newnode[ht]->left[ht] =
                    (pptr){.id = OLD_NODES_DPU_ID,
                           .addr = (uint32_t)left_predecessor[ht]};
            }
        }
        if (l < 0) {
            left_newnode[ht]->left[ht] = (pptr){
                .id = OLD_NODES_DPU_ID, .addr = (uint32_t)left_predecessor[ht]};
        }
    }

    barrier_wait(&L3_barrier);

    for (int i = 0; i < length; i++) {
        for (int ht = 0; ht < heights[i]; ht++) {
            if (newnode[i]->left[ht].id == OLD_NODES_DPU_ID) {
                mL3ptr ln = (mL3ptr)newnode[i]->left[ht].addr;
                newnode[i]->left[ht] =
                    (pptr){.id = DPU_ID, .addr = (uint32_t)ln};
                ln->right[ht] =
                    (pptr){.id = DPU_ID, .addr = (uint32_t)newnode[i]};
            }
            if (newnode[i]->right[ht].id == OLD_NODES_DPU_ID) {
                mL3ptr rn = (mL3ptr)newnode[i]->right[ht].addr;
                newnode[i]->right[ht] =
                    (pptr){.id = DPU_ID, .addr = (uint32_t)rn};
                rn->left[ht] =
                    (pptr){.id = DPU_ID, .addr = (uint32_t)newnode[i]};
            }
        }
    }
}