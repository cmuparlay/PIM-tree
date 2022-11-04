#pragma once

// OBSOLETE
// DO NOT USE

#include <barrier.h>
#include <alloc.h>
#include "node_dpu.h"
#include "task_framework_dpu.h"
#include "common.h"

// Range Scan
#include "dpu_buffer.h"

BARRIER_INIT(L3_barrier, NR_TASKLETS);
MUTEX_INIT(L3_lock);

extern int64_t DPU_ID;

// L3
mpuint8_t l3buffer;
__host uint32_t l3cnt = 8;
__host mL3ptr root;

static inline uint32_t L3_node_size(int height) {
    return sizeof(L3node) + sizeof(pptr) * height * 2;
}

static inline L3node* init_L3(int64_t key, int height, uint8_t* buffer,
                              __mram_ptr void* maddr) {
    L3node* nn = (L3node*)buffer;
    nn->key = key;
    nn->height = height;
    nn->left = (mppptr)(maddr + sizeof(L3node));
    nn->right = (mppptr)(maddr + sizeof(L3node) + sizeof(pptr) * height);
    // for (int i = 0; i < sizeof(pptr) * height * 2; i ++) {
    //     buffer[sizeof(L3node) + i] = (uint8_t)-1;
    // }
    memset(buffer + sizeof(L3node), -1, sizeof(pptr) * height * 2);
    return nn;
}

static inline __mram_ptr void* reserve_space_L3(uint32_t size) {
    __mram_ptr void* ret = l3buffer + l3cnt;
    l3cnt += size;
    // SPACE_IN_DPU_ASSERT(l3cnt < L3_BUFFER_SIZE, "rs3! of\n");
    return ret;
}

static inline mL3ptr get_new_L3(int64_t key, int height,
                                __mram_ptr void* maddr) {
    int size = L3_node_size(height);
    // __mram_ptr void* maddr = reserve_space_L3(size);
    uint8_t buffer[sizeof(L3node) + sizeof(pptr) * 2 * MAX_L3_HEIGHT];
    L3node* nn = init_L3(key, height, buffer, maddr);
    m_write((void*)nn, maddr, size);
    // mram_write((void*)nn, maddr, size);
    // ht_insert(l3ht, &l3htcnt, hash_to_addr(key, 0, LX_HASHTABLE_SIZE),
    //           (uint32_t)maddr);
    return (mL3ptr)maddr;
}

static inline void L3_init(L3_init_task *tit) {
    L3_IN_DPU_ASSERT(l3cnt == 8, "L3init: Wrong l3cnt\n");
    __mram_ptr void* maddr = reserve_space_L3(L3_node_size(tit->height));
    root = get_new_L3(INT64_MIN, tit->height, maddr);
    root->down = tit->down;
    // L3_init_reply tir = (L3_init_reply){.addr = (pptr){.id = DPU_ID, .addr = (uint32_t)root}};
    // print_pptr(tir.addr, "\n");
    // push_fixed_reply(0, &tir);
}

// static inline void L3_build_down(pptr addr, pptr down) {
//     L3_IN_DPU_ASSERT(addr.id == 0, "L3 build down: wrong addr.id\n");
//     // DEBUG({print_pptr(addr, " addr\n");print_pptr(down, " down\n");});
//     mL3ptr nn = (mL3ptr)addr.addr;
//     nn->down = down;
// }

static inline int64_t L3_search(int64_t key, int i, int record_height,
                                mL3ptr *rightmost) {
    mL3ptr tmp = root;
    int64_t ht = root->height - 1;
    while (ht >= 0) {
        #ifdef DPU_ENERGY
        op_count += 2;
        db_size_count += S64(2);
        #endif
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
    // L3_IN_DPU_ASSERT(rightmost != NULL, "L3 search: rightmost error");
    if (rightmost == NULL) {  // pure search task
        L3_search_reply tsr = (L3_search_reply){.addr = tmp->down};
        push_fixed_reply(i, &tsr);
    }
    return tmp->key;
}

static void L3_insert_parallel(int length, int l, int64_t *keys,
                                      int8_t *heights, pptr* down,
                                      uint32_t *newnode_size,
                                      int8_t *max_height_shared,
                                      mL3ptr *right_predecessor_shared,
                                      mL3ptr *right_newnode_shared) {
    const uint32_t OLD_NODES_DPU_ID = (uint32_t)-2;
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
        newnode[i] = get_new_L3(keys[i], heights[i], maddr);
        newnode[i]->down = down[i];
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

    L3_IN_DPU_ASSERT(max_height <= root->height,
                  "L3insert: Wrong newnode height\n");

    if (length > 0) {
        int i = 0;
        L3_search(keys[i], 0, heights[i], predecessor);
        for (int ht = 0; ht < heights[i]; ht++) {
            left_predecessor[ht] = right_predecessor[ht] = predecessor[ht];
            left_newnode[ht] = right_newnode[ht] = newnode[i];
        }
        max_height = heights[i];
        // print_nodes(heights[0], predecessor, true);
    }

    for (int i = 1; i < length; i++) {
        L3_search(keys[i], 0, heights[i], predecessor);
        int minheight = (max_height < heights[i]) ? max_height : heights[i];
        for (int ht = 0; ht < minheight; ht++) {
            if (right_predecessor[ht] == predecessor[ht]) {
                right_newnode[ht]->right[ht] =
                    (pptr){.id = DPU_ID, .addr = (uint32_t)newnode[i]};
                newnode[i]->left[ht] =
                    (pptr){.id = DPU_ID, .addr = (uint32_t)right_newnode[ht]};
            } else {
                // L3_IN_DPU_ASSERT(false, "L3 insert parallel : P1 ERROR");
                right_newnode[ht]->right[ht] =
                    (pptr){.id = OLD_NODES_DPU_ID,
                           .addr = right_predecessor[ht]->right[ht].addr};
                L3_IN_DPU_ASSERT(
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
            L3_IN_DPU_ASSERT(l >= -1 && l <= NR_TASKLETS, "L3 insert: l overflow");
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
                L3_IN_DPU_ASSERT(
                    right_predecessor_l[ht]->right[ht].id != INVALID_DPU_ID,
                    "L3 insert parallel: build l_newnode <-> right_successor "
                    "id error");
                L3_IN_DPU_ASSERT(
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

    // L3_insert_reply *tir = mem_alloc(sizeof(L3_insert_reply) * length);
    // for (int i = 0; i < length; i++) {
    //     tir[i] = (L3_insert_reply){
    //         .addr = (pptr){.id = DPU_ID, .addr = (uint32_t)newnode[i]}};
    // }
    // for (int i = 0; i < length; i ++) {
    //     L3_insert_reply tir = (L3_insert_reply){
    //         .addr = (pptr){.id = DPU_ID, .addr = (uint32_t)newnode[i]}};
    //     push_fixed_reply(l + i, &tir);
    // }
    // __mram_ptr L3_insert_reply *dst =
    //     (__mram_ptr L3_insert_reply *)send_content_start;
    // mram_write(tir, &dst[l], sizeof(L3_insert_reply) * length);
}

static void L3_remove_parallel(int length, mL3ptr *nodes,
                                      int8_t *max_height_shared,
                                      mL3ptr *left_node_shared) {
    uint32_t tasklet_id = me();
    // mL3ptr *nodes = mem_alloc(sizeof(mL3ptr) * length);
    int8_t *heights = mem_alloc(sizeof(int8_t) * length);
    int8_t max_height = 0;
    for (int i = 0; i < length; i ++) {
        heights[i] = nodes[i]->height;
    }
    mL3ptr *left_node = left_node_shared + tasklet_id * MAX_L3_HEIGHT;

    max_height = 0;
    for (int i = 0; i < length; i++) {
        int min_height = (heights[i] < max_height) ? heights[i] : max_height;
        for (int ht = 0; ht < min_height; ht++) {
            mL3ptr ln = (mL3ptr)(nodes[i]->left[ht].addr);
            ln->right[ht] = nodes[i]->right[ht];
            if (nodes[i]->right[ht].id != INVALID_DPU_ID) {
                mL3ptr rn = (mL3ptr)(nodes[i]->right[ht].addr);
                rn->left[ht] = nodes[i]->left[ht];
            }
            nodes[i]->left[ht] = nodes[i]->right[ht] = null_pptr;
        }
        if (heights[i] > max_height) {
            for (int ht = max_height; ht < heights[i]; ht++) {
                left_node[ht] = nodes[i];
            }
            max_height = heights[i];
        }
    }

    max_height_shared[tasklet_id] = max_height;

    barrier_wait(&L3_barrier);

    for (int l = (int)tasklet_id - 1, ht = 0; ht < max_height; ht++) {
        while (l >= 0 && ht >= max_height_shared[l]) {
            l--;
        }
        mL3ptr *left_node_l = left_node_shared + l * MAX_L3_HEIGHT;
        if (l < 0 || ((mL3ptr)left_node[ht]->left[ht].addr !=
                      left_node_l[ht])) {  // left most node in the level
            int r = tasklet_id + 1;
            mL3ptr rn = left_node[ht];
            for (; r < NR_TASKLETS; r++) {
                if (max_height_shared[r] <= ht) {
                    continue;
                }
                mL3ptr *left_node_r = left_node_shared + r * MAX_L3_HEIGHT;
                if (rn->right[ht].id == INVALID_DPU_ID ||
                    (mL3ptr)rn->right[ht].addr != left_node_r[ht]) {
                    break;
                }
                rn = left_node_r[ht];
            }
            L3_IN_DPU_ASSERT(left_node[ht]->left[ht].id == DPU_ID,
                          "L3 remove parallel: wrong leftid\n");
            mL3ptr ln = (mL3ptr)(left_node[ht]->left[ht].addr);
            ln->right[ht] = rn->right[ht];
            if (rn->right[ht].id != INVALID_DPU_ID) {
                rn = (mL3ptr)rn->right[ht].addr;
                rn->left[ht] = left_node[ht]->left[ht];
            }
        } else {  // not the left most node
            L3_IN_DPU_ASSERT(
                ((mL3ptr)left_node_l[ht]->right[ht].addr == left_node[ht]),
                "L3 remove parallel: wrong skip");
            // do nothing
        }
    }
}

#ifdef DPU_SCAN
static inline int L3_scan_search(int64_t begin, int64_t end, varlen_buffer_dpu *addrbuf) {
    mL3ptr tmp = root;
    int64_t ht = root->height - 1;
    while (ht >= 0) {
        pptr r = tmp->right[ht];
        if (r.id != INVALID_DPU_ID && ((mL3ptr)r.addr)->key <= begin) {
            tmp = (mL3ptr)r.addr;  // go right
            continue;
        }
        ht--;
    }
    
    varlen_buffer_reset_dpu(addrbuf);
    pptr l3_down = tmp->down;
    mL3ptr tmp2 = tmp;
    int num = 1;
    ht = 1;
    varlen_buffer_push_dpu(addrbuf, PPTR_TO_I64(l3_down));
    
    while(ht == 1) {
        pptr r = tmp2->right[0];
        if(r.id != INVALID_DPU_ID && ((mL3ptr)r.addr)->key <= end
        // && addrbuf->len <= 1024
        ) {
            tmp2 = (mL3ptr)r.addr;  // go right
            num++;
            l3_down = tmp2->down;
            varlen_buffer_push_dpu(addrbuf, PPTR_TO_I64(l3_down));
            continue;
        }
        ht = 0;
    }
    return num;
}
#endif
