#pragma once

#include "common.h"
// #include "task_dpu.h"
#include "storage.h"
#include <barrier.h>
#include <alloc.h>


// extern volatile int host_barrier;
BARRIER_INIT(L2_barrier, NR_TASKLETS);
BARRIER_INIT(L2_barrier1, NR_TASKLETS);
BARRIER_INIT(L2_barrier2, NR_TASKLETS);
MUTEX_INIT(L2_lock);

// extern int NR_TASKLETS;

extern int64_t DPU_ID;

extern __mram_ptr ht_slot l2ht[]; 


static inline mL2ptr L2_init(L2_init_task *sit) {
    IN_DPU_ASSERT(l2cnt == 8, "L2init: Wrong l2cnt\n");
    __mram_ptr void* maddr = reserve_space_L2(L2_node_size(sit->height));
    mL2ptr ret = get_new_L2(LLONG_MIN, sit->height, sit->addr, maddr);
    return ret;
}

static inline L2_get_node_reply L2_get_node(pptr addr, int height) {
    IN_DPU_ASSERT(addr.id == DPU_ID, "L2 get node: wrong addr.id");
    mL2ptr nn = (mL2ptr)addr.addr;
    // if (!(height >= -1 && height < nn->height)) {
    //     printf("** %lld %d %lld %d-%x\n", nn->key, height, nn->height, addr.id, addr.addr);
    // }
    IN_DPU_ASSERT_EXEC(height >= -1 && height < nn->height, {
        printf("L2 get node: wrong height=%d nnh=%d\n", height, nn->height);
    });
    L2_get_node_reply sgnr;
    if (height == -1) {
        sgnr = (L2_get_node_reply){.chk = nn->key, .right = nn->down};
    } else {
        sgnr = (L2_get_node_reply){.chk = nn->chk[height], .right = nn->right[height]};
    }
    return sgnr;
}

static inline pptr L2_search(int64_t key, pptr addr, int height) {
    IN_DPU_ASSERT(addr.id == DPU_ID, "L2 get node: wrong addr.id");
    mL2ptr nn = (mL2ptr)addr.addr;
    pptr right = nn->right[height];
    if (valid_pptr(right) && key >= nn->chk[height]) {
        return right;
    } else {
        return addr;
    }
}

static inline void L2_print_node(int i, mL2ptr node) {
    printf("*%d %lld %x\n", i, node->key, (uint32_t)node);
    int height = (node->height > LOWER_PART_HEIGHT) ? LOWER_PART_HEIGHT : node->height;
    for (int ht = 0; ht < height; ht++) {
        printf("%lld %d-%x %d-%x\n", node->chk[ht], node->left[ht].id, node->left[ht].addr,
                node->right[ht].id, node->right[ht].addr);
    }
}
static inline void L2_print_nodes(int length, mL2ptr *newnode, bool quit, bool lock) {
    uint32_t tasklet_id = me();
    if (lock) mutex_lock(L2_lock);
    printf("*** %d ***\n", tasklet_id);
    for (int i = 0; i < length; i++) {
        // printf("*%d %lld %x\n", i, newnode[i]->key, (uint32_t)newnode[i]);
        L2_print_node(i, newnode[i]);
        // printf("*%d %lld %x\n", i, newnode[i]->key, (uint32_t)newnode[i]);
        // for (int ht = 0; ht < newnode[i]->height; ht++) {
        //     printf("%lld %x %x\n", newnode[i]->chk[ht], newnode[i]->left[ht].addr,
        //            newnode[i]->right[ht].addr);
        // }
    }
    if (lock) mutex_unlock(L2_lock);
    if (quit) {
        EXIT();
    }
}

static inline void L2_insert_parallel(int l, int length, int64_t *keys,
                                      int8_t *heights, pptr *addrs,
                                      uint32_t *newnode_size) {
    uint32_t tasklet_id = me();
    mL2ptr *newnode = mem_alloc(sizeof(mL2ptr) * length);

    barrier_wait(&L2_barrier1);

    if (tasklet_id == 0) {
        for (int i = 0; i < NR_TASKLETS; i++) {
            newnode_size[i] = (uint32_t)reserve_space_L2(newnode_size[i]);
        }
    }

    barrier_wait(&L2_barrier2);

    __mram_ptr void* maddr = (__mram_ptr void*) newnode_size[tasklet_id];
    __mram_ptr void* rmaddr;
    if (tasklet_id + 1 < NR_TASKLETS) {
        rmaddr = (__mram_ptr void*) newnode_size[tasklet_id + 1];
    }

    barrier_wait(&L2_barrier1); // !!!!!

    for (int i = 0; i < length; i++) {
        newnode[i] = get_new_L2(keys[i], heights[i], addrs[i], maddr);
        int l2height = (heights[i] > LOWER_PART_HEIGHT) ? LOWER_PART_HEIGHT : heights[i];
        maddr += L2_node_size(l2height);
    }
    if (tasklet_id + 1 < NR_TASKLETS) {
        IN_DPU_ASSERT(maddr == rmaddr, "L2 insert parallel: wrong maddr");
    } else {
        IN_DPU_ASSERT(maddr == (l2buffer + l2cnt), "L2 insert parallel: wrong maddr");
    }

    for (int i = 0; i < length; i ++) {
        L2_insert_reply sir = (L2_insert_reply){
            .addr = (pptr){.id = DPU_ID, .addr = (uint32_t)newnode[i]}};
        push_fixed_reply(l + i, &sir);
    }
}

static inline void L2_build_lr(int64_t height, pptr addr, pptr val, int64_t chk) {
    IN_DPU_ASSERT(addr.id == DPU_ID, "L2 build lr: wrong id");
    mL2ptr nn = (mL2ptr)addr.addr;
    if (height >= 0) {
        IN_DPU_ASSERT(height >= 0 && height < nn->height, "L2 build lr: wrong height");
        nn->right[height] = val;
        nn->chk[height] = chk;
    } else {
        height = -1 - height;
        IN_DPU_ASSERT(height >= 0 && height < nn->height, "L2 build lr: wrong height");
        nn->left[height] = val;
        IN_DPU_ASSERT(chk == -1, "L2 build lr: wrong chk");
    }
}