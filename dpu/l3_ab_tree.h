#pragma once

#include <barrier.h>
#include <alloc.h>
#include "node_dpu.h"
#include "task_framework_dpu.h"
#include "common.h"
#include "gc.h"

// Range Scan
#include "dpu_buffer.h"

BARRIER_INIT(L3_barrier1, NR_TASKLETS);
BARRIER_INIT(L3_barrier2, NR_TASKLETS);
BARRIER_INIT(L3_barrier3, NR_TASKLETS);

MUTEX_INIT(L3_lock);

extern int64_t DPU_ID;

// L3
mL3Bptr l3bbuffer;
__host uint32_t l3bcnt = 1;
mL3Bptr l3bbuffer_start, l3bbuffer_end;
__host mL3Bptr root;


static inline bool in_l3bbuffer(mL3Bptr addr) {
    // IN_DPU_ASSERT(
    //     bbuffer_start == bbuffer + 1 &&
    //         bbuffer_end == (bbuffer_start + (B_BUFFER_SIZE / sizeof(Bnode)) -
    //         1),
    //     "ib! inv\n");
    return (addr >= l3bbuffer_start) && (addr < l3bbuffer_end);
}

static inline void l3b_node_init(L3Bnode *bn, int ht, pptr up, pptr right) {
    bn->height = ht;
    bn->up = up;
    bn->right = right;
    bn->size = 0;
    for (int i = 0; i < DB_SIZE; i++) {
        bn->keys[i] = INT64_MIN;
        bn->addrs[i] = null_pptr;
    }
}

void l3b_node_fill(mL3Bptr nn, L3Bnode *bn, int size, int64_t *keys,
                        pptr *addrs) {
    m_read(nn, bn, sizeof(L3Bnode));
    bn->size = size;
    for (int i = 0; i < size; i++) {
        bn->keys[i] = keys[i];
        bn->addrs[i] = addrs[i];
    }
    for (int i = size; i < DB_SIZE; i++) {
        bn->keys[i] = INT64_MIN;
        bn->addrs[i] = null_pptr;
    }
    m_write(bn, nn, sizeof(L3Bnode));
    if (bn->height > 0) {
        for (int i = 0; i < size; i ++) {
            mL3Bptr ch = pptr_to_ml3bptr(bn->addrs[i]);
            ch->up = ml3bptr_to_pptr(nn);
        }
    }
}

const int l3bcnt_limit = L3_BUFFER_SIZE / sizeof(L3Bnode);
const int HF_DB_SIZE = DB_SIZE >> 1;

static inline mL3Bptr alloc_l3bn() {
    mutex_lock(L3_lock);
    pptr recycle = alloc_node(&free_list_l3bnode, 1);
    mL3Bptr nn;
    if (recycle.id == 0) {
        nn = l3bbuffer + l3bcnt++;
    } else {
        nn = (mL3Bptr)recycle.addr;
    }
    mutex_unlock(L3_lock);
    IN_DPU_ASSERT_EXEC(l3bcnt < l3bcnt_limit, { printf("l3bcnt! of\n"); });
    return nn;
}

void L3_init(L3_init_task* tit) {
    IN_DPU_ASSERT(l3bcnt == 1, "l3i!\n");
    mL3Bptr nn = alloc_l3bn();
    L3Bnode bn;
    l3b_node_init(&bn, 0, null_pptr, null_pptr);
    bn.size = 1;
    bn.keys[0] = INT64_MIN;
    bn.addrs[0] = tit->down;
    m_write(&bn, nn, sizeof(L3Bnode));
    root = nn;
}

static inline int64_t l3b_search(int64_t key, mL3Bptr *addr, pptr *value) {
    mL3Bptr tmp = root;
    L3Bnode bn;
    while (true) {
        // mram_read(tmp, &bn, sizeof(L3Bnode));
        m_read(tmp, &bn, sizeof(L3Bnode));
        int64_t pred = INT64_MIN;
        pptr nxt_addr = null_pptr;
        for (int i = 0; i < bn.size; i++) {
            if (bn.keys[i] <= key) {
                pred = bn.keys[i];
                nxt_addr = bn.addrs[i];
            }
        }
        // IN_DPU_ASSERT((valid_pptr(nxt_addr) || bn.height == 0), "bs! inv\n");
        if (bn.height > 0) {
            tmp = pptr_to_ml3bptr(nxt_addr);
        } else {
            *addr = tmp;
            *value = nxt_addr;
            return pred;
        }
    }
}

static inline int get_r(pptr* addrs, int n, int l) {
    int r;
    pptr p1 = addrs[l];
    for (r = l; r < n; r++) {
        pptr p2 = addrs[r];
        if (EQUAL_PPTR(p1, p2)) {
            continue;
        } else {
            break;
        }
    }
    return r;
}

#define L3_TEMP_BUFFER_SIZE_FULL (500)

int64_t *mod_keys, *mod_keys2;
pptr *mod_values, *mod_values2;
pptr *mod_addrs, *mod_addrs2;
int64_t *mod_type, *mod_type2;

const int64_t remove_type = 1;
const int64_t change_key_type = 2;
const int64_t underflow_type = 4;

int *L3_lfts, *L3_rts;
int64_t L3_n;

#if (defined DPU_INSERT) || (defined DPU_BUILD)
void l3b_insert_onelevel(int n, int tid, int ht) {
    L3Bnode bn;
    int64_t nnkeys[DB_SIZE];
    pptr nnvalues[DB_SIZE];
    int lft = L3_lfts[tid];
    int rt = L3_rts[tid];
    int nxtlft = lft;
    int nxtrt = nxtlft;
    int siz = 0;

    int l, r;  // catch all inserts to the same node
    for (l = lft; l < rt; l = r) {
        r = get_r(mod_addrs, n, l);
        pptr addr = mod_addrs[l];
        mL3Bptr nn;
        pptr up, right;
        int nnsize;
        mL3Bptr nn0;
        if (valid_pptr(addr)) {
            nn0 = nn = pptr_to_ml3bptr(addr);
            m_read(nn, &bn, sizeof(L3Bnode));
            up = bn.up;
            right = bn.right;
            nnsize = bn.size;
            for (int i = 0; i < nnsize; i++) {
                nnkeys[i] = bn.keys[i];
                nnvalues[i] = bn.addrs[i];
            }
        } else {
            nn = alloc_l3bn();
            up = null_pptr;
            right = null_pptr;
            nnsize = 1;
            nnkeys[0] = INT64_MIN;
            nnvalues[0] = ml3bptr_to_pptr(root);
            if (ht > 0) {
                root->up = ml3bptr_to_pptr(nn);
                for (int i = l; i < r; i++) {
                    pptr child_addr = mod_values[i];
                    mL3Bptr ch = pptr_to_ml3bptr(child_addr);
                    ch->up = ml3bptr_to_pptr(nn);
                }
            }
            root = nn;
        }
        l3b_node_init(&bn, ht, up, right);
        int totsize = 0;

        {
            int nnl = 0;
            int i = l;
            while (i < r || nnl < nnsize) {
                if (i < r && nnl < nnsize) {
                    if (mod_keys[i] == nnkeys[nnl]) {
                        i++;
                        totsize--;  // replace
                    } else if (mod_keys[i] < nnkeys[nnl]) {
                        i++;
                    } else {
                        nnl++;
                    }
                } else if (i == r) {
                    nnl++;
                } else {
                    i++;
                }
                totsize++;
            }
        }

        int nnl = 0;
        bn.size = 0;

        int l0 = l;
        for (int i = 0; nnl < nnsize || l < r; i++) {
            if (nnl < nnsize && (l == r || nnkeys[nnl] < mod_keys[l])) {
                bn.keys[bn.size] = nnkeys[nnl];
                bn.addrs[bn.size] = nnvalues[nnl];
                nnl++;
            } else {
                bn.keys[bn.size] = mod_keys[l];
                bn.addrs[bn.size] = mod_values[l];
                if (nnl < nnsize && nnkeys[nnl] == mod_keys[l]) {  // replace
                    nnl++;
                }
                l++;
            }
            bn.size++;
            if (bn.size == 1 && (i > 0)) {  // newnode
                mod_keys2[nxtrt] = bn.keys[0];
                mod_values2[nxtrt] = ml3bptr_to_pptr(nn);
                mod_addrs2[nxtrt] = bn.up;
                nxtrt++;
                // IN_DPU_ASSERT_EXEC(bn.keys[0] != INT64_MIN, {
                //     printf("mod_keys2 = INT64_MIN\n");
                //     // printf("l3bbuffer=%x l3bcnt=%d\n", l3bbuffer, l3bcnt);
                //     // printf("ht=%d addr=%x\n", ht, nn0);
                //     // printf("l=%d\tr=%d\tnnl=%d\n", l0, r, nnl);
                //     // printf("i=%d\ttotsize=%d\n", i, totsize);
                //     // for (int x = 0; x < nnsize; x++) {
                //     //     printf("nn[%d]=%lld\n", x, nnkeys[x]);
                //     // }
                //     // // mram_read(nn0, &bn, sizeof(L3Bnode));
                //     // m_read(nn0, &bn, sizeof(L3Bnode));
                //     // printf("nn0size=%lld nnsize=%d\n", bn.size, nnsize);
                //     // for (int x = 0; x < bn.size; x++) {
                //     //     printf("nn0[%d]=%lld\n", x, bn.keys[x]);
                //     // }
                //     // for (int x = l0; x < r; x++) {
                //     //     printf("mod[%d]=%lld\n", x, mod_keys[x]);
                //     // }
                //     // printf("i=%d\ttotsize=%d\n", i, totsize);
                // });
            }
            if (bn.size == DB_SIZE ||
                (i + HF_DB_SIZE + 1 == totsize && bn.size > HF_DB_SIZE)) {
                for (int i = 0; i < bn.size; i++) {
                    if (bn.height > 0) {
                        IN_DPU_ASSERT(valid_pptr(bn.addrs[i]), "bio! inv\n");
                        mL3Bptr ch = pptr_to_ml3bptr(bn.addrs[i]);
                        ch->up = ml3bptr_to_pptr(nn);
                    }
                }
                if (nnl == nnsize && l == r) {
                    m_write(&bn, nn, sizeof(L3Bnode));
                } else {
                    pptr up = bn.up, right = bn.right;
                    int64_t ht = bn.height;

                    mL3Bptr nxt_nn = alloc_l3bn();
                    bn.right = ml3bptr_to_pptr(nxt_nn);
                    m_write(&bn, nn, sizeof(L3Bnode));

                    l3b_node_init(&bn, ht, up, right);
                    nn = nxt_nn;
                }
            }
            // if (nnl == nnsize && l == r) {
            //     IN_DPU_ASSERT_EXEC(i + 1 == totsize, {
            //         printf("i+1 != totsize\n");
            //         // printf("l3bbuffer=%x l3bcnt=%d\n", l3bbuffer, l3bcnt);
            //         // printf("ht=%d addr=%x\n", ht, nn0);
            //         // printf("l=%d\tr=%d\tnnl=%d\n", l0, r, nnl);
            //         // printf("i=%d\ttotsize=%d\n", i, totsize);
            //         // for (int x = 0; x < nnsize; x++) {
            //         //     printf("nn[%d]=%lld\n", x, nnkeys[x]);
            //         // }
            //         // // mram_read(nn0, &bn, sizeof(L3Bnode));
            //         // m_read(nn0, &bn, sizeof(L3Bnode));
            //         // printf("nn0size=%lld nnsize=%d\n", bn.size, nnsize);
            //         // for (int x = 0; x < bn.size; x++) {
            //         //     printf("nn0[%d]=%lld\n", x, bn.keys[x]);
            //         // }
            //         // for (int x = l0; x < r; x++) {
            //         //     printf("mod[%d]=%lld\n", x, mod_keys[x]);
            //         // }
            //         // printf("i=%d\ttotsize=%d\n", i, totsize);
            //     });
            // }
        }
        if (bn.size != 0) {
            for (int i = 0; i < bn.size; i++) {
                if (bn.height > 0) {
                    // IN_DPU_ASSERT(valid_pptr(bn.addrs[i]), "bio! inv2\n");
                    mL3Bptr ch = pptr_to_ml3bptr(bn.addrs[i]);
                    ch->up = ml3bptr_to_pptr(nn);
                }
            }
            m_write(&bn, nn, sizeof(L3Bnode));
        }
        // IN_DPU_ASSERT_EXEC(l == r && nnl == nnsize, {
        //     printf("l=%d\tr=%d\tnnl=%d\tnnsize=%d\n", l, r, nnl, nnsize);
        // });
    }
    L3_lfts[tid] = nxtlft;
    L3_rts[tid] = nxtrt;
}
#endif

#ifdef DPU_DELETE
void l3b_remove_onelevel(int n, int tid, int ht) {
    L3Bnode bn;
    int64_t nnkeys[DB_SIZE];
    pptr nnvalues[DB_SIZE];
    int lft = L3_lfts[tid];
    int rt = L3_rts[tid];
    int nxtlft = lft;
    int nxtrt = nxtlft;
    int siz = 0;

    int l, r;  // catch all inserts to the same node
    for (l = lft; l < rt; l = r) {
        r = get_r(mod_addrs, n, l);

        pptr addr = mod_addrs[l];
        // IN_DPU_ASSERT(valid_pptr(addr), "bro! inva\n");

        mL3Bptr nn = pptr_to_ml3bptr(addr);
        m_read(nn, &bn, sizeof(L3Bnode));
        pptr up = bn.up, right = bn.right;
        int nnsize = bn.size;
        for (int i = 0; i < nnsize; i++) {
            nnkeys[i] = bn.keys[i];
            nnvalues[i] = bn.addrs[i];
        }

        l3b_node_init(&bn, ht, up, right);
        int totsize = 0;

        {
            int nnl = 0;
            int i = l;
            while (nnl < nnsize) {
                if (i == r || nnkeys[nnl] < mod_keys[i]) {
                    bn.keys[bn.size] = nnkeys[nnl];
                    bn.addrs[bn.size] = nnvalues[nnl];
                    bn.size++;
                    nnl++;
                } else if (nnkeys[nnl] > mod_keys[i]) {
                    i++;
                } else {  // equal
                    if (mod_type[i] == change_key_type) {
                        bn.keys[bn.size] = pptr_to_int64(mod_values[i]);
                        bn.addrs[bn.size] = nnvalues[nnl];
                        bn.size++;
                    } else if (mod_type[i] == remove_type) {
                        mL3Bptr removed_child = nnvalues[nnl].addr;
                        if (in_l3bbuffer(removed_child)) {
                            free_node(&free_list_l3bnode, (mpvoid)removed_child);
                        }
                    }
                    nnl++;
                }
            }
        }
        m_write(&bn, nn, sizeof(L3Bnode));

        int future_modif = 0;
        if (bn.size < HF_DB_SIZE) {
            future_modif |= underflow_type; // underflow, requires merge / rotate
        }
        if (bn.size == 0 || nnkeys[0] != bn.keys[0]) {
            future_modif |= change_key_type; // pivot key changed
        }
        if (nn == root) {
            future_modif = 0; // shouldn't remove root
        }
        if (future_modif > 0) {
            mod_keys2[nxtrt] = nnkeys[0];
            mod_addrs2[nxtrt] = addr;
            mod_values2[nxtrt] = int64_to_pptr(bn.keys[0]);
            mod_type2[nxtrt] = future_modif;
            nxtrt++;
        }
    }
    L3_lfts[tid] = nxtlft;
    L3_rts[tid] = nxtrt;
}
#endif

const int SERIAL_HEIGHT = 2;
// temporarily switched off parallel insert / delete, as it may cause program crash for some reason
// switched on again as it seems not to be the cause.

#if (defined DPU_INSERT) || (defined DPU_BUILD)
void l3b_insert_parallel(int n, int l, int r) {
    int tid = me();

    barrier_wait(&L3_barrier2);

    // bottom up
    for (int i = l; i < r; i++) {
        int64_t key = mod_keys[i];
        pptr value;

        mL3Bptr nn;
        l3b_search(key, &nn, &value);
        mod_addrs[i] = ml3bptr_to_pptr(nn);
        // if (i > 0) {
        //     int64_t keyl = mod_keys[i - 1];
        //     // IN_DPU_ASSERT_EXEC(keyl < key, {
        //     //     printf("bip! eq %d %d %d %d %lld %lld\n", i, tid, l, r, key,
        //     //            keyl);
        //     //     for (int i = 0; i < n; i++) {
        //     //         printf("key[%d]=%lld\n", i, mod_keys[i]);
        //     //     }
        //     // });
        //     IN_DPU_ASSERT_EXEC(keyl < key, {});
        // }
    }

    L3_n = n;
    barrier_wait(&L3_barrier1);

    for (int ht = 0; ht <= root->height + 1; ht++) {
        if (ht < SERIAL_HEIGHT) {
            // distribute work
            n = L3_n;
            // if (tid == 0) {
            //     printf("PARALLEL:%d\n", n);
            // }
            int lft = n * tid / NR_TASKLETS;
            int rt = n * (tid + 1) / NR_TASKLETS;
            if (rt > lft) {
                if (lft != 0) {
                    lft = get_r(mod_addrs, n, lft - 1);
                }
                // IN_DPU_ASSERT(rt > 0, "bi! rt\n");
                rt = get_r(mod_addrs, n, rt - 1);
            }

            L3_lfts[tid] = lft;
            L3_rts[tid] = rt;
            barrier_wait(&L3_barrier2);

            // execute
            l3b_insert_onelevel(n, tid, ht);
            barrier_wait(&L3_barrier3);

            // distribute work
            if (tid == 0) {
                n = 0;
                for (int i = 0; i < NR_TASKLETS; i++) {
                    for (int j = L3_lfts[i]; j < L3_rts[i]; j++) {
                        mod_keys[n] = mod_keys2[j];
                        mod_values[n] = mod_values2[j];
                        mod_addrs[n] = mod_addrs2[j];
                        // IN_DPU_ASSERT(mod_keys[n] != INT64_MIN, "bi! min\n");
                        // if (n > 0) {
                        //     int64_t key = mod_keys[n];
                        //     int64_t keyl = mod_keys[n - 1];
                        //     // IN_DPU_ASSERT_EXEC(key > keyl, {
                        //     //     printf("bip! rev %d %d %d %d %lld %lld\n", i, j, tid, n, key,
                        //     //         keyl);
                        //     //     for (int i = n - 10; i < n + 10; i++) {
                        //     //         printf("key[%d]=%lld\n", i, mod_keys[i]);
                        //     //     }
                        //     // });
                        //     // IN_DPU_ASSERT_EXEC(key > keyl, {});
                        //     IN_DPU_ASSERT(mod_keys[n] > mod_keys[n - 1],
                        //                   "bip! rev\n");
                        // }
                        n++;
                    }
                }
                L3_n = n;
                // printf("n=%d\n", n);
            }
            barrier_wait(&L3_barrier1);
        } else {
            if (tid == 0 && n > 0) {
                // printf("SOLO:%d\n", n);
                L3_lfts[0] = 0;
                L3_rts[0] = n;
                l3b_insert_onelevel(n, tid, ht);
                n = L3_rts[0];
                for (int i = 0; i < n; i++) {
                    mod_keys[i] = mod_keys2[i];
                    mod_values[i] = mod_values2[i];
                    mod_addrs[i] = mod_addrs2[i];
                }
            } else {
                break;
            }
        }
    }
    barrier_wait(&L3_barrier3);
}
#endif

int64_t scnnkeys[DB_SIZE + HF_DB_SIZE + 3];
pptr scnnaddrs[DB_SIZE + HF_DB_SIZE + 3];

#ifdef DPU_DELETE
void l3b_remove_serial_compact(int n, int ht) {
    L3Bnode bn;
    int64_t *nnkeys = scnnkeys;
    pptr* nnaddrs = scnnaddrs;

    int nxt_n = 0;
    for (int l = 0; l < n;) {
        int mt = mod_type[l];
        if (mt == change_key_type) {
            mod_keys2[nxt_n] = mod_keys[l];
            mod_values2[nxt_n] = mod_values[l];
            mL3Bptr nn = pptr_to_ml3bptr(mod_addrs[l]);
            mod_addrs2[nxt_n] = nn->up;
            mod_type2[nxt_n] = change_key_type;
            nxt_n++;
            l++;
        } else if (mt & underflow_type) {
            int r = l;
            pptr addr = mod_addrs[l];
            int nnl = 0;
            while (nnl < HF_DB_SIZE) {
                mL3Bptr nn = pptr_to_ml3bptr(addr);
                m_read(nn, &bn, sizeof(L3Bnode));
                for (int i = 0; i < bn.size; i++) {
                    nnkeys[nnl] = bn.keys[i];
                    nnaddrs[nnl] = bn.addrs[i];
                    nnl++;
                }
                if (r < n && equal_pptr(addr, mod_addrs[r])) {
                    r++;
                } else {
                    // IN_DPU_ASSERT(bn.size >= HF_DB_SIZE ||
                    //        equal_pptr(bn.right, null_pptr), "brsc! mr\n");
                }
                if (nnl < HF_DB_SIZE && !equal_pptr(bn.right, null_pptr)) {
                    addr = bn.right;
                } else {
                    break;
                }
            }

            mL3Bptr left_nn = pptr_to_ml3bptr(mod_addrs[l]);
            pptr right_out = bn.right;

            // prepare for upper level
            if (nnl > 0) {
                if (nnkeys[0] != mod_keys[l]) {  // change key
                    mod_keys2[nxt_n] = mod_keys[l];
                    mod_values2[nxt_n] = int64_to_pptr(nnkeys[0]);
                    mod_addrs2[nxt_n] = left_nn->up;
                    mod_type2[nxt_n] = change_key_type;
                    nxt_n++;
                }
                left_nn->right = right_out;
            } else {
                mod_keys2[nxt_n] = mod_keys[l];
                mod_values2[nxt_n] = null_pptr;
                mod_addrs2[nxt_n] = left_nn->up;
                mod_type2[nxt_n] = remove_type;
                nxt_n++;
            }

            int mid = nnl >> 1;
            int64_t mid_key = nnkeys[mid];

            if (nnl > 0 && nnl <= DB_SIZE) {
                mL3Bptr nn = pptr_to_ml3bptr(mod_addrs[l]);
                l3b_node_fill(nn, &bn, nnl, nnkeys, nnaddrs);
                bool addr_covered = equal_pptr(addr, mod_addrs[l]);
                for (int i = l + 1; i < r; i++) {
                    if (equal_pptr(addr, mod_addrs[i])) {
                        addr_covered = true;
                    }
                    mL3Bptr nn = pptr_to_ml3bptr(mod_addrs[i]);
                    mod_keys2[nxt_n] = mod_keys[i];
                    mod_values2[nxt_n] = null_pptr;
                    mod_addrs2[nxt_n] = nn->up;
                    mod_type2[nxt_n] = remove_type;
                    nxt_n++;
                }
                if (addr_covered == false) {
                    nn = pptr_to_ml3bptr(addr);
                    m_read(nn, &bn, sizeof(L3Bnode));
                    // IN_DPU_ASSERT((bn.size >= HF_DB_SIZE ||
                    //        equal_pptr(bn.right, null_pptr)), "brsc! mr\n");
                    mod_keys2[nxt_n] = bn.keys[0];
                    mod_values2[nxt_n] = null_pptr;
                    mod_addrs2[nxt_n] = bn.up;
                    mod_type2[nxt_n] = remove_type;
                    nxt_n++;
                }
            } else if (nnl > DB_SIZE) {
                bool addr_covered = false;
                mL3Bptr nn = pptr_to_ml3bptr(mod_addrs[l]);
                l3b_node_fill(nn, &bn, mid, nnkeys, nnaddrs);
                int filpos = l;
                {
                    for (int i = l; i < r; i++) {
                        if (equal_pptr(addr, mod_addrs[i])) {
                            addr_covered = true;
                        }
                        if (mod_keys[i] <= mid_key) {
                            filpos = i;
                        }
                    }
                    if (!addr_covered) {
                        mL3Bptr nn = pptr_to_ml3bptr(addr);
                        // IN_DPU_ASSERT(nn->size >= HF_DB_SIZE ||
                        //    equal_pptr(nn->right, null_pptr), "brsc! mr\n");
                        if (nn->keys[0] <= mid_key) {
                            filpos = -1;
                        }
                    }
                }
                // IN_DPU_ASSERT(filpos != l, "fil=l\n");

                for (int i = l + 1; i < r; i++) {
                    mL3Bptr nn = pptr_to_ml3bptr(mod_addrs[i]);
                    if (i != filpos) {
                        mod_keys2[nxt_n] = mod_keys[i];
                        mod_addrs2[nxt_n] = nn->up;
                        mod_type2[nxt_n] = remove_type;
                        nxt_n++;
                    } else {
                        l3b_node_fill(nn, &bn, nnl - mid, nnkeys + mid,
                                    nnaddrs + mid);
                        left_nn->right = mod_addrs[i];
                        nn->right = right_out;

                        mod_keys2[nxt_n] = mod_keys[i];
                        mod_values2[nxt_n] = int64_to_pptr(mid_key);
                        mod_addrs2[nxt_n] = nn->up;
                        mod_type2[nxt_n] = change_key_type;
                        nxt_n++;
                    }
                }
                if (!addr_covered) {
                    mL3Bptr nn = pptr_to_ml3bptr(addr);
                    // IN_DPU_ASSERT(nn->size >= HF_DB_SIZE ||
                    //        equal_pptr(nn->right, null_pptr), "brsc! mr\n");
                    if (filpos != -1) {
                        mod_keys2[nxt_n] = nn->keys[0];
                        mod_addrs2[nxt_n] = nn->up;
                        mod_type2[nxt_n] = remove_type;
                        nxt_n++;
                    } else {
                        left_nn->right = addr;
                        nn->right = right_out;

                        mod_keys2[nxt_n] = nn->keys[0];
                        mod_values2[nxt_n] = int64_to_pptr(mid_key);
                        mod_addrs2[nxt_n] = nn->up;
                        mod_type2[nxt_n] = change_key_type;
                        nxt_n++;
                        l3b_node_fill(nn, &bn, nnl - mid, nnkeys + mid,
                                    nnaddrs + mid);
                    }
                }
            } else {
                // pass
            }

            l = r;
        } else {
            // IN_DPU_ASSERT(false, "mt\n");
        }
    }
    L3_lfts[0] = 0;
    L3_rts[0] = nxt_n;
}

void l3b_remove_parallel(int n, int l, int r) {
    int tid = me();
    barrier_wait(&L3_barrier2);

    // bottom up
    for (int i = l; i < r; i++) {
        int64_t key = mod_keys[i];
        pptr value;
        mL3Bptr nn;
        l3b_search(key, &nn, &value);
        mod_addrs[i] = ml3bptr_to_pptr(nn);
        mod_type[i] = remove_type;
    }

    L3_n = n;
    barrier_wait(&L3_barrier1);

    for (int ht = 0; ht <= root->height; ht++) {
        if (ht < SERIAL_HEIGHT) {
            // distribute work
            n = L3_n;
            int lft = n * tid / NR_TASKLETS;
            int rt = n * (tid + 1) / NR_TASKLETS;
            if (rt > lft) {
                if (lft != 0) {
                    lft = get_r(mod_addrs, n, lft - 1);
                }
                // IN_DPU_ASSERT(rt > 0, "br! rt\n");
                rt = get_r(mod_addrs, n, rt - 1);
            }

            L3_lfts[tid] = lft;
            L3_rts[tid] = rt;
            barrier_wait(&L3_barrier2);

            // execute
            l3b_remove_onelevel(n, tid, ht);
            barrier_wait(&L3_barrier3);

            // distribute work
            if (tid == 0) {
                n = 0;
                for (int i = 0; i < NR_TASKLETS; i++) {
                    for (int j = L3_lfts[i]; j < L3_rts[i]; j++) {
                        mod_keys[n] = mod_keys2[j];
                        mod_values[n] = mod_values2[j];
                        mod_addrs[n] = mod_addrs2[j];
                        mod_type[n] = mod_type2[j];
                        n++;
                    }
                }
                l3b_remove_serial_compact(n, ht);
                n = L3_rts[0];
                for (int i = 0; i < n; i++) {
                    mod_keys[i] = mod_keys2[i];
                    mod_values[i] = mod_values2[i];
                    mod_addrs[i] = mod_addrs2[i];
                    mod_type[i] = mod_type2[i];
                }
                L3_n = n;
            }
            barrier_wait(&L3_barrier1);
        } else {
            if (tid == 0 && n > 0) {
                // printf("SOLO:%d\n", n);
                L3_lfts[0] = 0;
                L3_rts[0] = n;
                l3b_remove_onelevel(n, tid, ht);
                n = L3_rts[0];
                for (int i = 0; i < n; i++) {
                    mod_keys[i] = mod_keys2[i];
                    mod_values[i] = mod_values2[i];
                    mod_addrs[i] = mod_addrs2[i];
                    mod_type[i] = mod_type2[i];
                }
                l3b_remove_serial_compact(n, ht);
                n = L3_rts[0];
                for (int i = 0; i < n; i++) {
                    mod_keys[i] = mod_keys2[i];
                    mod_values[i] = mod_values2[i];
                    mod_addrs[i] = mod_addrs2[i];
                    mod_type[i] = mod_type2[i];
                }
            } else {
                break;
            }
        }
    }
    barrier_wait(&L3_barrier3);
}
#endif

#ifdef DPU_SCAN
static inline void l3b_scan(int64_t lkey, int64_t rkey,
                               varlen_buffer_dpu *addr_buf,
                               varlen_buffer_dpu *up_buf,
                               varlen_buffer_dpu *down_buf) {
    mL3Bptr tmp;
    L3Bnode bn;
    varlen_buffer_dpu *tmp_buf;
    varlen_buffer_reset_dpu(addr_buf);
    varlen_buffer_reset_dpu(up_buf);
    varlen_buffer_reset_dpu(down_buf);
    pptr cur_addr = ml3bptr_to_pptr(root);
    varlen_buffer_push_dpu(up_buf, pptr_to_int64(cur_addr));
    bool flag;
    int64_t tmp_max_value, vee = INT64_MIN, tmp_key;
    int tmp_max_idx, jbb = -1, jee = -1;
    while (up_buf->len > 0) {
        for(int64_t j = 0; j < up_buf->len; j++) {
            tmp_key = varlen_buffer_element_dpu(up_buf, j);
            cur_addr = int64_to_pptr(tmp_key);
            tmp = pptr_to_ml3bptr(cur_addr);
            m_read(tmp, &bn, sizeof(L3Bnode));
            flag = false;
            tmp_max_value = INT64_MIN;
            for (int i = 0; i < bn.size; i++) {
                if(bn.height > 0) {
                    if(bn.keys[i] <= lkey) {
                        flag = true;
                        if(bn.keys[i] >= tmp_max_value) {
                            tmp_max_value = bn.keys[i];
                            tmp_max_idx = i;
                        }
                    }
                    else if(bn.keys[i] <= rkey) {
                        tmp_key = pptr_to_int64(bn.addrs[i]);
                        varlen_buffer_push_dpu(down_buf, tmp_key);
                    }
                        
                }
                else {
                    if(bn.keys[i] <= lkey) {
                        flag = true;
                        if(bn.keys[i] >= tmp_max_value) {
                            tmp_max_value = bn.keys[i];
                            tmp_max_idx = i;
                        }
                    }
                    else if (bn.keys[i] <= rkey) {
                        if(bn.keys[i] >= vee) {
                            vee = bn.keys[i];
                            jee = addr_buf->len;
                        }
                        tmp_key = pptr_to_int64(bn.addrs[i]);
                        varlen_buffer_push_dpu(addr_buf, tmp_key);
                    }
                }
            }
            if(flag) {
                if(bn.height > 0){
                    tmp_key = pptr_to_int64(bn.addrs[tmp_max_idx]);
                    varlen_buffer_push_dpu(down_buf, tmp_key);
                }
                else {
                    jbb = addr_buf->len;
                    tmp_key = pptr_to_int64(bn.addrs[tmp_max_idx]);
                    varlen_buffer_push_dpu(addr_buf, tmp_key);
                }
            }
        }
        tmp_buf = up_buf, up_buf = down_buf, down_buf = tmp_buf;
        varlen_buffer_reset_dpu(down_buf);
    }
    if(jbb >= 0 && jbb < addr_buf->len) {
        tmp_max_value = varlen_buffer_element_dpu(addr_buf, 0);
        vee = varlen_buffer_element_dpu(addr_buf, jbb);
        varlen_buffer_set_element_dpu(addr_buf, 0, vee);
        varlen_buffer_set_element_dpu(addr_buf, jbb, tmp_max_value);
    }
    if(jee >= 0 && jee < addr_buf->len) {
        tmp_max_value = varlen_buffer_element_dpu(addr_buf, addr_buf->len - 1);
        vee = varlen_buffer_element_dpu(addr_buf, jee);
        varlen_buffer_set_element_dpu(addr_buf, addr_buf->len - 1, vee);
        varlen_buffer_set_element_dpu(addr_buf, jee, tmp_max_value);
    }
}
#endif
