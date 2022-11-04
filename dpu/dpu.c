// the program may not run without this macro.
#define IRAM_FRIENDLY

#ifndef IRAM_FRIENDLY
#define DPU_INSERT
#define DPU_DELETE
#define DPU_SCAN
#define DPU_GET_UPDATE
#define DPU_PREDECESSOR
#define DPU_BUILD
#endif

#include "driver.h"

#ifdef L3_SKIP_LIST
#include "l3_skip_list.h"
#else
#include "l3_ab_tree.h"
#endif

#include "task.h"
#include "bnode.h"
#include "pnode.h"
#include "statistics.h"
#include "storage.h"
// IRAM friendly

BARRIER_INIT(exec_barrier1, NR_TASKLETS);
BARRIER_INIT(exec_barrier2, NR_TASKLETS);

MUTEX_INIT(dpu_lock);

extern volatile int64_t recv_epoch_number;
/* -------------- Storage -------------- */

// DPU ID
__host int64_t DPU_ID = -1;

void *bufferA_shared, *bufferB_shared;
int8_t *max_height_shared;
uint32_t *newnode_size;
int64_t *keys; pptr* addrs;

// Node Buffers & Hash Tables

#ifdef DPU_ENERGY
__host uint64_t op_count;
__host uint64_t db_size_count;
__host uint64_t cycle_count;
#endif

extern uint32_t l3cnt, bcnt, dbcnt;

static inline int get_r_mram(mppptr addrs, int n, int l) {
    int r;
    pptr p1 = addrs[l];
    for (r = l; r < n; r++) {
        pptr p2 = addrs[r];
        if (equal_pptr(p1, p2)) {
            continue;
        } else {
            break;
        }
    }
    return r;
}

static inline void dpu_init(dpu_init_task *it) {
    DPU_ID = it->dpu_id;
    statistic_init();
    bcnt = 1; dbcnt = 1; pcnt = 1;
    #ifndef SKIP_LIST
    l3bcnt = 1;
    l3bbuffer_start = l3bbuffer + 1;
    l3bbuffer_end = l3bbuffer_start + (L3_BUFFER_SIZE / sizeof (L3Bnode)) - 1;
    #endif
    bbuffer_start = bbuffer + 1;
    bbuffer_end = bbuffer_start + (B_BUFFER_SIZE / sizeof(Bnode)) - 1;
    pbuffer_start = pbuffer + 1;
    pbuffer_end = pbuffer_start + (P_BUFFER_SIZE / sizeof(Pnode)) - 1;
    dbbuffer_start = dbbuffer + 1;
    dbbuffer_end = dbbuffer_start + (DB_BUFFER_SIZE / sizeof(data_block)) - 1;
    cirbuffer_start = cirbuffer + 1;
    cirbuffer_end =
        cirbuffer + (CACHE_INIT_RECORD_SIZE / sizeof(cache_init_record));
    gc_init();
    storage_init();
}

void execute(int l, int r) {
    // IN_DPU_ASSERT(recv_block_task_type == INIT_TSK || DPU_ID != -1,
    //               "execute: id not initialized");
    uint32_t tasklet_id = me();
    int length = r - l;

    switch (recv_block_task_type) {
#ifdef DPU_INIT
        case task_id(dpu_init_task): {
            init_block_with_type(dpu_init_task, dpu_init_reply);

            if (tasklet_id == 0) {
                init_task_reader(0);
                dpu_init_task* it = (dpu_init_task*)get_task_cached(0);
                dpu_init(it);
                dpu_init_reply ir =
                    (dpu_init_reply){.bbuffer_start = (uint64_t)bbuffer_start,
                                 .bbuffer_end = (uint64_t)bbuffer_end,
                                 .pbuffer_start = (uint64_t)pbuffer_start,
                                 .pbuffer_end = (uint64_t)pbuffer_end};
                push_fixed_reply(0, &ir);
            }
            break;
        }
#endif

// #ifdef DPU_INSERT
#ifdef DPU_INIT
#ifdef L3_SKIP_LIST
        case L3_INIT_TSK: {
            init_block_with_type(L3_init_task, empty_task_reply);
            if (tasklet_id == 0) {
                init_task_reader(0);
                L3_init_task* tit = (L3_init_task*)get_task_cached(0);
                L3_init(tit);
            }
            break;
        }
#else
        case L3_INIT_TSK: {
            init_block_with_type(L3_init_task, empty_task_reply);
            if (tasklet_id == 0) {
                init_task_reader(0);
                L3_init_task* tit = (L3_init_task*)get_task_cached(0);
                L3_init(tit);
            }
            break;
        }
#endif
#endif

// #if (defined DPU_INSERT) 
#if (defined DPU_BUILD) || (defined DPU_INSERT) 
#ifdef L3_SKIP_LIST
        case L3_INSERT_TSK: {
            init_block_with_type(L3_insert_task, empty_task_reply);
            int64_t* keys = mem_alloc(sizeof(int64_t) * length);
            int8_t* heights = mem_alloc(sizeof(int8_t) * length);
            pptr* down = mem_alloc(sizeof(pptr) * length);

            init_task_reader(l);

            newnode_size[tasklet_id] = 0;
            for (int i = 0; i < length; i++) {
                L3_insert_task* tit = (L3_insert_task*)get_task_cached(i + l);
                keys[i] = tit->key;
                heights[i] = tit->height;
                down[i] = tit->down;
                newnode_size[tasklet_id] += L3_node_size(heights[i]);
                L3_IN_DPU_ASSERT(heights[i] > 0 && heights[i] < MAX_L3_HEIGHT,
                                 "execute: invalid height\n");
            }

            mL3ptr* right_predecessor_shared = bufferA_shared;
            mL3ptr* right_newnode_shared = bufferB_shared;
            L3_insert_parallel(length, l, keys, heights, down, newnode_size,
                               max_height_shared, right_predecessor_shared,
                               right_newnode_shared);
            break;
        }
#else
        case L3_INSERT_TSK: {
            init_block_with_type(L3_insert_task, empty_task_reply);
            init_task_reader(l);

            for (int i = l; i < r; i++) {
                L3_insert_task* tit = (L3_insert_task*)get_task_cached(i);
                mod_keys[i] = tit->key;
                mod_values[i] = tit->down;
            }

            l3b_insert_parallel(recv_block_task_cnt, l, r);
            break;
        }
#endif
#endif

#ifdef DPU_DELETE
#ifdef L3_SKIP_LIST
        case L3_REMOVE_TSK: {
            init_block_with_type(L3_remove_task, empty_task_reply);
            int64_t* keys = mem_alloc(sizeof(int64_t) * length);
            mL3ptr* nodes = mem_alloc(sizeof(mL3ptr) * length);

            init_task_reader(l);
            for (int i = 0; i < length; i++) {
                L3_remove_task* trt = (L3_remove_task*)get_task_cached(i + l);
                keys[i] = trt->key;
                L3_search(keys[i], 0, 1, &nodes[i]);
            }

            for (int i = 0; i < length; i++) {
                mL3ptr l3r = nodes[i];
                IN_DPU_ASSERT_EXEC(l3r->key == keys[i], {
                    printf("l3r: invkey=%llx\taddr=%x\tcorrectkey=%llx\n",
                           keys[i], nodes[i], l3r->key);
                });
            }

            mL3ptr* left_node_shared = bufferA_shared;
            L3_remove_parallel(length, nodes, max_height_shared,
                               left_node_shared);
            break;
        }
#else
        case L3_REMOVE_TSK: {
            init_block_with_type(L3_remove_task, empty_task_reply);

            init_task_reader(l);
            for (int i = l; i < r; i++) {
                L3_remove_task* trt = (L3_remove_task*)get_task_cached(i);
                mod_keys[i] = trt->key;
            }
            l3b_remove_parallel(recv_block_task_cnt, l, r);
            break;
        }
#endif
#endif

// #ifdef DPU_PREDECESSOR
#if (defined DPU_BUILD) || (defined DPU_PREDECESSOR)
#ifdef L3_SKIP_LIST
        case L3_SEARCH_TSK: {
            init_block_with_type(L3_search_task, L3_search_reply);
            init_task_reader(l);
            for (int i = l; i < r; i++) {
                L3_search_task* tst = (L3_search_task*)get_task_cached(i);
                L3_search(tst->key, i, 0, NULL);
            }
            break;
        }
#else
        case L3_SEARCH_TSK: {
            init_block_with_type(L3_search_task, L3_search_reply);
            init_task_reader(l);
            for (int i = l; i < r; i++) {
                L3_search_task* tst = (L3_search_task*)get_task_cached(i);
                mL3Bptr nn;
                pptr val = null_pptr;
                int64_t key = l3b_search(tst->key, &nn, &val);
                L3_search_reply tsr = (L3_search_reply){.addr = val};
                push_fixed_reply(i, &tsr);
            }
            break;
        }
#endif
#endif

#ifdef DPU_SCAN
#ifdef L3_SKIP_LIST
        case L3_SCAN_TSK: {
            init_block_with_type(L3_scan_task, L3_scan_reply);
            init_task_reader(l);
            varlen_buffer_dpu* addrsbuf = 
                varlen_buffer_new_dpu(VARLEN_BUFFER_SIZE, mrambuffer + M_BUFFER_SIZE / NR_TASKLETS * tasklet_id);
            for (int i = l; i < r; i++) {
                L3_scan_task* l3sst = (L3_scan_task*)get_task_cached(i);
                int64_t bb = l3sst->lkey;
                int64_t ee = l3sst->rkey;
                int num = L3_scan_search(bb, ee, addrsbuf);
                // IN_DPU_ASSERT(num == addrsbuf->len && num > 0, "l3 scan search error");
                __mram_ptr int64_t* replyptr =
                    (__mram_ptr int64_t*)push_variable_reply_zero_copy(
                        tasklet_id, S64(num + 1));
                replyptr[0] = num;
                varlen_buffer_to_mram_dpu(addrsbuf, replyptr + 1);
            }
            break;
        }
#else
        case L3_SCAN_TSK: {
            init_block_with_type(L3_scan_task, L3_scan_reply);
            init_task_reader(l);
            varlen_buffer_dpu* addrsbuf = 
                varlen_buffer_new_dpu(VARLEN_BUFFER_SIZE, mrambuffer + M_BUFFER_SIZE / NR_TASKLETS * tasklet_id);
            varlen_buffer_dpu* upbuf = 
                varlen_buffer_new_dpu(VARLEN_BUFFER_SIZE,
                    mrambuffer + (M_BUFFER_SIZE / NR_TASKLETS * tasklet_id) + (M_BUFFER_SIZE / NR_TASKLETS / 3));
            varlen_buffer_dpu* downbuf = 
                varlen_buffer_new_dpu(VARLEN_BUFFER_SIZE,
                    mrambuffer + (M_BUFFER_SIZE / NR_TASKLETS * tasklet_id) + (M_BUFFER_SIZE / NR_TASKLETS / 3 * 2));
            for (int i = l; i < r; i++) {
                L3_scan_task* l3sst = (L3_scan_task*)get_task_cached(i);
                int64_t bb = l3sst->lkey;
                int64_t ee = l3sst->rkey;
                l3b_scan(bb, ee, addrsbuf, upbuf, downbuf);
                __mram_ptr int64_t* replyptr =
                    (__mram_ptr int64_t*)push_variable_reply_zero_copy(
                        tasklet_id, S64(addrsbuf->len + 1));
                replyptr[0] = addrsbuf->len;
                varlen_buffer_to_mram_dpu(addrsbuf, replyptr + 1);
            }
            break;
        }
#endif
#endif
            
        case B_GET_NODE_TSK: {
            init_block_with_type(b_get_node_task, b_get_node_reply);

            init_task_reader(l);
            for (int i = l; i < r; i++) {
                b_get_node_task* bgnt = (b_get_node_task*)get_task_cached(i);
                pptr addr = bgnt->addr;
                mBptr nn = (mBptr)addr.addr;
                int nnlen = nn->len;

                mpint64_t replyptr = (mpint64_t)push_variable_reply_zero_copy(
                    tasklet_id, S64(nnlen * 2 + 1));
                replyptr[0] = nnlen;
                int len1 = data_block_to_mram(nn->keys, replyptr + 1);
                int len2 = data_block_to_mram(nn->addrs, replyptr + nnlen + 1);

                IN_DPU_ASSERT(len1 == nnlen && len2 == nnlen, "bgnt! len\n");
            }
            break;
        }

#ifdef DPU_SCAN
        case B_FETCH_CHILD_TSK: {
            init_block_with_type(b_fetch_child_task, b_fetch_child_reply);
            init_task_reader(l);
            for(int i = l; i < r; i++) {
                b_fetch_child_task* bfct = (b_fetch_child_task*)
                    get_task_cached(i);
                pptr addr = bfct->addr;
                mBptr nn = (mBptr)addr.addr;
                int64_t nnlen = nn->len;
                mpint64_t replyptr = (mpint64_t)push_variable_reply_zero_copy(
                    tasklet_id, S64(nnlen + 1));
                replyptr[0] = nnlen;
                data_block_to_mram(nn->addrs, replyptr + 1);
            }
            break;
        }
#endif

#ifdef DPU_SCAN
        case B_SCAN_SEARCH_TSK: {
            init_block_with_type(b_scan_search_task, b_scan_search_reply);
            init_task_reader(l);
            int64_t bb, ee, nnlen;
            pptr addr;
            mpint64_t replyptr;
            mBptr nn;
            for(int i = l; i < r; i++) {
                b_scan_search_task* bsst = (b_scan_search_task*)get_task_cached(i);
                bb = bsst->lkey;
                ee = bsst->rkey;
                addr = bsst->addr;
                nn = (mBptr)addr.addr;
                nnlen = nn->len;
                replyptr = (mpint64_t)push_variable_reply_zero_copy(
                    tasklet_id, S64((nnlen << 1) + 1));
                nnlen = b_scan(bb, ee, nn, replyptr + 1 + nnlen, replyptr + 1);
                replyptr[0] = nnlen;
            }
            break;
        }
#endif
            
#if (defined DPU_DELETE)
        case B_REMOVE_GET_NODE_TSK: {
            init_block_with_type(b_remove_get_node_task,
                                 b_remove_get_node_reply);

            Bnode bn;

            init_task_reader(l);
            for (int i = l; i < r; i++) {
                b_remove_get_node_task* bgnt =
                    (b_remove_get_node_task*)get_task_cached(i);
                pptr addr = bgnt->addr;
                mBptr nn = (mBptr)addr.addr;
                mram_read(nn, &bn, sizeof(Bnode));

                int nnlen = bn.len;

                __mram_ptr b_remove_get_node_reply* replyptr =
                    (__mram_ptr b_remove_get_node_reply*)
                        push_variable_reply_zero_copy(tasklet_id,
                                                      S64(nnlen * 2 + 3));
                replyptr->len = nnlen;
                replyptr->left = bn.left;
                replyptr->right = bn.right;

                int len1 = data_block_to_mram(bn.keys, replyptr->vals);
                int len2 = data_block_to_mram(bn.addrs, replyptr->vals + len1);

                // garbage collection
                remove_data_blocks(bn.keys);
                remove_data_blocks(bn.addrs);
                remove_data_blocks(bn.caddrs);

                // IN_DPU_ASSERT(len1 == nnlen && len2 == nnlen, "brgnt! len\n");
                // IN_DPU_ASSERT_EXEC(len1 == nnlen && len2 == nnlen, {
                //     printf("brgnt! len %d %d %d\n", len1, len2, nnlen);
                //     for (int i = 0; i < len1; i ++) {
                //         int64_t k_ = replyptr->vals[i];
                //         int64_t v_ = replyptr->vals[i + len1];
                //         printf("%d %llx %llx\n", i, k_, v_);
                //     }
                // });

                nn->len = -1ll;  // mark as removed
                free_node(&free_list_bnode_tmp, (mpvoid)nn);
            }
            break;
        }
#endif

// #ifdef DPU_INSERT
#if (defined DPU_INIT) || (defined DPU_BUILD) || (defined DPU_INSERT) 
        case B_NEWNODE_TSK: {
            init_block_with_type(b_newnode_task, b_newnode_reply);

            init_task_reader(l);

            for (int i = l; i < r; i++) {
                b_newnode_task* bnt = (b_newnode_task*)get_task_cached(i);
                
                mBptr bnn = alloc_bn();
                mdbptr n1 = alloc_db();
                mdbptr n2 = alloc_db();
                mdbptr n3 = alloc_db();

                b_newnode(bnn, n1, n2, n3, bnt->height);

                b_newnode_reply bnr = (b_newnode_reply){
                    .addr = (pptr){.id = DPU_ID, .addr = (uint32_t)bnn}};
                push_fixed_reply(i, &bnr);
            }
#ifdef DPU_STAT_BNODE_LENGTH
            node_count_add(0, r - l);
#endif
            break;
        }
#endif

// #if (defined DPU_INSERT) || (defined DPU_DELETE)
#if (defined DPU_INIT) || (defined DPU_BUILD) || (defined DPU_INSERT) || (defined DPU_DELETE)
        case B_INSERT_TSK: {  // do not fit in wram
            init_block_with_type(b_insert_task, empty_task_reply);

            varlen_buffer* keysbuf = varlen_buffer_new(L2_SIZE);
            varlen_buffer* addrsbuf = varlen_buffer_new(L2_SIZE);

            for (int i = l; i < r; i++) {
                __mram_ptr b_insert_task* mbit =
                    (__mram_ptr b_insert_task*)get_task(i);
                int len = (int)mbit->len;
                if (len <= 0) {
                    continue;
                }
                pptr addr = mbit->addr;
                mBptr nn = (mBptr)addr.addr;
                for (int j = 0; j < len; j += L2_SIZE) {
                    int curlen = MIN(L2_SIZE, len - j);
                    m_read_single(mbit->vals + j, keysbuf->ptr, S64(curlen));
                    m_read_single(mbit->vals + j + len, addrsbuf->ptr, S64(curlen));
                    b_insert(nn, curlen, keysbuf->ptr, addrsbuf->ptr);
                }

#ifdef DPU_STAT_BNODE_LENGTH
                node_count_add(nn->len, -1);
                node_count_add(nn->len + len, 1);
#endif
            }
            break;
        }
#endif

// #if (defined DPU_INSERT) || (defined DPU_DELETE)
#if (defined DPU_BUILD) || (defined DPU_INSERT) || (defined DPU_DELETE)
        case B_SET_LR_TSK: {
            init_block_with_type(b_set_lr_task, empty_task_reply);

            init_task_reader(l);
            for (int i = l; i < r; i++) {
                b_set_lr_task* bslt = (b_set_lr_task*)get_task_cached(i);
                mBptr nn = bslt->addr.addr;
                if (!equal_pptr(bslt->left, null_pptr)) {
                    nn->left = bslt->left;
                }
                if (!equal_pptr(bslt->right, null_pptr)) {
                    nn->right = bslt->right;
                }
            }
            break;
        }
#endif

#if (defined DPU_BUILD) || (defined DPU_INSERT) || (defined DPU_DELETE)
        case CACHE_NEWNODE_TSK: {
            init_block_with_type(cache_newnode_task, empty_task_reply);

            if (tasklet_id == 0) {
                l = 0;
                r = recv_block_task_cnt;

                varlen_buffer* taskbuf = varlen_buffer_new(L2_SIZE * 2 + 2);

                for (int i = l; i < r; i++) {
                    mBptr bnn = alloc_bn();
                    mdbptr n1 = alloc_db();
                    mdbptr n2 = alloc_db();
                    mdbptr n3 = alloc_db();

                    __mram_ptr cache_newnode_task* mbit =
                        (__mram_ptr cache_newnode_task*)get_task(i);
                    int len = (int)mbit->len;
                    int siz = 2 * len + 3;
                    varlen_buffer_set_capacity(taskbuf, siz);
                    m_read(mbit, taskbuf->ptr, S64(siz));
                    pptr addr = I64_TO_PPTR(taskbuf->ptr[0]);
                    pptr caddr = I64_TO_PPTR(taskbuf->ptr[1]);
                    mBptr nn = (mBptr)addr.addr;
                    int64_t* keys = &(taskbuf->ptr[3]);
                    int64_t* addrs = &(taskbuf->ptr[3 + len]);

                    b_newnode(bnn, n1, n2, n3, nn->height - 1);
                    b_insert(bnn, len, keys, addrs);
                    pptr new_caddr = PPTR(DPU_ID, bnn);
                    cache_newnode(nn, caddr, new_caddr);
                }
            }
            break;
        }
#endif

// #ifdef DPU_INSERT
#if (defined DPU_BUILD) || (defined DPU_INSERT)
        case CACHE_INSERT_TSK: {
            init_block_with_type(cache_insert_task, empty_task_reply);

            init_task_reader(l);
            mppptr mram_buffer = (mppptr)push_variable_reply_head(0);

            for (int i = l; i < r; i++) {
                cache_insert_task* cit = (cache_insert_task*)get_task_cached(i);
                mBptr nn = (mBptr)(cit->addr.addr);
                nn = cache_find(nn, cit->key, cit->height);
                pptr ad = (pptr){.id = DPU_ID, .addr = (uint32_t)nn};
                mram_buffer[i] = ad;
            }

            barrier_wait(&exec_barrier1);

            int l2 = l, r2 = r, n = recv_block_task_cnt;
            if (r > l) {
                if (l2 != 0) {
                    l2 = get_r_mram(mram_buffer, n, l2 - 1);
                }
                // IN_DPU_ASSERT(r2 > 0, "br! rt\n");
                r2 = get_r_mram(mram_buffer, n, r2 - 1);
            }

#ifdef KHB_DEBUG
            if (tasklet_id == 0) {
                for (int p = 0; p < n; p++) {
                    bool eq = true;
                    for (int j = p + 1; j < n; j++) {
                        if (!equal_pptr(mram_buffer[p], mram_buffer[j])) {
                            eq = false;
                        }
                        IN_DPU_ASSERT_EXEC((!equal_pptr(mram_buffer[p], mram_buffer[j]) || eq == true), {
                            for (int i = 0; i < n; i ++) {
                                pptr ad = mram_buffer[i];
                                printf("ad[%d]=%llx\n", i, PPTR_TO_I64(ad));
                            }
                        });
                    }
                }
            }
#endif

            init_task_reader(l2);
            for (int i = l2; i < r2; i++) {
                cache_insert_task* cit = (cache_insert_task*)get_task_cached(i);
                int64_t key = cit->key;
                pptr t_addr = cit->t_addr;
                pptr ad = mram_buffer[i];
                mBptr nn = (mBptr)ad.addr;
                b_insert(nn, 1, &key, (int64_t*)&t_addr);
            }
            break;
        }
#endif

#ifdef DPU_DELETE
        case CACHE_MULTI_INSERT_TSK: {
            init_block_with_type(cache_multi_insert_task, empty_task_reply);

            if (tasklet_id == 0) {
                l = 0;
                r = recv_block_task_cnt;

                varlen_buffer* taskbuf = varlen_buffer_new(L2_SIZE * 2 + 3);
                for (int i = l; i < r; i++) {
                    __mram_ptr cache_multi_insert_task* cmit =
                        (__mram_ptr cache_multi_insert_task*)get_task(i);
                    int len = (int)cmit->len;
                    if (len == 0) {
                        continue;
                    }
                    int siz = 2 * len + 3;
                    varlen_buffer_set_capacity(taskbuf, siz);
                    m_read(cmit, taskbuf->ptr, S64(siz));
                    pptr addr = I64_TO_PPTR(taskbuf->ptr[0]);
                    mBptr nn = (mBptr)addr.addr;
                    if (nn->len == -1) {
                        continue;
                    }
                    int64_t* keys = &(taskbuf->ptr[3]);
                    int64_t* addrs = &(taskbuf->ptr[3 + len]);
                    int64_t height = taskbuf->ptr[2];
                    cache_multi_insert(nn, len, keys, addrs, height);
                }
            }
            break;
        }
#endif

#ifdef DPU_DELETE
        case CACHE_REMOVE_TSK: {
            init_block_with_type(cache_remove_task, empty_task_reply);

            mppptr mram_buffer = (mppptr)push_variable_reply_head(0);
            int n = recv_block_task_cnt;
            printf("l=%d\n", S64(n));

            init_task_reader(l);
            for (int i = l; i < r; i ++) {
                cache_remove_task* crt = (cache_remove_task*)get_task_cached(i);
                mBptr nn = (mBptr)(crt->addr.addr);
                pptr ad;
                int nnlen = nn->len;
                if (nnlen > 0) {
                    nn = cache_find(nn, crt->key, crt->height);
                    ad = (pptr){.id = DPU_ID, .addr = (uint32_t)nn};
                } else {
                    ad = null_pptr;
                }
                mram_buffer[i] = ad;
            }

            barrier_wait(&exec_barrier1);

            send_varlen_task_size[0] = S64(n);
            if (tasklet_id == 0) {
                l = 0;
                r = recv_block_task_cnt;
                init_task_reader(l);
                for (int i = l; i < r; i++) {
                    cache_remove_task* crt =
                        (cache_remove_task*)get_task_cached(i);
                    mBptr nn = (mBptr)(crt->addr.addr);
                    int nnlen = nn->len;
                    if (nnlen <= 0) continue;
                    pptr ad2 = mram_buffer[i];
// #ifdef KHB_DEBUG
//                     nn = cache_find(nn, crt->key, crt->height);
//                     pptr ad = (pptr){.id = DPU_ID, .addr = nn};
//                     // m_read_single(mram_buffer + i, &ad2, sizeof(pptr));
//                     IN_DPU_ASSERT_EXEC(equal_pptr(ad, ad2), {
//                         int nnlen = nn->len;
//                         printf("crt! inv ad=%llx ad2=%llx i=%d r=%d nnlen=%d\n", PPTR_TO_I64(ad), PPTR_TO_I64(ad2), i, r, nnlen);
//                         int n = MIN(r, 10);
//                         for (int i = 0; i < n; i ++) {
//                             pptr ad = mram_buffer[i];
//                             printf("ad[%d]=%llx\n", i, PPTR_TO_I64(ad));
//                         }
//                     });
// #endif
                    nn = (mBptr)(ad2.addr);
                    b_remove(nn, 1, &crt->key);
                }

                move_free_list(&free_list_bnode_tmp, &free_list_bnode);
            }
            break;
        }
#endif

#if (defined DPU_BUILD) || (defined DPU_INSERT)
        case CACHE_TRUNCATE_TSK: {
            init_block_with_type(cache_truncate_task, empty_task_reply);

            varlen_buffer* keysbuf = varlen_buffer_new(L2_SIZE);
            varlen_buffer* addrsbuf = varlen_buffer_new(L2_SIZE);
            varlen_buffer* caddrsbuf = varlen_buffer_new(L2_SIZE);

            init_task_reader(l);
            if (tasklet_id == 0) {
                l = 0;
                r = recv_block_task_cnt;
                for (int i = l; i < r; i++) {
                    cache_truncate_task* ctt =
                        (cache_truncate_task*)get_task_cached(i);
                    mBptr nn = (mBptr)(ctt->addr.addr);
                    cache_truncate(nn, ctt->key, ctt->height, keysbuf, addrsbuf,
                                   caddrsbuf);
                }
            }
            break;
        }
#endif

#if (defined DPU_BUILD) || (defined DPU_INSERT) || (defined DPU_DELETE)
        case CACHE_INIT_REQ_TSK: {
            init_block_with_type(cache_init_request_task,
                                 cache_init_request_reply);

            if (tasklet_id == 0) {
                int len = circnt - 1;
                __mram_ptr cache_init_request_reply* replyptr =
                    (__mram_ptr cache_init_request_reply*)
                        push_variable_reply_zero_copy(tasklet_id,
                                                      S64(2 * len + 1));
                // IN_DPU_ASSERT(len < 1000, "cirt! of\n");
                replyptr->len = len;
                for (int i = 0; i < len; i++) {
                    cache_init_record cir;
                    mram_read(cirbuffer + i + 1, &cir,
                              sizeof(cache_init_record));
                    replyptr->vals[2 * i] = PPTR_TO_I64(cir.addr);
                    replyptr->vals[2 * i + 1] = PPTR_TO_I64(cir.request);
                }
                circnt = 1;  // set to empty
            }
            break;
        }
#endif

#ifdef DPU_PREDECESSOR
        case B_SEARCH_TSK: {
            init_block_with_type(b_search_task, b_search_reply);

            varlen_buffer* taskbuf = varlen_buffer_new(L2_SIZE + 2);
            varlen_buffer* replybuf = varlen_buffer_new(1);
            varlen_buffer* replykeysbuf = varlen_buffer_new(1);

            for (int i = l; i < r; i++) {
                __mram_ptr b_search_task* mbst =
                    (__mram_ptr b_search_task*)get_task(i);
                int len = (int)mbst->len;
                int siz = len + 2;
                varlen_buffer_set_capacity(taskbuf, siz);
                m_read(mbst, taskbuf->ptr, S64(siz));

                pptr addr = I64_TO_PPTR(taskbuf->ptr[0]);
                int64_t* keys = &(taskbuf->ptr[2]);

                mpuint8_t replyptr =
                    push_variable_reply_zero_copy(tasklet_id, S64(len + 1));

                varlen_buffer_set_capacity(replybuf, len + 1);
                varlen_buffer_set_capacity(replykeysbuf, len);

                replybuf->ptr[0] = len;
                int64_t* repkeys = replykeysbuf->ptr;
                pptr* repaddrs = (pptr*)replybuf->ptr + 1;
                for (int j = 0; j < len; j++) {
                    repaddrs[j] = addr;
                }

                nested_search(len, keys, repkeys, repaddrs, NULL, NULL, -1);

                m_write(replybuf->ptr, replyptr, S64(len + 1));
            }
            break;
        }
#endif

#ifdef DPU_DELETE
        case B_REMOVE_TSK: {
            init_block_with_type(b_remove_task, empty_task_reply);

            varlen_buffer* taskbuf = varlen_buffer_new(L2_SIZE + 2);
            varlen_buffer* replybuf = varlen_buffer_new(1);
            varlen_buffer* replykeysbuf = varlen_buffer_new(1);

            for (int i = l; i < r; i++) {
                __mram_ptr b_remove_task* mbrt =
                    (__mram_ptr b_remove_task*)get_task(i);
                int len = (int)mbrt->len;
                int siz = len + 2;
                varlen_buffer_set_capacity(taskbuf, siz);
                m_read(mbrt, taskbuf->ptr, S64(siz));
                pptr addr = I64_TO_PPTR(taskbuf->ptr[0]);
                mBptr nn = (mBptr)addr.addr;
                int64_t* keys = &(taskbuf->ptr[2]);

                b_remove(nn, len, keys);
                // int ret = b_remove(nn, len, keys);
                // IN_DPU_ASSERT(ret == len, "br! ne\n");
            }
            break;
        }
#endif

#ifdef DPU_PREDECESSOR
        case B_FIXED_SEARCH_TSK: {
            // init_send_block(tasklet_id, FIXED_LENGTH,
            //                 sizeof(b_fixed_search_task),
            //                 sizeof(b_fixed_search_reply));
            init_block_with_type(b_fixed_search_task, b_fixed_search_reply);
            init_task_reader(l);
            for (int i = l; i < r; i++) {
                b_fixed_search_task* bfst =
                    (b_fixed_search_task*)get_task_cached(i);
                pptr addr = bfst->addr;
                int64_t key = bfst->key;
                b_fixed_search_reply bfsr;
                if (!in_bbuffer(addr.addr)) {
                    IN_DPU_ASSERT(in_pbuffer(addr.addr), "bfst! inv\n");
                    bfsr = (b_fixed_search_reply){.addr = addr};
                } else {
                    pptr repaddr;
                    int64_t repkey;
                    b_search(addr.addr, 1, &key, &repkey, &repaddr, NULL);
                    bfsr = (b_fixed_search_reply){.addr = repaddr};
                }
                push_fixed_reply(i, &bfsr);
            }
            break;
        }
#endif

// #if (defined DPU_INSERT) || (defined DPU_PREDECESSOR)
#ifdef DPU_PREDECESSOR
        case B_SEARCH_WITH_PATH_TSK: {
            init_block_with_type(b_search_with_path_task,
                                 b_search_with_path_reply);

            varlen_buffer* taskbuf = varlen_buffer_new(2 * L2_SIZE + 2);
            varlen_buffer* replybuf = varlen_buffer_new(1);
            varlen_buffer* replykeysbuf = varlen_buffer_new(1);
            varlen_buffer* replyaddrsbuf = varlen_buffer_new(1);

            for (int i = l; i < r; i++) {
                __mram_ptr b_search_with_path_task* mbswpt =
                    (__mram_ptr b_search_with_path_task*)get_task(i);
                int len = (int)mbswpt->len;
                int siz = b_search_with_path_task_siz(len);
                varlen_buffer_set_capacity(taskbuf, siz);
                m_read(mbswpt, taskbuf->ptr, S64(siz));

                pptr addr = I64_TO_PPTR(taskbuf->ptr[0]);
                // mBptr nn = (mBptr)addr.addr;
                // IN_DPU_ASSERT(in_bbuffer(nn) && nn->height < L2_HEIGHT,
                //               "bswpt! nnht\n");

                int64_t* keys = &(taskbuf->ptr[2]);
                uint64_t* heights = (uint64_t*)(taskbuf->ptr + 2 + len);

                int max_anslen = 0;
                for (int j = 0; j < len; j++) {
                    IN_DPU_ASSERT(heights[j] == L2_HEIGHT || heights[j] == 1,
                                  "bswpt! ht\n");
                    max_anslen +=
                        (heights[j] == 1) ? (heights[j] + 1) : heights[j];
                }

                varlen_buffer_set_capacity(replykeysbuf, len);
                varlen_buffer_set_capacity(replyaddrsbuf, len);
                varlen_buffer_set_capacity(replybuf, max_anslen + 5);

                int64_t* repkeys = replykeysbuf->ptr;
                pptr* repaddrs = (pptr*)(replyaddrsbuf->ptr);
                for (int j = 0; j < len; j++) {
                    repaddrs[j] = addr;
                }

                int anslen = nested_search(
                    len, keys, repkeys, repaddrs, heights,
                    (offset_pptr*)(&(replybuf->ptr[1])), max_anslen);

                replybuf->ptr[0] = anslen;
                mpuint8_t replyptr =
                    push_variable_reply_zero_copy(tasklet_id, S64(anslen + 1));

                m_write(replybuf->ptr, replyptr, S64(anslen + 1));
            }
            break;
        }
#endif

// #ifdef DPU_INSERT
#if (defined DPU_BUILD) || (defined DPU_INSERT)
        case B_TRUNCATE_TSK: {  // cached
            init_block_with_type(b_truncate_task, b_truncate_reply);

            init_task_reader(l);

            for (int i = l; i < r; i++) {
                b_truncate_task* btt = (b_truncate_task*)get_task_cached(i);

                pptr addr = btt->addr;
                mBptr nn = (mBptr)addr.addr;

                IN_DPU_ASSERT(b_length_check(nn), "btt! nn\n");

                Bnode bn;
                mram_read(nn, &bn, sizeof(Bnode));

                int64_t key = btt->key;
                // IN_DPU_ASSERT(key > INT64_MIN, "btt! i64_min\n");

                int nnlen = bn.len;

                __mram_ptr b_truncate_reply* replytsk =
                    (__mram_ptr b_truncate_reply*)push_variable_reply_head(
                        tasklet_id);

                IN_DPU_ASSERT((send_varlen_task_size[tasklet_id] + 2 +
                               3 * nnlen) < MAX_TASK_BUFFER_SIZE_PER_TASKLET,
                              "btt! of\n");

                int replen = b_filter_cache_mram(
                    nn, key, INT64_MAX, replytsk->vals, replytsk->vals + nnlen,
                    replytsk->vals + 2 * nnlen);

                // #ifdef KHB_DEBUG
                //                 for (int j = 0; j < replen; j++) {
                //                     int64_t iaddr = replytsk->vals[nnlen +
                //                     j]; pptr addr = I64_TO_PPTR(iaddr); if
                //                     (!valid_pptr(addr)) {
                //                         for (int k = 0; k < replen; k++) {
                //                             int64_t iaddr =
                //                             replytsk->vals[nnlen + j];
                //                             printf("%d %llx\n", k, iaddr);
                //                         }
                //                         IN_DPU_ASSERT(false, "btt! inv
                //                         addr\n");
                //                     }
                //                 }
                // #endif

                replytsk->len = replen;
                replytsk->right = nn->right;
                for (int j = 0; j < replen; j++) {
                    replytsk->vals[replen + j] = replytsk->vals[nnlen + j];
                }
                push_variable_reply_commit(tasklet_id, S64(2 + 2 * replen));

                mpint64_t mram_buffer =
                    (mpint64_t)push_variable_reply_head(tasklet_id);
                // IN_DPU_ASSERT((send_varlen_task_size[tasklet_id] + 3 * nnlen) <
                //                   MAX_TASK_BUFFER_SIZE_PER_TASKLET,
                //               "btt! of2\n");

                int newlen = b_filter_cache_mram(
                    nn, INT64_MIN, key - 1, mram_buffer, mram_buffer + nnlen,
                    mram_buffer + nnlen * 2);

                data_block_from_mram(bn.keys, mram_buffer, newlen);
                data_block_from_mram(bn.addrs, mram_buffer + nnlen, newlen);
                data_block_from_mram(bn.caddrs, mram_buffer + nnlen * 2,
                                     newlen);

#ifdef DPU_STAT_BNODE_LENGTH
                node_count_add(nn->len, -1);
                node_count_add(newlen, 1);
#endif
                nn->len = newlen;
            }

#ifdef KHB_INFO
            int mem_consumption =
                S64(repkeysbuf->capacity + repaddrsbuf->capacity);
            printf("tid=%d mem_consump=%d\n", tasklet_id, mem_consumption);
#endif
            break;
        }
#endif

// #ifdef DPU_INSERT
#if (defined DPU_INIT) || (defined DPU_BUILD) || (defined DPU_INSERT)
        case P_NEWNODE_TSK: {
            init_block_with_type(p_newnode_task, p_newnode_reply);

            init_task_reader(l);

            for (int i = l; i < r; i++) {
                p_newnode_task* pnt = (p_newnode_task*)get_task_cached(i);
                mPptr pn = alloc_pn();
                p_newnode(pnt->key, pnt->value, pnt->height, pn);
                p_newnode_reply rep = (p_newnode_reply){
                    .addr = {.id = DPU_ID, .addr = (uint32_t)pn}};
                push_fixed_reply(i, &rep);
            }
            break;
        }
#endif

#if (defined DPU_PREDECESSOR) || (defined DPU_SCAN)
        case P_GET_KEY_TSK: {
            init_block_with_type(p_get_key_task, p_get_key_reply);

            init_task_reader(l);
            for (int i = l; i < r; i++) {
                p_get_key_task* pgkt = (p_get_key_task*)get_task_cached(i);
                p_get_key_reply rep = {.key = p_get_key(pgkt->addr),
                                       .value = p_get_value(pgkt->addr)};
                push_fixed_reply(i, &rep);
            }
            break;
        }
#endif

#ifdef DPU_GET_UPDATE
        case P_GET_TSK: {
            init_block_with_type(p_get_task, p_get_reply);

            init_task_reader(l);
            for (int i = l; i < r; i++) {
                p_get_task* pgt = (p_get_task*)get_task_cached(i);
                pptr addr = p_get(pgt->key);
                p_get_reply rep;
                if (equal_pptr(addr, null_pptr)) {
                    rep = (p_get_reply){.key = INT64_MIN, .value = INT64_MIN};
                } else {
                    rep = (p_get_reply){.key = p_get_key(addr),
                                        .value = p_get_value(addr)};
                }
                push_fixed_reply(i, &rep);
            }
            break;
        }
#endif

// set to GET_UPDATE to accelerate insert. should be in DPU_INSERT.
// as it's only used when doing insert
#ifdef DPU_GET_UPDATE 
        case P_UPDATE_TSK: {
            init_block_with_type(p_update_task, p_update_reply);

            init_task_reader(l);
            for (int i = l; i < r; i++) {
                p_update_task* pgt = (p_update_task*)get_task_cached(i);
                pptr addr = p_get(pgt->key);
                p_update_reply rep;
                if (equal_pptr(addr, null_pptr)) {
                    rep = (p_update_reply){.valid = 0};
                } else {
                    rep = (p_update_reply){.valid = 1};
                    mPptr nn = addr.addr;
                    nn->value = pgt->value;
                }
                push_fixed_reply(i, &rep);
            }
            break;
        }
#endif

#ifdef DPU_PREDECESSOR
        case P_GET_HEIGHT_TSK: {
            init_block_with_type(p_get_height_task, p_get_height_reply);

            init_task_reader(l);
            for (int i = l; i < r; i++) {
                p_get_height_task* pght =
                    (p_get_height_task*)get_task_cached(i);
                p_get_height_reply rep = {.height = p_get_height(pght->key)};
                push_fixed_reply(i, &rep);
            }
            break;
        }
#endif

        // may not work
        case STATISTICS_TSK: {
#ifdef DPU_STATISTICS
            init_block_with_type(statistic_task, empty_task_reply);
            if (tasklet_id == 0) {
                print_statistics();
            }
#endif
            break;
        }

        default: {
            printf("TT = %lld\n", recv_block_task_type);
            IN_DPU_ASSERT(false, "WTT\n");
            break;
        }
    }
    finish_reply(recv_block_task_cnt, tasklet_id);
}

void init() {
#ifdef L3_SKIP_LIST
    // #ifdef DPU_PREDECESSOR
    //     if (DPU_ID == 0 && me() == 0) {
    //         printf("l3_size=%d b_header_size=%d db_size=%d p_size=%d
    //         ht_size=%d\n",
    //                l3cnt, bcnt * sizeof(Bnode), dbcnt * sizeof(data_block),
    //                pcnt * sizeof(Pnode), htcnt * sizeof(int64_t));
    //     }
    // #endif
    switch (recv_block_task_type) {
        case L3_INIT_TSK:
        case L3_INSERT_TSK:
        case L3_SEARCH_TSK:
        case L3_REMOVE_TSK: {
            bufferA_shared =
                mem_alloc(sizeof(mL3ptr) * NR_TASKLETS * MAX_L3_HEIGHT);
            bufferB_shared =
                mem_alloc(sizeof(mL3ptr) * NR_TASKLETS * MAX_L3_HEIGHT);
            max_height_shared = mem_alloc(sizeof(int8_t) * NR_TASKLETS);
            newnode_size = mem_alloc(sizeof(uint32_t) * NR_TASKLETS);
            for (int i = 0; i < NR_TASKLETS; i++) {
                max_height_shared[i] = 0;
            }
            break;
        }
    }
#else
// #ifdef DPU_PREDECESSOR
//     if (DPU_ID == 0 && me() == 0) {
//         printf("l3_size=%d b_header_size=%d db_size=%d p_size=%d ht_size=%d\n",
//                l3bcnt * sizeof(L3Bnode), bcnt * sizeof(Bnode),
//                dbcnt * sizeof(data_block), pcnt * sizeof(Pnode),
//                htcnt * sizeof(int64_t));
//     }
// #endif
    switch (recv_block_task_type) {
        case L3_INIT_TSK:
        case L3_INSERT_TSK:
        case L3_SEARCH_TSK:
        case L3_REMOVE_TSK: {
            L3_lfts = mem_alloc(sizeof(int) * NR_TASKLETS);
            L3_rts = mem_alloc(sizeof(int) * NR_TASKLETS);

            int64_t *addr =
                mem_alloc(4 * sizeof(int64_t) * L3_TEMP_BUFFER_SIZE_FULL +
                          4 * sizeof(int64_t) * L3_TEMP_BUFFER_SIZE_FULL);
            mod_keys = addr;
            mod_keys2 = mod_keys + L3_TEMP_BUFFER_SIZE_FULL;
            mod_values = mod_keys2 + L3_TEMP_BUFFER_SIZE_FULL;
            mod_values2 = mod_values + L3_TEMP_BUFFER_SIZE_FULL;
            mod_addrs = mod_values2 + L3_TEMP_BUFFER_SIZE_FULL;
            mod_addrs2 = mod_addrs + L3_TEMP_BUFFER_SIZE_FULL;
            mod_type = mod_addrs2 + L3_TEMP_BUFFER_SIZE_FULL;
            mod_type2 = mod_type + L3_TEMP_BUFFER_SIZE_FULL;
        }
    }

#endif
}

int main() {
    run();
}
