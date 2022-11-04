#pragma once

#include <iostream>
#include <cstdio>
#include <string>
// #include "task_host.hpp"
#include "compile.hpp"
#include "timer.hpp"
#include "host.hpp"
#include "util.hpp"
#include <parlay/primitives.h>
#include <parlay/range.h>
#include <parlay/sequence.h>
#include <mutex>

using namespace std;

extern int batch_round;
// #define REMOVE_DEBUG

static inline int hh(int64_t key, uint64_t height, uint64_t M) {
    uint64_t v = parlay::hash64((uint64_t)key) + height;
    v = parlay::hash64(v);
    return v % M;
}

static inline int hash_to_dpu(int64_t key, uint64_t height, uint64_t M) {
    return hh(key, height, M);
}

mutex mut;

int push_pull_limit = L2_SIZE;
extern bool print_debug;
extern int64_t epoch_number;

extern dpu_set_t dpu_set, dpu;
extern uint32_t each_dpu;

int max_l3_height; // setting max height

pptr op_addrs[BATCH_SIZE];
pptr op_addrs2[BATCH_SIZE];
int32_t op_taskpos[BATCH_SIZE];
int32_t op_taskpos2[BATCH_SIZE];
int l2_root_count[NR_DPUS];

pptr l1_init_node = null_pptr;

struct dpu_memory_regions {
    uint32_t bbuffer_start;
    uint32_t bbuffer_end;
    uint32_t pbuffer_start;
    uint32_t pbuffer_end;
} dmr;

static inline bool in_bbuffer(uint32_t addr) {
    return addr >= dmr.bbuffer_start && addr < dmr.bbuffer_end;
}

static inline bool in_pbuffer(uint32_t addr) {
    return addr >= dmr.pbuffer_start && addr < dmr.pbuffer_end;
}

static inline bool valid_b_pptr(const pptr& x) {
    return valid_pptr(x, nr_of_dpus) && in_bbuffer(x.addr);
}

static inline bool valid_p_pptr(const pptr& x) {
    return valid_pptr(x, nr_of_dpus) && in_pbuffer(x.addr);
}

inline void build_cache() {
    for (int round = 0; round < 2; round++) {
        auto io = alloc_io_manager();
        ASSERT(io == io_managers[0]);
        io->init(epoch_number++);

        auto cache_init_request_batch = io->alloc_task_batch(
            direct, fixed_length, variable_length, CACHE_INIT_REQ_TSK,
            sizeof(cache_init_request_task),
            sizeof(
                cache_init_request_reply));  // !!! ??? !!! wrong reply length
        time_nested("cache init request taskgen", [&]() {
            parfor_wrap(0, nr_of_dpus, [&](size_t i) {
                cache_init_request_task *cirt =
                    (cache_init_request_task *)cache_init_request_batch
                        ->push_task_zero_copy(i, -1, false, NULL);
                *cirt = (cache_init_request_task){.nothing = 0};
            });
            io->finish_task_batch();
        });

        time_nested("cache init request", [&]() { ASSERT(io->exec()); });

        auto cache_init_len = parlay::sequence<int64_t>(nr_of_dpus);
        // parlay::sequence<int64_t> cache_init_addrs;
        // parlay::sequence<pptr> cache_init_requests;
        // parlay::sequence<int> cache_init_taskoffset;

        for (int i = 0; i < nr_of_dpus; i++) {
            cache_init_request_reply *cirr =
                (cache_init_request_reply *)cache_init_request_batch->get_reply(
                    0, i);
            cache_init_len[i] = cirr->len;
        }
        auto cache_init_len_prefix_pair = parlay::scan(cache_init_len);

        int64_t cache_init_len_total = cache_init_len_prefix_pair.second;
        auto cache_init_len_prefix_sum = cache_init_len_prefix_pair.first;

        ASSERT(round == 0 || cache_init_len_total == 0);

        if (cache_init_len_total == 0) {
            io->reset();
            io = nullptr;
            break;
        }

        auto cache_init_addrs = parlay::sequence<pptr>(cache_init_len_total);
        auto cache_init_requests = parlay::sequence<pptr>(cache_init_len_total);
        auto cache_init_taskoffset =
            parlay::sequence<int>(cache_init_len_total);

        parfor_wrap(0, nr_of_dpus, [&](size_t i) {
            cache_init_request_reply *cirr =
                (cache_init_request_reply *)cache_init_request_batch->get_reply(
                    0, i);

#ifdef KHB_CPU_DEBUG
            if ((int)i == (int)nr_of_dpus - 1) {
                assert(cache_init_len_total ==
                       cache_init_len_prefix_sum[i] + cirr->len);
            } else {
                assert(cache_init_len_prefix_sum[i + 1] ==
                       cache_init_len_prefix_sum[i] + cirr->len);
            }
#endif
            for (int j = 0; j < cirr->len; j++) {
                cache_init_addrs[cache_init_len_prefix_sum[i] + j] =
                    int64_to_pptr(cirr->vals[j * 2]);
                cache_init_requests[cache_init_len_prefix_sum[i] + j] =
                    int64_to_pptr(cirr->vals[j * 2 + 1]);
            }
        });

        io->reset();
        io = nullptr;

        // for (int i = 0; i < cache_init_len_total; i++) {
        //     printf("addr=%lx\trequest=%lx\n",
        //            pptr_to_int64(cache_init_addrs[i]),
        //            pptr_to_int64(cache_init_requests[i]));
        // }

        io = alloc_io_manager();
        ASSERT(io == io_managers[0]);
        io->init(epoch_number++);

        auto cache_init_getnode_batch = io->alloc_task_batch(
            direct, fixed_length, variable_length, B_GET_NODE_TSK,
            sizeof(b_get_node_task),
            sizeof(b_get_node_reply));  // !!! ??? !!! wrong reply length
        parfor_wrap(0, cache_init_len_total, [&](size_t i) {
            pptr request = cache_init_requests[i];

            b_get_node_task *bgnt =
                (b_get_node_task *)
                    cache_init_getnode_batch->push_task_zero_copy(
                        request.id, -1, true, &(cache_init_taskoffset[i]));
            *bgnt = (b_get_node_task){.addr = request};
        });
        io->finish_task_batch();

        time_nested("cache init getnode", [&]() { ASSERT(io->exec()); });

        auto io2 = alloc_io_manager();
        ASSERT(io2 == io_managers[1]);
        io2->init(epoch_number++);

        auto cache_init_newnode = io2->alloc_task_batch(
            direct, variable_length, fixed_length, CACHE_NEWNODE_TSK, -1, 0);
        {
            parfor_wrap(0, cache_init_len_total, [&](size_t i) {
                pptr addr = cache_init_addrs[i];
                pptr request = cache_init_requests[i];

                b_get_node_reply *bgnr =
                    (b_get_node_reply *)cache_init_getnode_batch->get_reply(
                        cache_init_taskoffset[i], request.id);
                int nnlen = bgnr->len;

                // int64_t *rep_vals = bgnr->vals;
                // int64_t *rep_keys = rep_vals;
                // pptr *rep_addrs = (pptr *)(rep_vals + nnlen);

                cache_newnode_task *cnt =
                    (cache_newnode_task *)
                        cache_init_newnode->push_task_zero_copy(
                            addr.id, S64(3 + nnlen * 2), true, NULL);

                cnt->addr = addr;
                cnt->caddr = request;
                cnt->len = nnlen;
                memcpy(cnt->vals, bgnr->vals, S64(2 * nnlen));
            });
            io2->finish_task_batch();
        }

        time_nested("cache init newnode", [&]() { ASSERT(!io2->exec()); });

        io->reset();
        io = nullptr;
        io2->reset();
        io2 = nullptr;
    }
}

void init_skiplist(uint32_t l3_height) {
    dpu_binary_switch_to(dpu_binary::insert_binary);
    
    ASSERT(l3_height > 0);
    max_l3_height = l3_height;

    printf("\n********** INIT SKIP LIST **********\n");

    pptr l3node = null_pptr;
    pptr l2nodes[L2_HEIGHT] = {null_pptr};
    pptr l1node = null_pptr;

    // insert nodes
    time_nested("nodes", [&]() {
        auto io = alloc_io_manager();
        ASSERT(io == io_managers[0]);
        io->init(epoch_number++);

        auto L3_init_batch = io->alloc_task_batch(
            broadcast, fixed_length, fixed_length, L3_INIT_TSK,
            sizeof(L3_init_task), sizeof(L3_init_reply));
        {
            auto batch = L3_init_batch;
            L3_init_task *tit =
                (L3_init_task *)batch->push_task_zero_copy(-1, -1, false);
            *tit = (L3_init_task){.key = INT64_MIN, .height = l3_height};
            io->finish_task_batch();
        }

        auto L2_init_batch = io->alloc_task_batch(
            direct, fixed_length, fixed_length, B_NEWNODE_TSK,
            sizeof(b_newnode_task), sizeof(b_newnode_task));
        {
            auto batch = L2_init_batch;
            for (int ht = 0; ht < L2_HEIGHT; ht++) {
                int target = hash_to_dpu(INT64_MIN, ht, nr_of_dpus);
                b_newnode_task *bnt =
                    (b_newnode_task *)batch->push_task_zero_copy(
                        target, -1, true, op_taskpos + ht);
                *bnt = (b_newnode_task){.height = ht};
            }
            io->finish_task_batch();
        }

        auto L1_init_batch = io->alloc_task_batch(
            direct, fixed_length, fixed_length, P_NEWNODE_TSK,
            sizeof(p_newnode_task), sizeof(p_newnode_reply));
        {
            auto batch = L1_init_batch;
            int target = hash_to_dpu(INT64_MIN, 0, nr_of_dpus);
            p_newnode_task *pnt =
                (p_newnode_task *)batch->push_task_zero_copy(target, -1, false);
            pnt->key = INT64_MIN;
            pnt->height = l3_height + L2_HEIGHT;
            io->finish_task_batch();
        }

        // io->send_task();
        // io->print_all_buffer(true);
        io->exec();
        // exit(0);
        // io->print_all_buffer();

        {
            auto batch = L3_init_batch;
            L3_init_reply *rep = (L3_init_reply *)batch->get_reply(0, -1);
            l3node = rep->addr;
        }
        {
            auto batch = L2_init_batch;
            for (int ht = 0; ht < L2_HEIGHT; ht++) {
                int target = hash_to_dpu(INT64_MIN, ht, nr_of_dpus);
                b_newnode_reply *rep =
                    (b_newnode_reply *)batch->get_reply(op_taskpos[ht], target);
                l2nodes[ht] = rep->addr;
            }
        }
        {
            auto batch = L1_init_batch;
            int target = hash_to_dpu(INT64_MIN, 0, nr_of_dpus);
            p_newnode_reply *rep =
                (p_newnode_reply *)batch->get_reply(0, target);
            l1node = rep->addr;
        }
        io->io_manager_state = idle;
    });

    // l1_init_node = l1node;

    string newline("\n");
    printf("l3node=%llx\n", pptr_to_int64(l3node));
    for (int ht = 0; ht < L2_HEIGHT; ht++) {
        printf("l2nodes[%d]=%llx\n", ht, pptr_to_int64(l2nodes[ht]));
    }
    printf("l1node=%llx\n", pptr_to_int64(l1node));
    memset(l2_root_count, 0, sizeof(l2_root_count));
    l2_root_count[l2nodes[L2_HEIGHT - 1].id] ++;
    // exit(0);

    // build up down pointers
    time_nested("ud ptrs", [&]() {
        auto io = alloc_io_manager();
        ASSERT(io == io_managers[0]);
        io->init(epoch_number++);
        auto L3_build_d_batch =
            io->alloc_task_batch(broadcast, fixed_length, fixed_length,
                                 L3_BUILD_D_TSK, sizeof(L3_build_d_task), 0);
        {
            auto batch = L3_build_d_batch;
            L3_build_d_task *tbdt =
                (L3_build_d_task *)batch->push_task_zero_copy(-1, -1, false);
            *tbdt = (L3_build_d_task){.addr = l3node,
                                      .down = l2nodes[L2_HEIGHT - 1]};
            io->finish_task_batch();
        }
        auto L2_insert_batch = io->alloc_task_batch(
            direct, variable_length, fixed_length, B_INSERT_TSK, -1, 0);
        {
            auto batch = L2_insert_batch;
            for (int i = 0; i < L2_HEIGHT; i++) {
                int len = 1;
                int target = hash_to_dpu(INT64_MIN, i, nr_of_dpus);
                b_insert_task *bit =
                    (b_insert_task *)batch->push_task_zero_copy(
                        target, S64(2 + 1 * 2), false);
                bit->addr = l2nodes[i];
                bit->len = len;
                bit->vals[0] = LLONG_MIN;
                pptr down = (i == 0) ? l1node : l2nodes[i - 1];
                memcpy(&(bit->vals[1]), &down, S64(1));
            }
            io->finish_task_batch();
        }
        io->exec();
    });
    build_cache();
}

void get(int length, int64_t *keys) {
    dpu_binary_switch_to(dpu_binary::get_update_binary);

    auto io = alloc_io_manager();
    ASSERT(io == io_managers[0]);
    io->init(epoch_number++);
    // for (int i = 0; i < length; i ++) {
    //     printf("keys[%d]=%llx\n", i, keys[i]);
    // }
    auto target_addr = parlay::tabulate(length, [&](int i) {
        return hash_to_dpu(keys[i], 0, nr_of_dpus);
    });
    auto get_batch =
        io->alloc_task_batch(direct, fixed_length, fixed_length, P_GET_TSK,
                             sizeof(p_get_task), sizeof(p_get_reply));
    time_nested("taskgen", [&]() {
        // get_batch->push_task_from_array(length, [&](uint32_t i) -> task_idx {
        //     return (task_idx){.id = target_addr[i], .size = -1, .offset = i};
        // }, [&](int i, int offset, uint8_t* addr) {
        //     p_get_task* pgt = (p_get_task*)addr;
        //     *pgt = (p_get_task){.key = keys[i]};
        //     op_taskpos[i] = offset;
        // });
        parfor_wrap(0, length, [&](size_t i) {
            int64_t key = keys[i];
            // int target = hash_to_dpu(key, 0, nr_of_dpus);
            p_get_task *pgt = (p_get_task *)batch->push_task_zero_copy(
                target_addr[i], -1, true, op_taskpos + i);
            *pgt = (p_get_task){.key = key};
        });
        io->finish_task_batch();
    });

    time_nested("exec", [&]() { io->exec(); });

    time_nested("get result", [&]() {
        auto batch = get_batch;
        parfor_wrap(0, length, [&](size_t i) {
            int64_t key = keys[i];
            // int target = hash_to_dpu(key, 0, nr_of_dpus);
            p_get_reply *pgr =
                (p_get_reply *)batch->get_reply(op_taskpos[i], target_addr[i]);
            if (pgr->addr.id == INVALID_DPU_ID) {
                op_results[i] = 0;
            } else {
                op_results[i] = 1;
            }
        });
    });

    io->io_manager_state = idle;
}

template <class T>
inline void print_parlay_sequence(const T& x) {
    for (size_t i = 0; i < x.size(); i ++) {
        printf("[%lu] = %d\n", i, x[i]);
    }
}

enum predecessor_type { predecessor_insert, predecessor_only };

template <class TT>
struct copy_scan {
  using T = TT;
  copy_scan() : identity(0) {}
  T identity;
  static T f(T a, T b) { return (b == -1) ? a : b; }
};

template<class TT>
struct add_scan {
    using T = TT;
    add_scan() : identity(0) {}
    T identity;
    static T f(T a, T b) { return a + b; }
};

// template <class TT>
// struct copy_scan_ptr {
//   using T = TT;
//   copy_scan_ptr() : identity(0) {}
//   T identity;
//   static T f(T a, T b) { return (b == nullptr) ? a : b; }
// };

inline bool predecessor_l2_one_round(int ht, int length, int64_t* keys, pptr* op_addrs,
                                  bool split, bool search, int limit,
                                  predecessor_type type, int32_t* heights, pptr **paths) {
    ASSERT(split);
    ASSERT(split || search);
    cout << ht << endl;

    auto task_start = parlay::tabulate(length, [&](int i) -> bool {
        return (i == 0) || (not_equal_pptr(op_addrs[i - 1], op_addrs[i]));
    });
    
    auto ll = parlay::pack_index(task_start);
    int llen = ll.size();
    
    if (llen == 0) return false;

    auto rr = parlay::sequence<int>(length, -1);

    parfor_wrap(0, llen, [&](size_t i) {
        rr[ll[i]] = ((int)i == llen - 1) ? length : ll[i + 1];
    });

    auto split_start = parlay::tabulate(length, [&](int i) -> bool {
        return task_start[i] && in_bbuffer(op_addrs[i].addr) && (rr[i] - i > limit);
    });
    parlay::sequence<size_t> split_ll = parlay::pack_index(split_start);
    int split_llen = split_ll.size();
    if (!search && (split_llen == 0)) {
        return false;
    }

    auto search_start = parlay::tabulate(length, [&](int i) -> bool {
        return task_start[i] && in_bbuffer(op_addrs[i].addr) && (rr[i] - i <= limit);
    });
    parlay::sequence<size_t> search_ll;
    int search_llen = 0;

    if (search) {
        search_ll = parlay::pack_index(search_start);
        // search_ll = parlay::pack(ll, search_start);
        search_llen = search_ll.size();
        if (!split && (search_llen == 0)) {
            return false;
        }
    }

    printf("llen=%d\nsplit_llen=%d\nsearch_llen=%d\n", llen, split_llen, search_llen);
    
    if (search_llen == 0) {
        search = false;
    }
    if (split_llen == 0) {
        split = false;
    }

    auto this_ll = parlay::sequence<int>(length, -1);
    auto print_starts = [&]() {
        printf("starts=%d\n", llen);
        print_parlay_sequence(ll);
        printf("split=%d\n", split_llen);
        if (split_llen > 0) {
            print_parlay_sequence(split_ll);
        }
        printf("search=%d\n", search_llen);
        if (search_llen > 0) {
            print_parlay_sequence(search_ll);
        }
        for (int i = 0; i < length; i ++) {
            printf("this_ll[%d]=%d\n", i, this_ll[i]);
        }
    };

    parfor_wrap(
        0, llen,
        [&](size_t i) {
            int l = ll[i];
            ASSERT(task_start[l]);
            if (search_start[l]) {
                if (type == predecessor_insert) {
                    this_ll[l] = -2;
                } else {
                    this_ll[l] = search ? l : -2;
                }
            } else if (split_start[l]) {
                this_ll[l] = split ? l : -2;
            } else {
                ASSERT(in_pbuffer(op_addrs[l].addr));
                this_ll[l] = -2;
            }
        });

    parlay::scan_inclusive_inplace(this_ll, copy_scan<int>());

    // print_starts();

#ifdef KHB_CPU_DEBUG
    {
        std::mutex mut;
        parfor_wrap(0, length, [&](size_t i) {
            if (!in_bbuffer(op_addrs[i].addr)) {
                ASSERT(in_pbuffer(op_addrs[i].addr));
                ASSERT(!search_start[i]);
                ASSERT(!split_start[i]);
                return;
            }
            if (task_start[i]) {
                ASSERT(i == 0 || !equal_pptr(op_addrs[i], op_addrs[i - 1]));
            }
            if (i == 0 || !equal_pptr(op_addrs[i], op_addrs[i - 1])) {
                ASSERT(task_start[i] == true);
                int l = i;
                int r = rr[i];
                for (int j = l; j < r; j++) {
                    ASSERT(equal_pptr(op_addrs[l], op_addrs[j]));
                }
                ASSERT_EXEC(r == length || task_start[r] ||
                                in_pbuffer(op_addrs[r].addr),
                            {
                                mut.lock();
                                for (int k = l; k <= r; k++) {
                                    printf("k=%d addr=%lx ts=%d\n", k,
                                           pptr_to_int64(op_addrs[k]),
                                           task_start[k]);
                                }
                                mut.unlock();
                            });
            }
        });
        parfor_wrap(0, llen, [&](int i) {
            int l = ll[i];
            int r = rr[l];
            ASSERT(this_ll[l] == l || this_ll[l] == -2);
            for (int j = l; j < r; j++) {
                ASSERT(this_ll[j] == this_ll[l]);
            }
        });
    }
#endif

    auto io = alloc_io_manager();
    ASSERT(io == io_managers[0]);
    io->init(epoch_number++);

    // split
    IO_Task_Batch *L2_get_node_batch = nullptr;
    if (split) {
        time_nested("get node taskgen", [&]() {
            int expected_get_node_reply = S64(L2_SIZE * 2 + 2);
            L2_get_node_batch = io->alloc_task_batch(
                direct, fixed_length, variable_length, B_GET_NODE_TSK,
                sizeof(b_get_node_task), expected_get_node_reply);
            ASSERT(L2_get_node_batch == &io->tbs[0]);
            auto batch = L2_get_node_batch;
            L2_get_node_batch->push_task_from_array(
                split_llen,
                [&](uint32_t i) -> task_idx {
                    int l = split_ll[i];
                    return (task_idx){
                        .id = op_addrs[l].id, .size = -1, .offset = l};
                },
                [&](int i, int offset, uint8_t *addr) {
                    b_get_node_task* bgnt = (b_get_node_task*)addr;
                    *bgnt = (b_get_node_task){.addr = op_addrs[i]};
                    op_taskpos[i] = offset;
                });
            // parfor_wrap(0, split_llen, [&](size_t i) {
            //     int l = split_ll[i];
            //     b_get_node_task *bgnt =
            //         (b_get_node_task *)batch->push_task_zero_copy(
            //             op_addrs[l].id, -1, true, op_taskpos + l);
            //     *bgnt = (b_get_node_task){.addr = op_addrs[l]};
            // });
            printf("get node count = %d\n", split_llen);
            io->finish_task_batch();
        });
    }

    bool fixed_length_search = (search_llen > (length / 2)) && (ht == 0);
    // fixed_length_search = false;

    // search: predecessor only
    IO_Task_Batch *L2_search_batch = nullptr;
    if (search && type == predecessor_only) {
        time_nested("search taskgen", [&]() {
            if (fixed_length_search) {
                L2_search_batch = io->alloc_task_batch(
                    direct, fixed_length, fixed_length, B_FIXED_SEARCH_TSK,
                    sizeof(b_fixed_search_task), sizeof(b_fixed_search_reply));
                ASSERT((split && L2_search_batch == &io->tbs[1]) ||
                       (!split && L2_search_batch == &io->tbs[0]));
                auto batch = L2_search_batch;
                // L2_search_batch->push_task_from_array(
                //     length,
                //     [&](uint32_t i) -> task_idx {
                //         int l = split_ll[i];
                //         int r = rr[l];
                //         int len = r - l;
                //         return (task_idx){.id = op_addrs[l].id,
                //                           .size = S64(2 + len),
                //                           .offset = l};
                //     },
                //     [&](int i, int offset, uint8_t *addr) {
                //         int r = rr[i];
                //         int len = r - i;
                //         b_search_task *bst = (b_search_task *)addr;
                //         bst->addr = op_addrs[i];
                //         bst->len = len;
                //         memcpy(bst->keys, keys + l, S64(len));
                //         op_taskpos[i] = offset;
                //     });
                parfor_wrap(0, search_llen, [&](size_t i) {
                    int l = search_ll[i];
                    int r = rr[l];
                    int len = r - l;
                    ASSERT(len > 0 && len <= push_pull_limit);
                    for (int j = l; j < r; j++) {
                        pptr addr = op_addrs[j];
                        b_fixed_search_task *bfst =
                            (b_fixed_search_task *)batch->push_task_zero_copy(
                                addr.id, -1, true, op_taskpos + j);
                        // printf("search[%d] addr=%lx key=%lx\n", j, addr,
                        // keys[j]);
                        *bfst =
                            (b_fixed_search_task){.addr = addr, .key = keys[j]};
                    }
                });
                io->finish_task_batch();
                // exit(-1);
            } else {
                int expected_search_reply = S64(2);
                L2_search_batch = io->alloc_task_batch(
                    direct, variable_length, variable_length, B_SEARCH_TSK, -1,
                    expected_search_reply);
                ASSERT((split && L2_search_batch == &io->tbs[1]) ||
                       (!split && L2_search_batch == &io->tbs[0]));
                auto batch = L2_search_batch;
                // L2_search_batch->push_task_from_array(
                //     search_llen,
                //     [&](uint32_t i) -> task_idx {
                //         int l = search_ll[i];
                //         int r = rr[l];
                //         int len = r - l;
                //         return (task_idx){.id = op_addrs[l].id,
                //                           .size = S64(2 + len),
                //                           .offset = l};
                //     },
                //     [&](int i, int offset, uint8_t *addr) {
                //         int r = rr[i];
                //         int len = r - i;
                //         b_search_task *bst = (b_search_task *)addr;
                //         bst->addr = op_addrs[i];
                //         bst->len = len;
                //         memcpy(bst->keys, keys + i, S64(len));
                //         op_taskpos[i] = offset;
                //     });
                parfor_wrap(0, search_llen, [&](size_t i) {
                    int l = search_ll[i];
                    int r = rr[l];
                    // printf("search %d %d\n", l, r);
                    int len = r - l;
                    ASSERT(len > 0 && len <= push_pull_limit);
                    b_search_task *bst =
                        (b_search_task *)batch->push_task_zero_copy(
                            op_addrs[l].id, S64(2 + len), true, op_taskpos +
                            l);
                    bst->addr = op_addrs[l];
                    bst->len = len;
                    memcpy(bst->keys, keys + l, sizeof(int64_t) * len);
                    // for (int j = l; j < r; j++) {
                    //     bst->keys[j - l] = keys[j];
                    // }
                });
                io->finish_task_batch();
            }
        });
    }

    // generate search tasks with path record
    IO_Task_Batch *L2_search_with_path_batch = nullptr;
    if (search && type == predecessor_insert) {
        time_nested("search with path taskgen", [&]() {
            int expected_search_with_path_reply = S64(2);
            L2_search_with_path_batch = io->alloc_task_batch(
                direct, variable_length, variable_length,
                B_SEARCH_WITH_PATH_TSK, -1, expected_search_with_path_reply);
            ASSERT((split && L2_search_with_path_batch == &io->tbs[1]) ||
                   (!split && L2_search_with_path_batch == &io->tbs[0]));
            auto batch = L2_search_with_path_batch;
            parfor_wrap(0, search_llen, [&](size_t i) {
                int l = search_ll[i];
                int r = rr[l];
                int len = r - l;
                ASSERT(len > 0 && len <= push_pull_limit);
                b_search_with_path_task *bst =
                    (b_search_with_path_task *)batch->push_task_zero_copy(
                        op_addrs[l].id, S64(2 + len * 2), true, op_taskpos + l);
                bst->addr = op_addrs[l];
                ASSERT(valid_b_pptr(op_addrs[l]));
                bst->len = len;
                for (int j = 0; j < len; j++) {
                    bst->vals[j] = keys[l + j];
                    bst->vals[j + len] = heights[l + j];
                    ASSERT(heights[l + j] == 1 || heights[l + j] == L2_HEIGHT);
                }
                // memcpy(bst->vals, keys + l, S64(len));
                // memcpy(bst->vals + len, heights + l, S64(len));
            });
            io->finish_task_batch();
        });
    }

    time_nested("exec", [&]() { io->exec(); });

    auto active_pos = parlay::pack_index(parlay::delayed_tabulate(
        length, [&](int i) -> bool { return this_ll[i] >= 0; }));
    int active_pos_len = active_pos.size();

    time_nested("get result", [&]() {
        if (type == predecessor_insert && search) {
            printf("predecessor_insert_search\n");
            parfor_wrap(0, search_llen, [&](size_t i) {
                int loop_l = search_ll[i];
                int loop_r = rr[loop_l];
                int len = loop_r - loop_l;
                b_search_with_path_reply *bsr =
                    (b_search_with_path_reply *)L2_search_with_path_batch
                        ->get_reply(op_taskpos[loop_l], op_addrs[loop_l].id);
                ASSERT(len <= bsr->len);
                offset_pptr *task_op = bsr->ops;
                for (int j = 0; j < bsr->len; j++) {
                    int off = task_op[j].offset + loop_l;
                    pptr ad =
                        (pptr){.id = task_op[j].id, .addr = task_op[j].addr};
                    ASSERT(in_pbuffer(ad.addr) || in_bbuffer(ad.addr));

                    op_addrs[off] = ad;
                    if (!in_bbuffer(ad.addr)) {
                        continue;
                    }
                    if (heights[off] == 1) {
                        paths[off][0] = ad;
                    } else {
                        ASSERT(heights[off] == L2_HEIGHT);
                        for (int k = L2_HEIGHT - 1; k >= 0; k--) {
                            if (paths[off][k].id == INVALID_DPU_ID) {
                                paths[off][k] = ad;
                                break;
                            }
                            if (k == 0) {
                                for (int kk = 0; kk < bsr->len; kk++) {
                                    int off = task_op[kk].offset + loop_l;
                                    pptr ad = (pptr){.id = task_op[kk].id,
                                                     .addr = task_op[kk].addr};
                                    printf("j=%d ad=%lx\n", off,
                                           pptr_to_int64(ad));
                                }
                                for (int kk = L2_HEIGHT - 1; kk >= 0; kk--) {
                                    printf("j=%d kk=%d path=%lx\n", off, kk,
                                           pptr_to_int64(paths[off][kk]));
                                }
                                printf("j=%d path=%lx\n", off,
                                       pptr_to_int64(ad));
                                fflush(stdout);
                                print_log(0, true);
                                ASSERT(false);
                            }
                            // ASSERT(k != 0);
                        }
                    }
                }
#ifdef KHB_CPU_DEBUG
                for (int j = loop_l; j < loop_r; j++) {
                    ASSERT_EXEC(valid_pptr(op_addrs[j], nr_of_dpus), {
                        mut.lock();
                        for (int xx = loop_l; xx < loop_r; xx++) {
                            printf("%d %lx\n", xx, pptr_to_int64(op_addrs[xx]));
                        }
                        fflush(stdout);
                        mut.unlock();
                    });
                }
#endif
                // int pre_pos = loop_l;
                // for (int j = loop_l + 1; j < loop_r; j++) {
                //     if (not_equal_pptr(op_addrs[j], op_addrs[j - 1])) {
                //         rr[pre_pos] = j;
                //         task_start[j] = true;
                //         rr[j] = loop_r;
                //         pre_pos = j;
                //     }
                // }
                // for (int j = loop_l; j < loop_r; j++) {
                //     if (!in_bbuffer(op_addrs[j].addr)) {
                //         ASSERT(in_pbuffer(op_addrs[j].addr));
                //         task_start[j] = false;
                //     }
                // }
            });
        }
        parfor_wrap(0, active_pos_len, [&](size_t x) {
            int i = active_pos[x];
            int l = this_ll[i];
            int j = i - l;
            if (search_start[l] && fixed_length_search) {
                ASSERT(!split_start[l] && search && type == predecessor_only);
                // printf("%x %lx\n", op_taskpos[i], op_addrs[i]);
                b_fixed_search_reply *bfsr =
                    (b_fixed_search_reply *)L2_search_batch->get_reply(
                        op_taskpos[i], op_addrs[i].id);
                op_addrs[i] = bfsr->addr;
            } else if (search_start[l]) {
                ASSERT(!split_start[l] && search && type == predecessor_only);
                b_search_reply *bsr =
                    (b_search_reply *)L2_search_batch->get_reply(
                        op_taskpos[l], op_addrs[i].id);
                ASSERT(j < bsr->len);
                op_addrs[i] = int64_to_pptr(bsr->addrs[j]);
            } else if (split_start[l]) {
                ASSERT(!search_start[l] && split);
                b_get_node_reply *bgnr =
                    (b_get_node_reply *)L2_get_node_batch->get_reply(
                        op_taskpos[l], op_addrs[i].id);
                int nnlen = bgnr->len;
                int64_t *rep_vals = bgnr->vals;
                int64_t *rep_keys = rep_vals;
                pptr *rep_addrs = (pptr *)(rep_vals + nnlen);

                int64_t maxval = INT64_MIN;
                pptr maxaddr = null_pptr;
                for (int k = 0; k < nnlen; k++) {
                    if (rep_keys[k] <= keys[i] && rep_keys[k] >= maxval) {
                        maxval = rep_keys[k];
                        maxaddr = rep_addrs[k];
                    }
                }
                op_addrs[i] = maxaddr;
                if (type == predecessor_insert) {
                    if (ht >= 1 && ht <= heights[i]) {  // split
                        paths[i][ht - 1] = maxaddr;
                    }
                }
                ASSERT(valid_pptr(op_addrs[i], nr_of_dpus));
            } else {
                ASSERT(false);
            }
        });
    });
    io->reset();

    return true;
}

inline void predecessor(predecessor_type type, int length, int64_t *keys,
                        int32_t *heights = NULL, pptr **paths = NULL) {
    if (type == predecessor_only) {
        dpu_binary_switch_to(dpu_binary::predecessor_binary);
    }

    if (type == predecessor_only) {
        auto seq = parlay::make_slice(keys, keys + length);
        parlay::sort_inplace(seq);
    }

    epoch_number++;
    // printf("START PREDECESSOR\n");

    // cout<<length<<endl;
    // for (int i = 0; i < length; i += 1000) {
    //     printf("%lld\n", keys[i]);
    // }
    // exit(0);

    time_nested("L3", [&]() {
        auto io = alloc_io_manager();
        ASSERT(io == io_managers[0]);
        io->init(epoch_number++);

        auto L3_predecessor_batch = io->alloc_task_batch(
            direct, fixed_length, fixed_length, L3_SEARCH_TSK,
            sizeof(L3_search_task), sizeof(L3_search_reply));
        time_nested("taskgen", [&]() {
            auto batch = L3_predecessor_batch;
            parfor_wrap(0, nr_of_dpus, [&](size_t i) {
                int l = (int64_t)length * i / nr_of_dpus;
                int r = (int64_t)length * (i + 1) / nr_of_dpus;
                for (int j = l; j < r; j++) {
                    L3_search_task *tst =
                        (L3_search_task *)batch->push_task_zero_copy(i, -1,
                                                                     false);
                    *tst = (L3_search_task){.key = keys[j]};
                }
            });
            io->finish_task_batch();
        });

        time_nested("exec", [&]() { io->exec(); });

        time_nested("get result", [&]() {
            auto batch = L3_predecessor_batch;
            parfor_wrap(0, nr_of_dpus, [&](size_t i) {
                int l = (int64_t)length * i / nr_of_dpus;
                int r = (int64_t)length * (i + 1) / nr_of_dpus;
                for (int j = l; j < r; j++) {
                    L3_search_reply *tsr =
                        (L3_search_reply *)batch->get_reply(j - l, i);
                    op_addrs[j] = tsr->addr;
                }
            });
        });

        io->io_manager_state = idle;
    });

    // for (int i = 0; i < length; i ++) {
    //     print_pptr(op_addrs[i], "\n");
    // }
    // exit(0);

    if (type == predecessor_insert) {
        time_nested("record path", [&]() {
            parfor_wrap(0, length, [&](size_t i) {
                if (heights[i] == L2_HEIGHT) {
                    paths[i][L2_HEIGHT - 1] = op_addrs[i];
                    for (int j = 0; j < L2_HEIGHT - 1; j++) {
                        paths[i][j] = null_pptr;
                    }
                } else if (heights[i] == 1) {
                    paths[i][0] = null_pptr;
                } else {
                    ASSERT(false);
                }
            });
        });
    }

    time_nested("L2", [&]() {
#ifdef READ_OPTIMIZED
        predecessor_l2_one_round(2, length, keys, op_addrs, true, true,
                                 /*push_pull_limit*/ 16, type, heights, paths);
        predecessor_l2_one_round(1, length, keys, op_addrs, true, false,
                                 push_pull_limit, type, heights, paths);
        predecessor_l2_one_round(0, length, keys, op_addrs, true, true,
                                 push_pull_limit, type, heights, paths);
#else
        time_nested("l2_one_round", [&](){
            predecessor_l2_one_round(2, length, keys, op_addrs, true, false,
                                push_pull_limit, type, heights, paths);
        });
        time_nested("l2_one_round", [&](){
            predecessor_l2_one_round(1, length, keys, op_addrs, true, true,
                                push_pull_limit, type, heights, paths);
        });
        if (type == predecessor_only) {
            time_nested("l2_one_round", [&](){
                predecessor_l2_one_round(0, length, keys, op_addrs, true, true,
                                push_pull_limit, type, heights, paths);
            });
        }

#endif
    });

#ifdef KHB_CPU_DEBUG
    if (type == predecessor_insert) {
        for (int i = 0; i < length; i++) {
            for (int j = 0; j < heights[i]; j++) {
                ASSERT(valid_pptr(paths[i][j], nr_of_dpus));
            }
        }
    }
#endif

    if (type == predecessor_only) {  // not insert_predecessor

        // cout << "Height: " << 0 << endl;
        // for (int i = 0; i < length; i++) {
        //     printf("%d\t%lx\t%lx\n", i, keys[i], pptr_to_int64(op_addrs[i]));
        // }

        // exit(0);

        time_nested("L1", [&]() {
            auto io = alloc_io_manager();
            ASSERT(io == io_managers[0]);
            io->init(epoch_number++);
            auto L1_search_batch = io->alloc_task_batch(
                direct, fixed_length, fixed_length, P_GET_KEY_TSK,
                sizeof(p_get_key_task), sizeof(p_get_key_reply));

            // for (int i = 0; i < length; i ++) {
            //     if (keys[i] == 0xbe1637538c228456ll) {
            //         printf("pptr=%llx\n", pptr_to_int64(op_addrs[i]));
            //     }
            // }
            time_nested("taskgen", [&]() {
                auto batch = L1_search_batch;
                parfor_wrap(0, length, [&](size_t i) {
                    // for (int i = 0; i < length; i ++) {
                    // if (!((int)op_addrs[i].id < nr_of_dpus)) {
                    //     printf("%d\t%lx\t%lx\n", (int)i, keys[i],
                    //            pptr_to_int64(op_addrs[i]));
                    //     fflush(stdout);
                    //     exit(-1);
                    // }
                    p_get_key_task *pgkt =
                        (p_get_key_task *)batch->push_task_zero_copy(
                            op_addrs[i].id, -1, true, op_taskpos + i);
                    *pgkt = (p_get_key_task){.addr = op_addrs[i]};
                });
                // }
                io->finish_task_batch();
            });

            time_nested("exec", [&]() { io->exec(); });

            time_nested("get result", [&]() {
                auto batch = L1_search_batch;
                parfor_wrap(0, length, [&](size_t i) {
                    p_get_key_reply *pgkr = (p_get_key_reply *)batch->get_reply(
                        op_taskpos[i], op_addrs[i].id);
                    op_results[i] = pgkr->key;
                });
            });
            io->reset();
        });
    }

    // for (int i = 0; i < length; i ++) {
    //     printf("%ld %ld\n", keys[i], op_results[i]);
    // }
}

auto deduplication(int64_t *arr, int &length) {  // assume sorted
    auto seq = parlay::make_slice(arr, arr + length);
    parlay::sort_inplace(seq);

    auto dup = parlay::tabulate(
        length, [&](int i) -> bool { return i == 0 || seq[i] != seq[i - 1]; });
    auto packed = parlay::pack(seq, dup);
    length = packed.size();
    return packed;
}

int newnode_offset_buffer[BATCH_SIZE * 3];
int newnode_count[L2_HEIGHT + 2];
int *node_id[L2_HEIGHT + 2];

pptr insert_path_addrs_buf[BATCH_SIZE * 3];
pptr l2_addrs_buf[BATCH_SIZE * 3];
int l2_addrs_taskpos_buf[BATCH_SIZE * 3];
int insert_truncate_taskpos_buf[BATCH_SIZE * 3];
// int64_t insert_path_chks_buf[BATCH_SIZE * 2];

// pptr *insert_path_addrs[BATCH_SIZE];
// pptr *insert_path_rights[BATCH_SIZE];
// int64_t *insert_path_chks[BATCH_SIZE];

inline void print_height_distribution(int all_length,
                                      parlay::sequence<int> all_heights) {
    int cnt[20] = {0};
    for (int i = 0; i < all_length; i++) {
        for (int j = 0; j <= all_heights[i]; j++) {
            cnt[j]++;
        }
    }
    for (int i = 0; i < 20; i++) {
        printf("%d %d\n", i, cnt[i]);
    }
}

inline bool is_permutation(parlay::sequence<uint32_t> seq) {
    auto tmp = parlay::map(seq, [&](uint32_t x) { return x; });
    parlay::integer_sort_inplace(tmp);
    int len = tmp.size();
    parfor_wrap(0, len, [&](size_t i) { ASSERT(tmp[i] == i); });
    return true;
}

inline void horizontal_reduce(int *heights, int length) {
    const int BLOCK = 128;
    int newnode_count_threadlocal[BLOCK + 1][L2_HEIGHT + 2];
    int *node_offset_threadlocal[BLOCK + 1][L2_HEIGHT + 2];
    memset(newnode_count, 0, sizeof(newnode_count));
    memset(newnode_count_threadlocal, 0, sizeof(newnode_count_threadlocal));
    std::mutex reduce_mutex;
    parfor_wrap(
        0, BLOCK,
        [&](size_t i) {
            int l = (int64_t)length * i / BLOCK;
            int r = (int64_t)length * (i + 1) / BLOCK;
            for (int j = l; j < r; j++) {
                int height = min(heights[j], L2_HEIGHT + 1);
                for (int ht = 0; ht < height; ht++) {
                    newnode_count_threadlocal[i][ht]++;
                }
            }
            reduce_mutex.lock();
            for (int ht = 0; ht <= L2_HEIGHT; ht++) {
                newnode_count[ht] += newnode_count_threadlocal[i][ht];
            }
            reduce_mutex.unlock();
        },
        true, 1);
    node_id[0] = newnode_offset_buffer;
    for (int i = 1; i <= L2_HEIGHT + 1; i++) {
        node_id[i] = node_id[i - 1] + newnode_count[i - 1];
        // printf("*1 %d %ld\n", i, node_id[i] - node_id[i - 1]);
    }
    ASSERT(node_id[L2_HEIGHT + 1] <= newnode_offset_buffer + BATCH_SIZE * 2);
    for (int i = 0; i <= BLOCK; i++) {
        for (int ht = 0; ht <= L2_HEIGHT; ht++) {
            if (i == 0) {
                node_offset_threadlocal[0][ht] = node_id[ht];
            } else {
                node_offset_threadlocal[i][ht] =
                    node_offset_threadlocal[i - 1][ht] +
                    newnode_count_threadlocal[i - 1][ht];
            }
            if (i == BLOCK) {
                ASSERT(node_offset_threadlocal[i][ht] == node_id[ht + 1]);
            }
            // printf("*2 %d %d %ld\n", i, ht,
            //        node_offset_threadlocal[i][ht] -
            //        newnode_offset_buffer);
        }
    }
    parfor_wrap(0, BLOCK, [&](size_t i) {
        int l = (int64_t)length * i / BLOCK;
        int r = (int64_t)length * (i + 1) / BLOCK;
        for (int j = l; j < r; j++) {
            int height = min(heights[j], L2_HEIGHT + 1);
            for (int ht = 0; ht < height; ht++) {
                *(node_offset_threadlocal[i][ht]++) = j;
            }
        }
    });
}

inline void insert(int length, int64_t *insert_keys, int64_t *insert_values) {
    // dpu_binary_switch_to(dpu_binary::insert_binary);
    printf("\n********** INIT SKIP LIST **********\n");

    timer *insert_init = start_timer("insert_init");

    printf("\n**** INIT HEIGHT ****\n");
    auto keys = deduplication(insert_keys, length);

    auto heights = parlay::sequence<int>::from_function(
        length,
        [&](size_t i) -> int {
            (void)&i;
            int t = randint64(parlay::worker_id());
            t = __builtin_ctz(t) + 1;
            if (t <= L2_HEIGHT * L2_SIZE_LOG) {
                t = (t - 1) / L2_SIZE_LOG + 1;
            } else {
                t = t - L2_HEIGHT * L2_SIZE_LOG + L2_HEIGHT;
                t = min(t, L2_HEIGHT + max_l3_height);
            }
            return t;
        },
        (PARALLEL_ON) ? 0 : INT32_MAX);  // 1111, 2222, 3333, 4, 5, 6, ...

    // for (int i = 0; i < length; i ++) {
    //     if (keys[i] == 0x3e5d92c2f036a056ll) {
    //         printf("!!! key=%lx !!! i=%d !!! height=%d\n", keys[i], i,
    //         heights[i]);
    //     }
    // }

    auto predecessor_record = parlay::map(heights, [&](int32_t x) {
        return (x >= CACHE_HEIGHT) ? L2_HEIGHT : 1;
    });

    time_nested("horizontal reduce",
                [&]() { horizontal_reduce(heights.data(), length); });

    // for (int i = 0; i <= L2_HEIGHT; i ++) {
    //     printf("height=%d cnt=%d\n", i, newnode_count[i]);
    //     for (int j = 0; j < newnode_count[i]; j ++) {
    //         int k = node_id[i][j];
    //         printf("id=%d height=%d\n", k, heights[k]);
    //     }
    // }
    // exit(0);

    auto predecessor_record_sum_pair = parlay::scan(predecessor_record);

    auto predecessor_record_total = predecessor_record_sum_pair.second;
    auto predecessor_record_prefix_sum = predecessor_record_sum_pair.first;

    auto insert_path_addrs =
        parlay::map(predecessor_record_prefix_sum,
                    [&](int32_t x) { return insert_path_addrs_buf + x; });

#ifdef READ_OPTIMIZED
    ASSERT(predecessor_record_total < BATCH_SIZE * 3);
#else
    ASSERT(predecessor_record_total < BATCH_SIZE * 1.2);
#endif

    insert_init->end();

    printf("\n**** INSERT PREDECESSOR ****\n");
    time_nested("predecessor", [&]() {
        predecessor(predecessor_insert, keys.size(), keys.data(),
                    predecessor_record.data(), insert_path_addrs.data());
        // insert_path_rights.data(), insert_path_chks.data());
    });

    dpu_binary_switch_to(dpu_binary::insert_binary);

    // if (epoch_number == 12) {
    //     print_log(0, true);
    // }

    // if (epoch_number > 136) {
    // for (int i = 0; i < length; i++) {
    //     int ht = min(L2_HEIGHT, predecessor_record[i]);
    //     for (int j = 0; j < ht; j++) {
    //         printf("insert_path_addrs[%d][%d]=%lx\n", i, j,
    //                pptr_to_int64(insert_path_addrs[i][j]));
    //     }
    // }
    //     exit(0);
    // }

    auto keys_target = parlay::tabulate(length, [&](uint32_t i) {
        return hash_to_dpu(keys[i], 0, nr_of_dpus);
    });

    auto l2_root_target = parlay::sequence<int>(length, -1);
    for (int x = 0; x < newnode_count[L2_HEIGHT]; x++) {
        int i = node_id[L2_HEIGHT][x];
        int cnc = INT32_MAX;
        for (int KK = 0; KK < 4; KK++) {
            int t = abs(randint64(parlay::worker_id())) % nr_of_dpus;
            if (l2_root_count[t] < cnc) {
                cnc = l2_root_count[t];
                l2_root_target[i] = t;
            }
        }
        l2_root_count[l2_root_target[i]]++;
    }
    // {
    //     int l2_root_max = 0, l2_root_sum = 0;
    //     for (int i = 0; i < nr_of_dpus; i ++) {
    //         l2_root_max = max(l2_root_max, l2_root_count[i]);
    //         l2_root_sum += l2_root_count[i];
    //     }
    //     printf("l2root: sum=%d max=%d ratio=%f\n", l2_root_sum, l2_root_max,
    //     (double)l2_root_max * nr_of_dpus / l2_root_sum);
    // }

    // auto io = alloc_io_manager();
    // ASSERT(io == io_managers[0]);
    // io->init(epoch_number++);
    // auto L2_insert_batch =
    //     io2->alloc_task_batch(direct, fixed_length, fixed_length,
    //     L2_INSERT_TSK,
    //                           sizeof(L2_insert_task),
    //                           sizeof(L2_insert_reply));
    // time_nested("insert L2 taskgen", [&]() {
    //     parfor_wrap(0, length, [&](size_t i) {
    //         ASSERT(heights[i] > 0);
    //         L2_insert_task *sit =
    //             (L2_insert_task *)L2_insert_batch->push_task_zero_copy(
    //                 keys_target[i], -1, true, op_taskpos + i);
    //         *sit = (L2_insert_task){.key = keys[i], .height = heights[i]};
    //     });
    //     io2->finish_task_batch();
    // });

    // to upper nodes
    // auto length = upper_idx.size();
    // ASSERT(length > 0);
    // auto keys = parlay::pack(all_keys, not_l1_only);
    // auto heights = parlay::pack(all_heights, not_l1_only);
    // auto insert_path_addrs = parlay::pack(all_insert_path_addrs,
    // not_l1_only); auto insert_path_rights =
    // parlay::pack(all_insert_path_rights, not_l1_only); auto insert_path_chks
    // = parlay::pack(all_insert_path_chks, not_l1_only); auto l1_pred_addrs =
    // parlay::pack(all_addrs, not_l1_only);

    // // printf("\n**** INSERT L1 ****\n");

    // auto insert_L1_prepare = start_timer("insert_L1_prepare");
    // auto idx_l1_pred = parlay::tabulate(length, [&](uint32_t i) { return i;
    // }); auto ll_l1_pred = parlay::sequence<int>(nr_of_dpus); auto rr_l1_pred
    // = parlay::sequence<int>(nr_of_dpus);
    // parlay::integer_sort_inplace(  // error ???
    //     idx_l1_pred, [&l1_pred_addrs](const auto &x) {
    //         return (uint32_t)l1_pred_addrs[x].id;
    //     });
    // parfor_wrap(0, nr_of_dpus, [&](size_t i) {
    //     uint32_t l = i, r = i + 1;
    //     int loop_l = binary_search_local_r(-1, length, [&](int x) {
    //         return l <= l1_pred_addrs[idx_l1_pred[x]].id;
    //     });
    //     int loop_r = binary_search_local_r(-1, length, [&](int x) {
    //         return r <= l1_pred_addrs[idx_l1_pred[x]].id;
    //     });
    //     ll_l1_pred[i] = loop_l;
    //     rr_l1_pred[i] = loop_r;
    //     // parlay::make_slice(idx_l1_pred).cut(ll_l1_pred[i], )
    //     sort(idx_l1_pred.data() + loop_l, idx_l1_pred.data() + loop_r);
    // });
    // for (int i = 0; i < nr_of_dpus; i++) {
    //     int loop_l = ll_l1_pred[i], loop_r = rr_l1_pred[i];
    //     int len = loop_r - loop_l;
    //     for (int j = 0; j < len; j++) {
    //         int k = idx_l1_pred[loop_l + j];
    //         printf("i=%d\tk=%d\tid=%d\n", i, k,
    // l1_pred_addrs[k].id);
    //         // printf("idx=%d\tid=%d\n", idx_l1_pred[i],
    //         l1_pred_addrs[i].id);
    //     }
    // }
    // #ifdef KHB_CPU_DEBUG
    //     ASSERT(is_permutation(idx_l1_pred));
    //     // checking
    //     parfor_wrap(0, nr_of_dpus, [&](size_t i) {
    //         int loop_l = ll_l1_pred[i], loop_r = rr_l1_pred[i];
    //         ASSERT(i != 0 || loop_l == 0);
    //         ASSERT(i == 0 || ll_l1_pred[i] == rr_l1_pred[i - 1]);
    //         ASSERT((int)i != (nr_of_dpus - 1) || loop_r == (int)length);
    //         for (int j = ll_l1_pred[i] + 1; j < rr_l1_pred[i]; j++) {
    //             if (!(idx_l1_pred[j - 1] < idx_l1_pred[j])) {
    //                 printf("j=%d idx_l1_pred[j - 1]=%d
    //                 idx_l1_pred[j]=%d\n", j,
    //                        idx_l1_pred[j - 1], idx_l1_pred[j]);
    //                 fflush(stdout);
    //                 ASSERT(false);
    //             }
    //         }
    //     });
    // #endif
    //     insert_L1_prepare->end();

    printf("\n**** INSERT L123 ****\n");

    auto l1_addrs = parlay::sequence(length, null_pptr);
    auto l3_addrs = parlay::sequence(length, null_pptr);
    auto l2_addrs = parlay::map(predecessor_record_prefix_sum, [&](int32_t x) {
        return l2_addrs_buf + x;
    });  // may waste little memory
    auto l2_addrs_taskpos =
        parlay::map(predecessor_record_prefix_sum,
                    [&](int32_t x) { return l2_addrs_taskpos_buf + x; });

    auto io = alloc_io_manager();
    ASSERT(io == io_managers[0]);
    io->init(epoch_number++);

    IO_Task_Batch *L3_insert_batch = nullptr;
    int l3_length = 0;
    // int L3_id[MAX_TASK_COUNT_PER_DPU_PER_BLOCK];

    time_nested("L3 taskgen", [&]() {
        l3_length = newnode_count[L2_HEIGHT];
        if (l3_length == 0) {
            return;
        }
        L3_insert_batch = io->alloc_task_batch(
            broadcast, fixed_length, fixed_length, L3_INSERT_TSK,
            sizeof(L3_insert_task), sizeof(L3_insert_reply));
        for (int t = 0; t < l3_length; t++) {
            int i = node_id[L2_HEIGHT][t];
            // printf("T3: %d %d %ld\n", t, i, keys[i]);
            L3_insert_task *tit =
                (L3_insert_task *)L3_insert_batch->push_task_zero_copy(-1, -1,
                                                                       false);
            *tit = (L3_insert_task){.key = keys[i],
                                    .height = heights[i] - L2_HEIGHT};
        }
        io->finish_task_batch();
    });

    // if (epoch_number == 13) {
    //     io->exec();
    //     exit(-1);
    // }

    auto L2_newnode_batch =
        io->alloc_task_batch(direct, fixed_length, fixed_length, B_NEWNODE_TSK,
                             sizeof(b_newnode_task), sizeof(b_newnode_reply));
    time_nested("L2 taskgen", [&]() {
        parfor_wrap(0, length, [&](size_t i) {
            ASSERT(heights[i] > 0);
            int curheight = min(L2_HEIGHT, heights[i] - 1);
            for (int ht = 0; ht < curheight; ht++) {
                int target;
                if (ht == L2_HEIGHT - 1) {
                    target = l2_root_target[i];
                } else {
                    target = hash_to_dpu(keys[i], ht, nr_of_dpus);
                }
                b_newnode_task *bnt =
                    (b_newnode_task *)L2_newnode_batch->push_task_zero_copy(
                        target, -1, true, &(l2_addrs_taskpos[i][ht]));
                *bnt = (b_newnode_task){.height = ht};
            }
        });
        io->finish_task_batch();
    });

    auto L1_newnode_batch =
        io->alloc_task_batch(direct, fixed_length, fixed_length, P_NEWNODE_TSK,
                             sizeof(p_newnode_task), sizeof(p_newnode_reply));
    time_nested("L1 taskgen", [&]() {
        parfor_wrap(0, length, [&](size_t i) {
            p_newnode_task *pnt =
                (p_newnode_task *)L1_newnode_batch->push_task_zero_copy(
                    keys_target[i], -1, true, op_taskpos + i);
            *pnt = (p_newnode_task){.key = keys[i], .height = heights[i]};
        });
        io->finish_task_batch();
    });

    // if (epoch_number == 13) {
    //     io->send_task();
    //     exit(0);
    // }

    auto insert_truncate_taskpos =
        parlay::map(predecessor_record_prefix_sum,
                    [&](int32_t x) { return insert_truncate_taskpos_buf + x; });

    auto is_truncator = [&](int ht, int t, int newnode_ht) -> bool {
        if (t == 0) {
            return true;
        }
        int i = node_id[ht][t];
        int l = node_id[ht][t - 1];
        ASSERT(l < i);
        return not_equal_pptr(insert_path_addrs[i][newnode_ht],
                              insert_path_addrs[l][newnode_ht]);
    };

    int L2_truncate_batch_erl = sizeof(b_truncate_reply) + S64(L2_SIZE);
    auto L2_truncate_batch = io->alloc_task_batch(
        direct, fixed_length, variable_length, B_TRUNCATE_TSK,
        sizeof(b_truncate_task), L2_truncate_batch_erl);
    time_nested("L2 truncate", [&]() {
        auto new_truncate_task = [&](int i, int ht) {
            pptr addr = insert_path_addrs[i][ht];
            b_truncate_task *btt =
                (b_truncate_task *)L2_truncate_batch->push_task_zero_copy(
                    addr.id, -1, true, &(insert_truncate_taskpos[i][ht]));
            *btt = (b_truncate_task){.addr = insert_path_addrs[i][ht],
                                     .key = keys[i]};
        };
        for (int ht = 0; ht < L2_HEIGHT; ht++) {
            int nodeht = ht + 1;
            parfor_wrap(0, newnode_count[nodeht], [&](size_t t) {
                if (is_truncator(nodeht, t, ht)) {
                    new_truncate_task(node_id[nodeht][t], ht);
                }
            });
        }
        io->finish_task_batch();
    });

    auto cache_truncate_batch = io->alloc_task_batch(
        direct, fixed_length, fixed_length, CACHE_TRUNCATE_TSK,
        sizeof(cache_truncate_task), 0);
    time_nested("cache truncate", [&]() {
        // if (epoch_number != 13) {
        auto new_cache_truncate_task = [&](int i, int ht) {
            ASSERT(CACHE_HEIGHT == ht + 1);
            pptr addr = insert_path_addrs[i][ht + 1];
            cache_truncate_task *ctt =
                (cache_truncate_task *)cache_truncate_batch
                    ->push_task_zero_copy(addr.id, -1, true);
            *ctt = (cache_truncate_task){
                .addr = addr, .key = keys[i], .height = ht};
        };
        {
            int ht = CACHE_HEIGHT - 1;
            int nodeht = ht + 1;
            parfor_wrap(0, newnode_count[nodeht], [&](size_t t) {
                if (is_truncator(nodeht, t, ht)) {
                    new_cache_truncate_task(node_id[nodeht][t], ht);
                }
            });
        }
        // }
        io->finish_task_batch();
    });

    time_nested("newnode L123 + truncate L2 exec",
                [&]() { ASSERT(io->exec()); });

    // if (epoch_number == 13) {
    //     exit(-1);
    // }
    // if (length > 5e5) {
    //     print_log(0, true);
    // }
    // io->print_all_buffer(true);
    ASSERT(length > 0);
    time_nested("newnode L123 + truncate get result", [&]() {
        {
            for (int t = 0; t < l3_length; t++) {
                int i = node_id[L2_HEIGHT][t];
                L3_insert_reply *tir =
                    (L3_insert_reply *)L3_insert_batch->get_reply(t, -1);
                l3_addrs[i] = tir->addr;
            }
        }
        {
            parfor_wrap(0, length, [&](size_t i) {
                int curheight = min(L2_HEIGHT, heights[i] - 1);
                for (int ht = 0; ht < curheight; ht++) {
                    int target;
                    if (ht == L2_HEIGHT - 1) {
                        target = l2_root_target[i];
                    } else {
                        target = hash_to_dpu(keys[i], ht, nr_of_dpus);
                    }
                    b_newnode_reply *bir =
                        (b_newnode_reply *)L2_newnode_batch->get_reply(
                            l2_addrs_taskpos[i][ht], target);
                    l2_addrs[i][ht] = bir->addr;
                }
            });
        }
        {
            parfor_wrap(0, length, [&](size_t i) {
                ASSERT(op_taskpos[i] != -1);
                p_newnode_reply *pir =
                    (p_newnode_reply *)L1_newnode_batch->get_reply(
                        op_taskpos[i], keys_target[i]);
                l1_addrs[i] = pir->addr;
            });
        }
    });

    // if (epoch_number > 136) {
    //     for (int i = 0; i < length; i++) {
    //         ASSERT(heights[i] > 0);
    //         int curheight = min(L2_HEIGHT, heights[i] - 1);
    //         if (curheight == 0) {
    //             continue;
    //         }
    //         printf("** %d %d **\n", i, curheight);
    //         // print_pptr(l3_addrs[i], " l3 \n");
    //         printf("l1=%lx\n", pptr_to_int64(l1_addrs[i]));
    //         for (int ht = 0; ht < curheight; ht++) {
    //             printf("l2[%d]=%lx\n", ht, pptr_to_int64(l2_addrs[i][ht]));
    //         }
    //     }
    // }

    // timer *insert_L1_newnode = start_timer("insert L1 newnode");
    // auto L1_newnode_batch =
    //     io2->alloc_task_batch(direct, variable_length, fixed_length,
    //                           L1_NEWNODE_TSK, -1,
    //                           sizeof(L1_newnode_reply));
    // auto rptr = parlay::sequence(length, null_pptr);
    // time_nested("insert L1 taskgen", [&]() {
    //     auto prebatch = L1_truncate_batch;
    //     auto batch = L1_newnode_batch;
    //     // parfor_wrap(0, nr_of_dpus, [&](size_t i) {
    //     for (int i = 0; i < nr_of_dpus; i++) {
    //         int loop_l = ll_l1_pred[i], loop_r = rr_l1_pred[i];
    //         int len = loop_r - loop_l;
    //         for (int j = 0; j < len; j++) {
    //             int k = idx_l1_pred[loop_l + j];
    //             L1_truncate_reply *ftr =
    //                 (L1_truncate_reply *)prebatch->get_reply(j, i);
    //             rptr[k] = ftr->right;
    //             L1_newnode_task *fnt =
    //                 (L1_newnode_task *)batch->push_task_zero_copy(
    //                     keys_target[k], L1_newnode_task_size(ftr->cnt +
    //                     1), true, op_taskpos2 + k);
    //             fnt->cnt = ftr->cnt + 1;
    //             // ASSERT(ftr->keys[0] != keys[k]);
    //             if (ftr->keys[0] == keys[k]) {
    //                 printf("key=%ld\n", keys[k]);
    //                 for (int i = 0; i < ftr->cnt; i++) {
    //                     printf("ftr->cnt[%d]=%ld\n", i, ftr->keys[i]);
    //                 }
    //                 ASSERT(false);
    //             }
    //             fnt->keys[0] = keys[k];
    //             memcpy(fnt->keys + 1, ftr->keys, sizeof(int64_t) *
    //             ftr->cnt);
    //             // printf("id=%d fnt->cnt=%ld\t", keys_target[k],
    //                 fnt->cnt);
    //                 // for (int p = 0; p < fnt->cnt; p ++) {
    //                 //     printf("%ld\t", fnt->keys[p]);
    //                 // }
    //                 // printf("\n");
    //         }
    //     }
    //     // });
    //     io2->finish_task_batch();
    // });

    printf("\n**** INSERT L123 ud ****\n");

    auto io2 = alloc_io_manager();
    ASSERT(io2 == io_managers[1]);
    io2->init(epoch_number++);
    time_nested("build ud ptrs", [&]() {
        if (l3_length > 0) {
            time_nested("l3", [&]() {
                auto L3_build_d_batch = io2->alloc_task_batch(
                    broadcast, fixed_length, fixed_length, L3_BUILD_D_TSK,
                    sizeof(L3_build_d_task), 0);
                {
                    auto batch = L3_build_d_batch;
                    for (int t = 0; t < l3_length; t++) {
                        int i = node_id[L2_HEIGHT][t];
                        ASSERT(heights[i] > L2_HEIGHT);
                        L3_build_d_task *tbdt =
                            (L3_build_d_task *)batch->push_task_zero_copy(
                                -1, -1, false);
                        *tbdt = (L3_build_d_task){
                            .addr = l3_addrs[i],
                            .down = l2_addrs[i][L2_HEIGHT - 1]};
                    }
                    io2->finish_task_batch();
                }
            });
            time_nested("l2 set u", [&]() {
                auto L2_set_u_batch =
                    io2->alloc_task_batch(direct, fixed_length, fixed_length,
                                          B_SET_U_TSK, sizeof(b_set_u_task), 0);
                // printf("len = %lld\n", l3_length);
                {
                    parfor_wrap(0, l3_length, [&](size_t t) {
                        int i = node_id[L2_HEIGHT][t];
                        pptr addr = l2_addrs[i][L2_HEIGHT - 1];
                        b_set_u_task *bsut =
                            (b_set_u_task *)L2_set_u_batch->push_task_zero_copy(
                                addr.id, -1, true);
                        *bsut = (b_set_u_task){.addr = addr, .up = l3_addrs[i]};
#ifdef REMOVE_DEBUG
                        printf("addr=%llx\tup=%llx\n", addr, l3_addrs[i]);
#endif
                    });
                    io2->finish_task_batch();
                }
            });
        }

        auto L2_key_insert_batch = io2->alloc_task_batch(
            direct, variable_length, fixed_length, B_INSERT_TSK, -1, 0);
        {
            auto is_inserter = [&](int t, int ht, pptr &addr) -> bool {
                int i = node_id[ht][t];
                if (t == 0) {
                    addr = insert_path_addrs[i][ht];
                    return true;
                }
                int l = node_id[ht][t - 1];
                ASSERT(heights[l] > ht);
                if (not_equal_pptr(insert_path_addrs[i][ht],
                                   insert_path_addrs[l][ht])) {
                    addr = insert_path_addrs[i][ht];
                    return true;
                }
                if (heights[l] > ht + 1) {
                    addr = l2_addrs[l][ht];
                    return true;
                }
                return false;
            };

            // insert each key to the upmost node
            for (int ht = 0; ht < L2_HEIGHT; ht++) {
                time_nested("upmost taskgen", [&]() {
                    parfor_wrap(0, newnode_count[ht], [&](size_t t) {
                        int i = node_id[ht][t];
                        ASSERT(heights[i] > ht);
                        if (heights[i] == ht + 1) {
                            pptr insert_addr;
                            if (is_inserter(t, ht, insert_addr)) {
                                int r;
                                for (r = t + 1; r < newnode_count[ht]; r++) {
                                    int rid = node_id[ht][r];
                                    if (heights[rid] > ht + 1) {
                                        break;
                                    }
                                    if (not_equal_pptr(
                                            insert_path_addrs[i][ht],
                                            insert_path_addrs[rid][ht])) {
                                        break;
                                    }
                                }
                                // printf("insert lr ht=%d l=%d r=%d\n", ht, t,
                                // r);
                                int tsklen = r - t;
                                b_insert_task *bit =
                                    (b_insert_task *)L2_key_insert_batch
                                        ->push_task_zero_copy(
                                            insert_addr.id, S64(2 + tsklen * 2),
                                            true);
                                bit->addr = insert_addr;
                                bit->len = tsklen;
                                int64_t *tskkeys = bit->vals;
                                pptr *tskaddrs = (pptr *)(bit->vals + tsklen);
                                for (int j = t; j < r; j++) {
                                    int rid = node_id[ht][j];
                                    tskkeys[j - t] = keys[rid];
                                    tskaddrs[j - t] =
                                        (ht == 0) ? l1_addrs[rid]
                                                  : l2_addrs[rid][ht - 1];
                                }
                            }
                        }
                    });
                });
            }
            io2->finish_task_batch();
        }
        // {
        //     int i = 0;
        //     for (int ht = 0; ht < L2_HEIGHT; ht++) {
        //         pptr pred_addr = insert_path_addrs[i][ht];
        //         b_truncate_reply *rep =
        //             (b_truncate_reply *)L2_truncate_batch->get_reply(
        //                 insert_truncate_taskpos[i][ht], pred_addr.id);
        //         int64_t *repkeys = rep->vals;
        //         pptr *repaddrs = (pptr *)(rep->vals + rep->len);

        //         if (rep->len > 0) {
        //             printf("ht=%d len=%ld\n", ht, rep->len);
        //             for (int k = 0; k < 3; k++) {
        //                 printf("%d\t%ld\t%lx\n", k, repkeys[k],
        //                        pptr_to_int64(repaddrs[k]));
        //             }
        //         }
        //     }
        //     fflush(stdout);
        //     ASSERT(false);
        // }

        auto L2_newnode_init_batch = io2->alloc_task_batch(
            direct, variable_length, fixed_length, B_INSERT_TSK, -1, 0);
        {
            for (int ht = 0; ht < L2_HEIGHT; ht++) {
                time_nested("nodefill taskgen", [&]() {
                    int nodeht = ht + 1;
                    parfor_wrap(0, newnode_count[nodeht], [&](size_t t) {
                        // for (int t = 0; t < newnode_count[nodeht]; t++) {
                        if (is_truncator(nodeht, t, ht)) {
                            // [t, r) all truncate the same node
                            int i = node_id[nodeht][t];
                            pptr pred_addr = insert_path_addrs[i][ht];
                            int r;
                            for (r = t + 1; r < newnode_count[nodeht]; r++) {
                                int rid = node_id[nodeht][r];
                                if (not_equal_pptr(
                                        insert_path_addrs[i][ht],
                                        insert_path_addrs[rid][ht])) {
                                    break;
                                }
                            }
                            // printf("newnode init lr ht=%d l=%d r=%d\n",
                            // ht, t, r);

                            // init reply
                            b_truncate_reply *rep =
                                (b_truncate_reply *)L2_truncate_batch
                                    ->get_reply(insert_truncate_taskpos[i][ht],
                                                pred_addr.id);
                            int64_t *repkeys = rep->vals;
                            pptr *repaddrs = (pptr *)(rep->vals + rep->len);

#ifdef KHB_CPU_DEBUG
                            for (int j = 0; j < rep->len; j++) {
                                if ((int)repaddrs[j].id >= nr_of_dpus) {
                                    for (int k = 0; k < rep->len; k++) {
                                        printf("%lx %lx\n", repkeys[k],
                                               pptr_to_int64(repaddrs[k]));
                                    }
                                    fflush(stdout);
                                    ASSERT(false);
                                }
                            }
#endif

                            // if (rep->len > 0) {
                            //     printf("ht=%d l=%d r=%d\n", ht, t, r);
                            //     for (int k = 0; k < rep->len; k++) {
                            //         printf("%d\t%ld\t%lx\n", k,
                            //         repkeys[k],
                            //                pptr_to_int64(repaddrs[k]));
                            //     }
                            //     fflush(stdout);
                            //     ASSERT(false);
                            // }

                            int64_t key_r = INT64_MAX;
                            for (int j = r - 1; j >= (int)t; j--) {
                                int tsklen = 0;
                                int rid = node_id[nodeht][j];
                                int64_t key_l = keys[rid];
                                for (int k = 0; k < rep->len; k++) {
                                    if (repkeys[k] >= key_l &&
                                        repkeys[k] <= key_r) {
                                        tsklen++;
                                    }
                                }

                                pptr newnode_addr = l2_addrs[rid][ht];
                                b_insert_task *bit =
                                    (b_insert_task *)L2_newnode_init_batch
                                        ->push_task_zero_copy(
                                            newnode_addr.id,
                                            S64(2 + (tsklen + 1) * 2), true);
                                bit->addr = newnode_addr;
                                bit->len = tsklen + 1;
                                int64_t *tskkeys = bit->vals;
                                pptr *tskaddrs =
                                    (pptr *)(bit->vals + tsklen + 1);
                                tskkeys[0] = keys[rid];
                                tskaddrs[0] = (ht == 0) ? l1_addrs[rid]
                                                        : l2_addrs[rid][ht - 1];
                                tsklen = 1;
                                for (int k = 0; k < rep->len; k++) {
                                    if (repkeys[k] >= key_l &&
                                        repkeys[k] <= key_r) {
                                        tskkeys[tsklen] = repkeys[k];
                                        tskaddrs[tsklen] = repaddrs[k];
                                        tsklen++;
                                    }
                                }
                                ASSERT(bit->len == tsklen);
                                key_r = key_l - 1;
                            }
                        }
                    });
                    // }
                });
            }
            io2->finish_task_batch();
        }

        auto L2_set_lr_batch =
            io2->alloc_task_batch(direct, fixed_length, fixed_length,
                                  B_SET_LR_TSK, sizeof(b_set_lr_task), 0);
        {
            for (int ht = 0; ht < L2_HEIGHT; ht++) {
                time_nested("set lr taskgen", [&]() {
                    int nodeht = ht + 1;
                    parfor_wrap(0, newnode_count[nodeht], [&](size_t t) {
                        // for (int t = 0; t < newnode_count[nodeht]; t++) {
                        if (is_truncator(nodeht, t, ht)) {
                            // [t, r) all truncate the same node
                            int i = node_id[nodeht][t];
                            pptr pred_addr = insert_path_addrs[i][ht];
                            int r;
                            for (r = t + 1; r < newnode_count[nodeht]; r++) {
                                int rid = node_id[nodeht][r];
                                if (not_equal_pptr(
                                        insert_path_addrs[i][ht],
                                        insert_path_addrs[rid][ht])) {
                                    break;
                                }
                            }
                            // printf("newnode init lr ht=%d l=%d r=%d\n",
                            // ht, t, r);

                            // init reply
                            b_truncate_reply *rep =
                                (b_truncate_reply *)L2_truncate_batch
                                    ->get_reply(insert_truncate_taskpos[i][ht],
                                                pred_addr.id);
                            pptr right = rep->right;
                            pptr left = pred_addr;

                            for (int j = t; j < r; j++) {
                                int rid = node_id[nodeht][j];
                                pptr newnode_addr = l2_addrs[rid][ht];
                                if (j == t) {
                                    b_set_lr_task *bslt =
                                        (b_set_lr_task *)L2_set_lr_batch
                                            ->push_task_zero_copy(pred_addr.id,
                                                                  -1, true);
                                    bslt->addr = pred_addr;
                                    bslt->left = null_pptr;
                                    bslt->right = newnode_addr;
                                }
                                {
                                    b_set_lr_task *bslt =
                                        (b_set_lr_task *)L2_set_lr_batch
                                            ->push_task_zero_copy(
                                                newnode_addr.id, -1, true);
                                    bslt->addr = newnode_addr;
                                    bslt->left = left;
                                    bslt->right =
                                        (j == r - 1)
                                            ? right
                                            : l2_addrs[node_id[nodeht][j + 1]]
                                                      [ht];
                                }
                                left = newnode_addr;
                                if (j == r - 1 &&
                                    not_equal_pptr(right, null_pptr)) {
                                    b_set_lr_task *bslt =
                                        (b_set_lr_task *)L2_set_lr_batch
                                            ->push_task_zero_copy(right.id, -1,
                                                                  true);
                                    bslt->addr = right;
                                    bslt->left = left;
                                    bslt->right = null_pptr;
                                }
                            }
                        }
                    });
                });
            }
            io2->finish_task_batch();
        }
    });

    io->reset();
    io = nullptr;

    // io2->send_task();
    // io2->print_all_buffer(true);
    // exit(0);
    time_nested("build udlr exec", [&]() { ASSERT(!io2->exec()); });
    // print_log(0, true);

    io2->reset();
    io2 = nullptr;

    time_nested("build cache", [&]() { build_cache(); });

    ASSERT(L2_HEIGHT == 3);

    io = alloc_io_manager();
    ASSERT(io == io_managers[0]);
    io->init(epoch_number++);

    ASSERT(CACHE_HEIGHT == 2);
    auto cache_insert_batch_1to3 =
        io->alloc_task_batch(direct, fixed_length, fixed_length,
                             CACHE_INSERT_TSK, sizeof(cache_insert_task), 0);
    {
        auto batch = cache_insert_batch_1to3;
        // new L2-1 to L2-3
        parfor_wrap(0, newnode_count[1], [&](size_t nid) {
            int i = node_id[1][nid];
            if (heights[i] != 2) {
                return;
            }
            ASSERT(predecessor_record[i] == L2_HEIGHT);
            pptr source = insert_path_addrs[i][2];
            for (int preid = nid - 1; preid >= 0; preid--) {
                int pre = node_id[1][preid];
                ASSERT(heights[pre] >= 1);
                if (not_equal_pptr(source, insert_path_addrs[pre][2])) {
                    break;
                }
                if (heights[pre] > 3) {
                    // no need to do cache insert
                    return;
                }
                if (heights[pre] == 3 &&
                    equal_pptr(insert_path_addrs[i][1],
                               insert_path_addrs[pre][1])) {
                    return;
                }
            }

            cache_insert_task *cit =
                (cache_insert_task *)batch->push_task_zero_copy(source.id, -1,
                                                                true, NULL);
            *cit = (cache_insert_task){.addr = source,
                                       .key = keys[i],
                                       .t_addr = l2_addrs[i][0],
                                       .height = 1};
        });
        io->finish_task_batch();
    };

    time_nested("cache insert", [&]() { ASSERT(!io->exec()); });
}

inline void remove(int length, int64_t *remove_keys) {
    auto keys = deduplication(remove_keys, length);
    cout << "********************************" << length
         << " after dedup ********************************" << endl;
    auto heights = parlay::sequence<int>(length);

#ifdef REMOVE_DEBUG
    for (int i = 0; i < length; i++) {
        printf("key[%d]=%llx\n", i, keys[i]);
    }
#endif

    {  // get height, also remove from hash tables
        dpu_binary_switch_to(dpu_binary::delete_binary);

        auto io = alloc_io_manager();
        ASSERT(io == io_managers[0]);
        io->init(epoch_number++);
        auto keys_target = parlay::tabulate(length, [&](uint32_t i) {
            return hash_to_dpu(keys[i], 0, nr_of_dpus);
        });

        auto get_height_batch = io->alloc_task_batch(
            direct, fixed_length, fixed_length, P_GET_HEIGHT_TSK,
            sizeof(p_get_height_task), sizeof(p_get_height_reply));
        time_nested("taskgen", [&]() {
            auto batch = get_height_batch;
            parfor_wrap(0, length, [&](size_t i) {
                int64_t key = keys[i];
                int target = keys_target[i];
                p_get_height_task *pght =
                    (p_get_height_task *)batch->push_task_zero_copy(
                        target, -1, true, op_taskpos + i);
                *pght = (p_get_height_task){.key = key};
            });
            io->finish_task_batch();
        });

        time_nested("exec", [&]() { io->exec(); });

        time_nested("get result", [&]() {
            auto batch = get_height_batch;
            parfor_wrap(0, length, [&](size_t i) {
                int64_t key = keys[i];
                int target = keys_target[i];
                p_get_height_reply *pghr =
                    (p_get_height_reply *)batch->get_reply(op_taskpos[i],
                                                           target);
                heights[i] = pghr->height;
            });
        });
        io->io_manager_state = idle;
    }

    auto valid_remove = parlay::tabulate(
        length, [&](int i) -> bool { return heights[i] >= 0; });

    keys = parlay::pack(keys, valid_remove);
    heights = parlay::pack(heights, valid_remove);
    length = keys.size();
    cout << "********************************" << length
         << " to remove ********************************" << endl;
    ASSERT(keys.size() == heights.size());

    time_nested("horizontal reduce",
                [&]() { horizontal_reduce(heights.data(), length); });

    // for (int i = 0; i <= L2_HEIGHT; i ++) {
    //     printf("i=%d\n", i);
    //     for (int j = 0; j < newnode_count[i]; j ++) {
    //         int x = node_id[i][j];
    //         printf("key=%lld height=%d\n", keys[x], height[x]);
    //     }
    //     printf("\n\n");
    // }
    // exit(0);

    auto predecessor_record = parlay::map(heights, [&](int32_t x) {
        return (x >= CACHE_HEIGHT) ? L2_HEIGHT : 1;
    });

    auto predecessor_record_sum_pair = parlay::scan(predecessor_record);

    auto predecessor_record_total = predecessor_record_sum_pair.second;
    auto predecessor_record_prefix_sum = predecessor_record_sum_pair.first;

    auto remove_path_addrs =
        parlay::map(predecessor_record_prefix_sum,
                    [&](int32_t x) { return insert_path_addrs_buf + x; });

#ifdef READ_OPTIMIZED
    ASSERT(predecessor_record_total < BATCH_SIZE * 3);
#else
    ASSERT(predecessor_record_total < BATCH_SIZE * 1.2);
#endif

    printf("\n**** DELETE PREDECESSOR ****\n");
    time_nested("predecessor", [&]() {
        predecessor(predecessor_insert, keys.size(), keys.data(),
                    predecessor_record.data(), remove_path_addrs.data());
    });

    // if (epoch_number > 136) {
#ifdef REMOVE_DEBUG
    for (int i = 0; i < length; i++) {
        int ht = min(L2_HEIGHT, predecessor_record[i]);
        for (int j = 0; j < ht; j++) {
            printf("remove_path_addrs[%d][%d]=%lx\n", i, j,
                   pptr_to_int64(remove_path_addrs[i][j]));
        }
    }
#endif
    //     exit(0);
    // }

    dpu_binary_switch_to(dpu_binary::delete_binary);

    auto remove_truncate_taskpos =
        parlay::map(predecessor_record_prefix_sum,
                    [&](int32_t x) { return insert_truncate_taskpos_buf + x; });

    {
        auto io = alloc_io_manager();
        ASSERT(io == io_managers[0]);
        io->init(epoch_number++);

        IO_Task_Batch *L2_get_u_batch = nullptr;
        time_nested("L2 get u", [&]() {
            int ht = L2_HEIGHT;
            int nodelen = newnode_count[ht];
            if (nodelen == 0) {
                return;
            }
            L2_get_u_batch = io->alloc_task_batch(
                direct, fixed_length, fixed_length, B_GET_U_TSK,
                sizeof(b_get_u_task), sizeof(b_get_u_reply));

            parfor_wrap(0, nodelen, [&](size_t t) {
                int i = node_id[ht][t];
                ASSERT(heights[i] > L2_HEIGHT);
                pptr addr = remove_path_addrs[i][ht - 1];
                b_get_u_task *bgut =
                    (b_get_u_task *)L2_get_u_batch->push_task_zero_copy(
                        addr.id, -1, true, op_taskpos + i);
#ifdef REMOVE_DEBUG
                printf("get u for i=%d at pim=%d at pos=%d\n", i, addr.id,
                       op_taskpos[i]);
#endif
                bgut->addr = addr;
            });
            // #ifdef REMOVE_DEBUG
            //                     printf("L2remove: addr=%llx\tht=%d\n",
            //                     pptr_to_int64(addr),
            //                            ht);
            //                     for (int j = 0; j < len; j++) {
            //                         printf("%llx ", brt->keys[j]);
            //                     }
            //                     printf("\n\n");
            // #endif
            io->finish_task_batch();
        });

        auto L2_remove_batch = io->alloc_task_batch(
            direct, variable_length, fixed_length, B_REMOVE_TSK, -1, 0);
        time_nested("L2 remove", [&]() {
            for (int ht = 0; ht < L2_HEIGHT; ht++) {
                int nodelen = newnode_count[ht];
                auto task_start = parlay::tabulate(nodelen, [&](int t) -> bool {
                    if (t == 0) {
                        return true;
                    }
                    int i = node_id[ht][t];
                    int l = node_id[ht][t - 1];
                    ASSERT(l < i);
                    return not_equal_pptr(remove_path_addrs[i][ht],
                                          remove_path_addrs[l][ht]);
                });
                auto l = parlay::pack_index(task_start);
                int llen = l.size();
                parfor_wrap(0, llen, [&](size_t t) {
                    int ll = l[t];
                    int rr = (t == llen - 1) ? nodelen : l[t + 1];
                    int len = rr - ll;
                    int i = node_id[ht][ll];
                    if (heights[i] >= ht + 1) {
                        pptr addr = remove_path_addrs[i][ht];
                        b_remove_task *brt =
                            (b_remove_task *)
                                L2_remove_batch->push_task_zero_copy(
                                    addr.id, S64(2 + len), true);
                        brt->addr = addr;
                        brt->len = rr - ll;
                        for (int j = 0; j < len; j++) {
                            brt->keys[j] = keys[node_id[ht][ll + j]];
                        }
#ifdef REMOVE_DEBUG
                        printf("L2remove: addr=%llx\tht=%d\n",
                               pptr_to_int64(addr), ht);
                        for (int j = 0; j < len; j++) {
                            printf("%llx ", brt->keys[j]);
                        }
                        printf("\n\n");
#endif
                    }
                });
            }
            io->finish_task_batch();
        });

        int expected_get_node_reply = S64(L2_SIZE * 2 + 4);
        auto L2_remove_getnode_batch = io->alloc_task_batch(
            direct, fixed_length, variable_length, B_REMOVE_GET_NODE_TSK,
            sizeof(b_remove_get_node_task), expected_get_node_reply);
        time_nested("L2 remove get node taskgen", [&]() {
            for (int ht = 0; ht < L2_HEIGHT; ht++) {
                int sht = ht + 1;  // (height-1) of the node to cause merging
                parfor_wrap(0, newnode_count[sht], [&](size_t t) {
                    int i = node_id[sht][t];
                    ASSERT(heights[i] > sht);
                    pptr addr = remove_path_addrs[i][ht];
                    b_remove_get_node_task *brgnt =
                        (b_remove_get_node_task *)
                            L2_remove_getnode_batch->push_task_zero_copy(
                                addr.id, -1, true,
                                &(remove_truncate_taskpos[i][ht]));
                    brgnt->addr = addr;
#ifdef REMOVE_DEBUG
                    printf("L2removeget: addr=%llx\t sht=%d ht=%d\n",
                           pptr_to_int64(addr), heights[i], ht);
#endif
                });
            }
            io->finish_task_batch();
        });

        time_nested("L2 remove exec", [&]() { ASSERT(io->exec()); });

        auto io2 = alloc_io_manager();
        ASSERT(io2 == io_managers[1]);
        io2->init(epoch_number++);

        auto io3 = alloc_io_manager();
        ASSERT(io3 == io_managers[2]);
        io3->init(epoch_number++);

        IO_Task_Batch *L3_remove_batch = nullptr;
        time_nested("L3 remove", [&]() {
            int ht = L2_HEIGHT;
            int nodelen = newnode_count[ht];
            if (nodelen == 0) {
                return;
            }
            L3_remove_batch =
                io2->alloc_task_batch(broadcast, fixed_length, fixed_length,
                                      L3_REMOVE_TSK, sizeof(L3_remove_task), 0);
            for (int t = 0; t < nodelen; t++) {
                int i = node_id[ht][t];
                pptr addr = remove_path_addrs[i][ht - 1];
                b_get_u_reply *bgur =
                    (b_get_u_reply *)L2_get_u_batch->get_reply(op_taskpos[i],
                                                               addr.id);
                L3_remove_task *trt =
                    (L3_remove_task *)L3_remove_batch->push_task_zero_copy(
                        -1, -1, false);
                trt->addr = bgur->up;
                trt->key = keys[i];
#ifdef REMOVE_DEBUG
                printf(
                    "L3 remove: i=%d\tpim=%d\tkey=%llx\taddr=%llx\tup=%llx\n",
                    i, addr.id, keys[i], pptr_to_int64(addr),
                    pptr_to_int64(bgur->up));
#endif
                ASSERT(not_equal_pptr(bgur->up, null_pptr));
            }
            io2->finish_task_batch();
        });

        auto is_merger = [&](int ht, int t, pptr left) {
            if (t == 0) {
                return true;
            }
            int sht = ht + 1;
            int i = node_id[sht][t];
            int l = node_id[sht][t - 1];
            pptr addr = remove_path_addrs[i][ht];
            // b_remove_get_node_reply *brgnr =
            //     (b_remove_get_node_reply *)
            //         L2_remove_getnode_batch->get_reply(
            //             remove_truncate_taskpos[i][ht], addr.id);
            // pptr left = brgnr->left;
            ASSERT(not_equal_pptr(remove_path_addrs[i][ht],
                                  remove_path_addrs[l][ht]));
            return not_equal_pptr(left, remove_path_addrs[l][ht]);
        };

        auto L2_insert_batch = io2->alloc_task_batch(
            direct, variable_length, fixed_length, B_INSERT_TSK, -1, 0);
        auto cache_insert_batch =
            io3->alloc_task_batch(direct, variable_length, fixed_length,
                                  CACHE_MULTI_INSERT_TSK, -1, 0);
        time_nested("L2 insert taskgen", [&]() {
            for (int ht = 0; ht < L2_HEIGHT; ht++) {
                int sht = ht + 1;  // (height-1) of the node to cause merging
                auto task_start =
                    parlay::sequence<bool>(newnode_count[sht], false);

                auto siz = parlay::sequence<int>(newnode_count[sht], 0);
                auto taskpos = parlay::sequence<int>(newnode_count[sht], -1);

                parfor_wrap(0, newnode_count[sht], [&](size_t t) {
                    int i = node_id[sht][t];
                    ASSERT(heights[i] > sht);
                    pptr addr = remove_path_addrs[i][ht];
                    b_remove_get_node_reply *brgnr =
                        (b_remove_get_node_reply *)
                            L2_remove_getnode_batch->get_reply(
                                remove_truncate_taskpos[i][ht], addr.id);

                    int replen = brgnr->len;

#ifdef REMOVE_DEBUG
                    printf("L2 insert prepare: key=%llx\n", keys[i]);
                    printf("addr=%llx\tht=%d\n", pptr_to_int64(addr), ht);
                    printf("left=%llx\tright=%llx\n",
                           pptr_to_int64(brgnr->left),
                           pptr_to_int64(brgnr->right));
                    for (int j = 0; j < replen * 2; j++) {
                        printf("%llx ", brgnr->vals[j]);
                    }
                    printf("\n\n");
#endif
                    task_start[t] = is_merger(ht, t, brgnr->left);

                    bool found = false;
                    int64_t *repkeys = brgnr->vals;
                    pptr *repaddrs = (pptr *)(brgnr->vals + replen);
                    int reducedlen = replen;

                    for (int j = 0; j < reducedlen; j++) {  // keys in reply
                        ASSERT(repkeys[j] > keys[i]);
                        while (repkeys[j] == INT64_MAX) {
                            if (j >= reducedlen) {
                                break;
                            }
                            repkeys[j] = repkeys[reducedlen - 1];
                            repaddrs[j] = repaddrs[reducedlen - 1];
                            reducedlen--;
                        }
                    }
                    for (int j = 0; j < reducedlen; j++) {
                        brgnr->vals[j + reducedlen] =
                            pptr_to_int64(repaddrs[j]);
                    }
                    siz[t] = brgnr->len = replen = reducedlen;

#ifdef REMOVE_DEBUG
                    printf("L2 insert after: key=%llx merger=%d\n", keys[i],
                           (int)task_start[t]);
                    printf("addr=%llx\tht=%d\n", pptr_to_int64(addr), ht);
                    printf("left=%llx\tright=%llx\n",
                           pptr_to_int64(brgnr->left),
                           pptr_to_int64(brgnr->right));
                    for (int j = 0; j < replen * 2; j++) {
                        printf("%llx ", brgnr->vals[j]);
                    }
                    printf("\n\n");
#endif
                });

                parlay::scan_inclusive_inplace(siz, add_scan<int>());

                // auto task_start =
                // parlay::delayed_tabulate(newnode_count[sht], [&](int t) ->
                // bool{
                //     return is_merger(ht, t);
                // });

                auto l = parlay::pack_index(task_start);
                int llen = l.size();
                if (llen == 0) continue;
                auto taskptr = parlay::sequence<b_insert_task *>(llen, nullptr);
                parfor_wrap(0, llen, [&](size_t t) {
                    int ll = l[t];
                    int rr = (t == llen - 1) ? newnode_count[sht] : l[t + 1];
                    int cursiz = siz[rr - 1];
                    cursiz -= (t == 0) ? 0 : siz[ll - 1];
                    int i = node_id[sht][ll];
                    pptr addr = remove_path_addrs[i][ht];
                    b_remove_get_node_reply *brgnr =
                        (b_remove_get_node_reply *)
                            L2_remove_getnode_batch->get_reply(
                                remove_truncate_taskpos[i][ht], addr.id);
                    {
                        pptr target = brgnr->left;
                        b_insert_task *bit =
                            (b_insert_task *)
                                L2_insert_batch->push_task_zero_copy(
                                    target.id, S64(2 + 2 * cursiz), true);
                        bit->addr = target;
                        bit->len = cursiz;
                        taskptr[t] = bit;
                        taskpos[l[t]] = t;
                    }
                    if (rr - ll >
                        1) {  // prepare correct right pointers to set lr
                        int ri = node_id[sht][rr - 1];
                        pptr addrr = remove_path_addrs[ri][ht];
                        b_remove_get_node_reply *brgnrr =
                            (b_remove_get_node_reply *)
                                L2_remove_getnode_batch->get_reply(
                                    remove_truncate_taskpos[ri][ht], addrr.id);
                        brgnr->right = brgnrr->right;
#ifdef REMOVE_DEBUG
                        printf("L2 lr after: key=%llx\tleft=%llx\tright=%llx\n",
                               keys[i], brgnr->left, brgnr->right);
#endif
                    }
                });
                parlay::scan_inclusive_inplace(taskpos, copy_scan<int>());

                parfor_wrap(0, newnode_count[sht], [&](size_t t) {
                    int tp = taskpos[t];
                    int offset = (t == 0) ? 0 : siz[t - 1];
                    offset -= (l[tp] == 0) ? 0 : siz[l[tp] - 1];
                    b_insert_task *bit = taskptr[tp];
                    int i = node_id[sht][t];
                    pptr addr = remove_path_addrs[i][ht];
                    b_remove_get_node_reply *brgnr =
                        (b_remove_get_node_reply *)
                            L2_remove_getnode_batch->get_reply(
                                remove_truncate_taskpos[i][ht], addr.id);
                    ASSERT(not_equal_pptr(brgnr->left, null_pptr));
                    if (l[tp] != t) {
                        brgnr->left = brgnr->right =
                            null_pptr;  // marked as internal nodes
                    }
                    for (int j = 0; j < brgnr->len; j++) {
                        bit->vals[offset + j] = brgnr->vals[j];
                        bit->vals[offset + j + bit->len] =
                            brgnr->vals[brgnr->len + j];
                    }
                });

#ifdef REMOVE_DEBUG
                for (int i = 0; i < llen; i++) {
                    b_insert_task *bit = taskptr[i];
                    printf("L2 insert: addr=%llx len=%lld\n", bit->addr,
                           bit->len);
                    for (int i = 0; i < bit->len * 2; i++) {
                        printf("%llx ", bit->vals[i]);
                    }
                    printf("\n\n");
                }
#endif
                ASSERT(CACHE_HEIGHT == 2);
                if (ht != 1) {
                    continue;
                }
                parfor_wrap(0, llen, [&](size_t t) {
                    int ll = l[t];
                    int rr = (t == llen - 1) ? newnode_count[sht] : l[t + 1];
                    int i = node_id[sht][ll];
                    b_insert_task *bit = taskptr[t];
                    // if (bit->len == 0) return;
                    int lpos = ll - 1;
                    // deletions from node_id[sht][lpos + 1] to
                    // node_id[sht][ll] share the same L2 root
                    ASSERT(heights[i] >= L2_HEIGHT);
                    pptr target;
                    if (heights[i] > L2_HEIGHT) {
                        pptr addr = remove_path_addrs[i][2];
                        b_remove_get_node_reply *brgnr =
                            (b_remove_get_node_reply *)
                                L2_remove_getnode_batch->get_reply(
                                    remove_truncate_taskpos[i][2], addr.id);
                        target = brgnr->left;
                        // printf("tgt=%llx\n", pptr_to_int64(target));
                        lpos = -1;
                    } else {
                        for (; lpos >= 0; lpos--) {
                            int li = node_id[sht][lpos];
                            if (not_equal_pptr(remove_path_addrs[i][2],
                                               remove_path_addrs[li][2])) {
                                lpos = -1;
                                break;
                            }
                            if (heights[li] > L2_HEIGHT) {
                                break;
                            }
                        }
                        if (lpos == -1) {
                            target = remove_path_addrs[i][2];
                        }
                    }
                    if (lpos != -1) return;
                    cache_multi_insert_task *cmit =
                        (cache_multi_insert_task *)
                            cache_insert_batch->push_task_zero_copy(
                                target.id, S64(bit->len * 2 + 3), true);
                    cmit->addr = target;
                    cmit->height = 1;
                    cmit->len = bit->len;
                    for (int j = 0; j < cmit->len; j++) {
                        cmit->vals[j] = bit->vals[j];
                        cmit->vals[j + cmit->len] = bit->vals[j + bit->len];
                    }
#ifdef REMOVE_DEBUG
                    printf("L2 cache insert: addr=%llx len=%lld\n", cmit->addr,
                           cmit->len);
                    for (int i = 0; i < cmit->len * 2; i++) {
                        printf("%llx ", cmit->vals[i]);
                    }
                    printf("\n\n");
#endif
                });
                io3->finish_task_batch();
            }
            io2->finish_task_batch();
        });

        auto L2_set_lr_batch =
            io2->alloc_task_batch(direct, fixed_length, fixed_length,
                                  B_SET_LR_TSK, sizeof(b_set_lr_task), 0);
        time_nested("L2 set lr", [&]() {
            for (int ht = 0; ht < L2_HEIGHT; ht++) {
                int sht = ht + 1;
                int nodecount = newnode_count[sht];
                parfor_wrap(0, nodecount, [&](size_t t) {
                    int i = node_id[sht][t];
                    pptr addr = remove_path_addrs[i][ht];
                    b_remove_get_node_reply *brgnr =
                        (b_remove_get_node_reply *)
                            L2_remove_getnode_batch->get_reply(
                                remove_truncate_taskpos[i][ht], addr.id);
                    if (equal_pptr(brgnr->left, null_pptr)) {
                        return;
                    }
#ifdef REMOVE_DEBUG
                    printf("L2 set lr: key=%llx\tleft=%llx\tright=%llx\n",
                           keys[i], brgnr->left, brgnr->right);
#endif
                    pptr left = brgnr->left;
                    pptr right = brgnr->right;

                    b_set_lr_task *bslt =
                        (b_set_lr_task *)L2_set_lr_batch->push_task_zero_copy(
                            left.id, -1, true);
                    *bslt = (b_set_lr_task){
                        .addr = left, .left = null_pptr, .right = right};

                    if (not_equal_pptr(brgnr->right, null_pptr)) {
                        bslt = (b_set_lr_task *)L2_set_lr_batch
                                   ->push_task_zero_copy(right.id, -1, true);
                        *bslt = (b_set_lr_task){
                            .addr = right, .left = left, .right = null_pptr};
                    }
                });
            }
            io2->finish_task_batch();
        });

        time_nested("L2 remove insert exec", [&]() { ASSERT(!io2->exec()); });

        io->reset();
        io2->reset();

        ASSERT(CACHE_HEIGHT == 2);
        build_cache();

        time_nested("L2 cache remove", [&]() {
            auto cache_remove_batch = io3->alloc_task_batch(
                direct, fixed_length, fixed_length, CACHE_REMOVE_TSK,
                sizeof(cache_remove_task), 0);
            int ht = 1;
            int nodecount = newnode_count[ht];
            parfor_wrap(0, newnode_count[ht], [&](size_t t) {
                int i = node_id[ht][t];
                pptr target = remove_path_addrs[i][2];
                cache_remove_task *crt =
                    (cache_remove_task *)cache_remove_batch
                        ->push_task_zero_copy(target.id, -1, true);
                *crt = (cache_remove_task){
                    .addr = target, .key = keys[i], .height = 1};
            });
            io3->finish_task_batch();
        });

        time_nested("cache update", [&]() { ASSERT(!io3->exec()); });
    }
}

inline void print_statistics() {
    auto io = alloc_io_manager();
    ASSERT(io == io_managers[0]);
    io->init(epoch_number++);

    auto statistic_batch =
        io->alloc_task_batch(broadcast, fixed_length, fixed_length,
                             STATISTICS_TSK, sizeof(statistic_task), 0);

    auto batch = statistic_batch;
    statistic_task *st =
        (statistic_task *)batch->push_task_zero_copy(-1, -1, false);
    *st = (statistic_task){.id = INVALID_DPU_ID};
    io->finish_task_batch();
    io->exec();
    print_log(0, true);
    io->io_manager_state = idle;
}