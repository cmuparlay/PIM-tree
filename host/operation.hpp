#pragma once
#include <parlay/primitives.h>

#include <mutex>

#include "compile.hpp"
#include "debug.hpp"
#include "dpu_control.hpp"
#include "operation_def.hpp"
#include "oracle.hpp"
#include "papi_counters.hpp"
#include "task.hpp"
#include "task_framework_host.hpp"
#include "value.hpp"
#include "add_parlay_lib.hpp"
using namespace std;

static inline int hh(int64_t key, uint64_t height, uint64_t M) {
    uint64_t v = parlay::hash64((uint64_t)key) + height;
    v = parlay::hash64(v);
    return v % M;
}

static inline int hash_to_dpu(int64_t key, uint64_t height, uint64_t M) {
    return hh(key, height, M);
}

class pim_skip_list {
   public:

    // definition & const
    enum predecessor_type {
        predecessor_insert,
        predecessor_only
    };
    static constexpr int push_pull_limit = L2_SIZE * 2;
    const double bias_limit = 3;
    const int max_l3_height = 14;

    // static member
    struct dpu_memory_regions {
        uint32_t bbuffer_start;
        uint32_t bbuffer_end;
        uint32_t pbuffer_start;
        uint32_t pbuffer_end;
    };
    inline static dpu_memory_regions dmr;
    static bool in_bbuffer(uint32_t addr) {
        return addr >= dmr.bbuffer_start && addr < dmr.bbuffer_end;
    }

    static bool in_pbuffer(uint32_t addr) {
        return addr >= dmr.pbuffer_start && addr < dmr.pbuffer_end;
    }

    static bool valid_b_pptr(const pptr &x) {
        return valid_pptr(x) && in_bbuffer(x.addr);
    }

    static bool valid_p_pptr(const pptr &x) {
        return valid_pptr(x) && in_pbuffer(x.addr);
    }
    inline static mutex mut;

    // parameters for evaluation
    int push_pull_limit_dynamic = L2_SIZE;

    // input
    int length;
    int length_before_deduplication;
    key_value kv_input[BATCH_SIZE];
    int64_t i64_input[BATCH_SIZE];
    uint32_t back_trace_offset_startpos[BATCH_SIZE];
    uint32_t back_trace_offset[BATCH_SIZE];

    // output
    key_value kv_output[BATCH_SIZE];
    int64_t i64_output[BATCH_SIZE];

    // io internal
    pair<int, int64_t> keys_with_offset_sorted[BATCH_SIZE];

    // member
    pptr op_addrs[BATCH_SIZE];
    int32_t op_taskpos[BATCH_SIZE];
    inline static int l2_root_count[NR_DPUS];
    int newnode_offset_buffer[BATCH_SIZE * 3];
    int newnode_count[L2_HEIGHT + 2];
    int *node_id[L2_HEIGHT + 2];

    pptr insert_path_addrs_buf[BATCH_SIZE * 2];
    pptr l2_addrs_buf[BATCH_SIZE * 2];
    int l2_addrs_taskpos_buf[BATCH_SIZE * 2];
    int insert_truncate_taskpos_buf[BATCH_SIZE * 2];

    void build_cache() {
        for (int round = 0; round < 1; round++) {
            auto io = alloc_io_manager();
            // ASSERT(io == io_managers[0]);
            io->init();

            auto cache_init_request_batch = io->alloc_task_batch(
                direct, fixed_length, variable_length, CACHE_INIT_REQ_TSK,
                sizeof(cache_init_request_task),
                sizeof(cache_init_request_reply));  // !!! ??? !!! wrong reply
                                                    // length
            time_nested("cache init request taskgen", [&]() {
                parfor_wrap(0, nr_of_dpus, [&](size_t i) {
                    cache_init_request_task *cirt =
                        (cache_init_request_task *)cache_init_request_batch
                            ->push_task_zero_copy(i, -1, false, NULL);
                    *cirt = (cache_init_request_task){{.nothing = 0}};
                });
                io->finish_task_batch();
            });

            time_nested("cache init request", [&]() { ASSERT(io->exec()); });

            auto cache_init_len = parlay::sequence<int64_t>(nr_of_dpus);

            for (int i = 0; i < nr_of_dpus; i++) {
                cache_init_request_reply *cirr =
                    (cache_init_request_reply *)
                        cache_init_request_batch->get_reply(0, i);
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

            auto cache_init_addrs =
                parlay::sequence<pptr>(cache_init_len_total);
            auto cache_init_requests =
                parlay::sequence<pptr>(cache_init_len_total);
            auto cache_init_taskoffset =
                parlay::sequence<int>(cache_init_len_total);

            parfor_wrap(0, nr_of_dpus, [&](size_t i) {
                cache_init_request_reply *cirr =
                    (cache_init_request_reply *)
                        cache_init_request_batch->get_reply(0, i);

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

            io = alloc_io_manager();
            // ASSERT(io == io_managers[0]);
            io->init();

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
                *bgnt = (b_get_node_task){{.addr = request}};
            });
            io->finish_task_batch();

            time_nested("cache init getnode", [&]() { ASSERT(io->exec()); });

            auto io2 = alloc_io_manager();
            ASSERT(io2 == io_managers[1]);
            io2->init();

            auto cache_init_newnode =
                io2->alloc_task_batch(direct, variable_length, fixed_length,
                                      CACHE_NEWNODE_TSK, -1, 0);
            {
                parfor_wrap(0, cache_init_len_total, [&](size_t i) {
                    pptr addr = cache_init_addrs[i];
                    pptr request = cache_init_requests[i];

                    b_get_node_reply *bgnr =
                        (b_get_node_reply *)cache_init_getnode_batch->get_reply(
                            cache_init_taskoffset[i], request.id);
                    int nnlen = bgnr->len;

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

    void init_skiplist() {
        dpu_binary_switch_to(dpu_binary::init_binary);

        int l3_height = max_l3_height;
        ASSERT(l3_height > 0);

        printf("\n********** INIT SKIP LIST **********\n");

        pptr l3node = null_pptr;
        pptr l2nodes[L2_HEIGHT] = {null_pptr};
        pptr l1node = null_pptr;

        // insert nodes
        time_nested("nodes", [&]() {
            auto io = alloc_io_manager();
            // ASSERT(io == io_managers[0]);
            io->init();

            auto L2_init_batch =
                io->alloc<b_newnode_task, b_newnode_reply>(direct);
            {
                auto batch = L2_init_batch;
                for (int ht = 0; ht < L2_HEIGHT; ht++) {
                    int target = hash_to_dpu(INT64_MIN, ht, nr_of_dpus);
                    b_newnode_task *bnt =
                        (b_newnode_task *)batch->push_task_zero_copy(
                            target, -1, true, op_taskpos + ht);
                    *bnt = (b_newnode_task){{.height = ht}};
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
                    (p_newnode_task *)batch->push_task_zero_copy(target, -1,
                                                                 false);
                pnt->key = INT64_MIN;
                pnt->height = l3_height + L2_HEIGHT;
                io->finish_task_batch();
            }

            time_nested("exec", [&]() { ASSERT(io->exec()); });

            {
                auto batch = L2_init_batch;
                for (int ht = 0; ht < L2_HEIGHT; ht++) {
                    int target = hash_to_dpu(INT64_MIN, ht, nr_of_dpus);
                    b_newnode_reply *rep = (b_newnode_reply *)batch->get_reply(
                        op_taskpos[ht], target);
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
            io->reset();
        });

        memset(l2_root_count, 0, sizeof(l2_root_count));
        l2_root_count[l2nodes[L2_HEIGHT - 1].id]++;

        // build up down pointers
        time_nested("ud ptrs", [&]() {
            auto io = alloc_io_manager();
            // ASSERT(io == io_managers[0]);
            io->init();
            auto L3_init_batch =
                io->alloc<L3_init_task, empty_task_reply>(broadcast);
            {
                auto batch = L3_init_batch;
                L3_init_task *tit =
                    (L3_init_task *)batch->push_task_zero_copy(-1, -1, false);
                *tit = (L3_init_task){{.key = INT64_MIN,
                                       .height = l3_height,
                                       .down = l2nodes[L2_HEIGHT - 1]}};
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
            time_nested("exec", [&]() { ASSERT(!io->exec()); });
            io->reset();
        });

        dpu_binary_switch_to(dpu_binary::insert_binary);
        time_nested("build cache", [&]() { build_cache(); });
    }

    void init() {
        time_nested("init_skiplist", [&]() {
            {
                cpu_coverage_timer->start();
                init_skiplist();
                cpu_coverage_timer->end();
                cpu_coverage_timer->reset();
                pim_coverage_timer->reset();
            }
        });
    }

    template <class TT>
    struct copy_scan {
        using T = TT;
        copy_scan() : identity(0) {}
        T identity;
        static T f(T a, T b) { return (b == -1) ? a : b; }
    };

    template <class TT>
    struct add_scan {
        using T = TT;
        add_scan() : identity(0) {}
        T identity;
        static T f(T a, T b) { return a + b; }
    };

    struct key_addr {
        int64_t key;
        pptr addr;
    };

    // 27 bytes per element
    bool predecessor_l2_one_round_task_start[BATCH_SIZE];
    uint32_t predecessor_l2_one_round_ll[BATCH_SIZE]; // reused by predecessor_core_ll
    uint32_t predecessor_l2_one_round_rr[BATCH_SIZE];
    bool predecessor_l2_one_round_split_start[BATCH_SIZE];
    uint32_t predecessor_l2_one_round_split_ll[BATCH_SIZE];
    bool predecessor_l2_one_round_search_start[BATCH_SIZE];
    uint32_t predecessor_l2_one_round_search_ll[BATCH_SIZE];
    int predecessor_l2_one_round_this_ll[BATCH_SIZE];
    uint32_t predecessor_l2_one_round_active_pos[BATCH_SIZE];

    bool predecessor_l2_one_round(int ht, int length, int64_t *keys,
                                  pptr *op_addrs, int limit, bool split,
                                  bool search, bool shadow_shortcut,
                                  predecessor_type type, int32_t *heights,
                                  pptr **paths) {
        ASSERT(split || search);

#ifdef VERBOSE
        cout << ht << endl;
#endif

        // task starts
        time_start("init");
        auto task_start = parlay::make_slice(predecessor_l2_one_round_task_start,
                predecessor_l2_one_round_task_start + length);
        {
            task_start[0] = true;
            parlay::parallel_for(1, length, [&](size_t i) {
                task_start[i] = (not_equal_pptr(op_addrs[i - 1], op_addrs[i]));
            });
        }

        auto ll_full_slice = parlay::make_slice(predecessor_l2_one_round_ll, predecessor_l2_one_round_ll + BATCH_SIZE);
        auto llen = parlay::pack_index_into(task_start, ll_full_slice);
        auto ll = ll_full_slice.cut(0, llen);

        if (llen == 0) {
            time_end("init");
            return false;
        }

        if (shadow_shortcut) {
            parlay::parallel_for(0, llen, [&](size_t i) {
                int l = ll[i];
                int r = (i == llen - 1) ? length : ll[i + 1];
                for (int x = l; x < r; x += limit) {
                    ASSERT(equal_pptr(op_addrs[x], op_addrs[l]));
                    task_start[x] = true;
                }
            });
            llen = pack_index_into(task_start, ll_full_slice);
            ll = ll_full_slice.cut(0, llen);
        }

        auto rr = parlay::make_slice(predecessor_l2_one_round_rr, predecessor_l2_one_round_rr + length);
        { // initialize rr[]
            parlay::parallel_for(0, length, [&](size_t i) {
                rr[i] = -1;
            });
            parlay::parallel_for(0, llen - 1, [&](size_t i) {
                rr[ll[i]] = ll[i + 1];
            });
            rr[ll[llen - 1]] = length;
        }

        auto split_start = parlay::make_slice(predecessor_l2_one_round_split_start,
                predecessor_l2_one_round_split_start + length);
        {
            parlay::parallel_for(0, length, [&](size_t i) {
                split_start[i] = task_start[i] && in_bbuffer(op_addrs[i].addr) &&
                   (rr[i] - i > limit);
            });
        }

        auto split_ll = parlay::make_slice(predecessor_l2_one_round_split_ll,
                predecessor_l2_one_round_split_ll + BATCH_SIZE);
        uint32_t split_llen = 0;

        if (split) {
            split_llen = parlay::pack_index_into(split_start, split_ll);
            split_ll = split_ll.cut(0, split_llen);

            if (!search && (split_llen == 0)) {
                time_end("init");
                return false;
            }
        }

        ASSERT(!shadow_shortcut || split_llen == 0);

        auto search_start = parlay::make_slice(predecessor_l2_one_round_search_start,
                predecessor_l2_one_round_search_start + length);
        {
            parlay::parallel_for(0, length, [&](size_t i) {
                search_start[i] = task_start[i] && in_bbuffer(op_addrs[i].addr) &&
                   (rr[i] - i <= limit);
            });
        }

        auto search_ll = parlay::make_slice(predecessor_l2_one_round_search_ll,
                predecessor_l2_one_round_search_ll + BATCH_SIZE);
        uint32_t search_llen = 0;

        if (search) {
            search_llen = parlay::pack_index_into(search_start, search_ll);
            search_ll = search_ll.cut(0, search_llen);

            if (!split && (search_llen == 0)) {
                time_end("init");
                return false;
            }
        }

#ifdef VERBOSE
        printf("llen=%d\nsplit_llen=%d\nsearch_llen=%d\n", llen, split_llen,
               search_llen);
#endif

        if (search_llen == 0) {
            search = false;
        }
        if (split_llen == 0) {
            split = false;
        }

        auto this_ll = parlay::make_slice(predecessor_l2_one_round_this_ll,
                predecessor_l2_one_round_this_ll + length);
        {
            parlay::parallel_for(0, length, [&](size_t i) {
                this_ll[i] = -1;
            });
        }

        parfor_wrap(0, llen, [&](size_t i) {
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

        auto active_pos = parlay::make_slice(predecessor_l2_one_round_active_pos,
                predecessor_l2_one_round_active_pos + length);
        uint32_t active_pos_len = 0;
        {
            active_pos_len = parlay::pack_index_into(
                parlay::delayed_tabulate(
                    length, [&](int i) -> bool { return this_ll[i] >= 0; }),
                active_pos);
            active_pos = active_pos.cut(0, active_pos_len);
        }

        time_end("init");

#ifdef KHB_CPU_DEBUG
        {
            std::mutex mut;
            parfor_wrap(0, length, [&](int i) {
                if (!in_bbuffer(op_addrs[i].addr)) {
                    ASSERT(in_pbuffer(op_addrs[i].addr));
                    ASSERT(!search_start[i]);
                    ASSERT(!split_start[i]);
                    return;
                }
                if (task_start[i]) {
                    if (shadow_shortcut) {
                        if (i > 0 && equal_pptr(op_addrs[i], op_addrs[i - 1])) {
                            bool valid = true;
                            valid = valid && (equal_pptr(op_addrs[i],
                                                         op_addrs[i - limit]));
                            valid = valid && (task_start[i - limit]);
                            ASSERT_EXEC(valid, {
                                mut.lock();
                                for (int x = max(0, i - 20);
                                     x < min(length, i + 20); x++) {
                                    printf("op_addr[%d]=%lx\n", x,
                                           pptr_to_int64(op_addrs[x]));
                                }
                                printf("i=%d opa=%lx opal=%lx\n", i,
                                       pptr_to_int64(op_addrs[i]),
                                       pptr_to_int64(op_addrs[i - limit]));
                                mut.unlock();
                            });
                        }
                    } else {
                        ASSERT(i == 0 ||
                               !equal_pptr(op_addrs[i], op_addrs[i - 1]));
                    }
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
        // ASSERT(io == io_managers[0]);
        io->init();

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

                parfor_wrap(0, split_llen, [&](size_t i) {
                    int l = split_ll[i];
                    b_get_node_task *bgnt =
                        (b_get_node_task *)batch->push_task_zero_copy(
                            op_addrs[l].id, -1, true, op_taskpos + l);
                    *bgnt = (b_get_node_task){.addr = op_addrs[l]};
                });
#ifdef VERBOSE
                printf("get node count = %d\n", split_llen);
#endif
                io->finish_task_batch();
            });
        }

        bool fixed_length_search = (search_llen > (length / 2)) && (ht == 0);

        // search: predecessor only
        IO_Task_Batch *L2_search_batch = nullptr;
        if (search && type == predecessor_only) {
            time_nested("search taskgen", [&]() {
                if (fixed_length_search) {
                    L2_search_batch =
                        io->alloc<b_fixed_search_task, b_fixed_search_reply>(
                            direct);

                    ASSERT((split && L2_search_batch == &io->tbs[1]) ||
                           (!split && L2_search_batch == &io->tbs[0]));
                    auto batch = L2_search_batch;

                    // one buffer filling method. not used.
                    // batch->push_task_from_array_by_isort<true>(
                    //     idx.size(),
                    //     [&](size_t x) {
                    //         int i = idx[x];
                    //         return (b_fixed_search_task){
                    //             {.addr = op_addrs[i], .key = keys[i]}};
                    //     },
                    //     [&](const b_fixed_search_task &t) { return t.addr.id;
                    //     }, make_slice(pos));
                    // parlay::parallel_for(0, idx.size(), [&](size_t x) {
                    //     int i = idx[x];
                    //     op_taskpos[i] = pos[x];
                    // });

                    // the used buffer filling method.
                    parfor_wrap(0, search_llen, [&](size_t i) {
                        int l = search_ll[i];
                        int r = rr[l];
                        int len = r - l;
                        ASSERT(len > 0 && len <= limit);
                        for (int j = l; j < r; j++) {
                            pptr addr = op_addrs[j];
                            b_fixed_search_task *bfst =
                                (b_fixed_search_task *)
                                    batch->push_task_zero_copy(
                                        addr.id, -1, true, op_taskpos + j);
                            *bfst = (b_fixed_search_task){
                                {.addr = addr, .key = keys[j]}};
                        }
                    });
                    io->finish_task_batch();
                } else {
                    int expected_search_reply = S64(2);
                    L2_search_batch = io->alloc_task_batch(
                        direct, variable_length, variable_length, B_SEARCH_TSK,
                        -1, expected_search_reply);
                    ASSERT((split && L2_search_batch == &io->tbs[1]) ||
                           (!split && L2_search_batch == &io->tbs[0]));
                    auto batch = L2_search_batch;
                    // unused buffer filling method
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
                        int len = r - l;
                        ASSERT(len > 0 && len <= limit);
                        b_search_task *bst =
                            (b_search_task *)batch->push_task_zero_copy(
                                op_addrs[l].id, S64(2 + len), true,
                                op_taskpos + l);
                        bst->addr = op_addrs[l];
                        bst->len = len;
                        memcpy(bst->keys, keys + l, sizeof(int64_t) * len);
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
                    B_SEARCH_WITH_PATH_TSK, -1,
                    expected_search_with_path_reply);
                ASSERT((split && L2_search_with_path_batch == &io->tbs[1]) ||
                       (!split && L2_search_with_path_batch == &io->tbs[0]));
                auto batch = L2_search_with_path_batch;
                parfor_wrap(0, search_llen, [&](size_t i) {
                    int l = search_ll[i];
                    int r = rr[l];
                    int len = r - l;
                    ASSERT(len > 0 && len <= limit);
                    b_search_with_path_task *bst =
                        (b_search_with_path_task *)batch->push_task_zero_copy(
                            op_addrs[l].id, S64(2 + len * 2), true,
                            op_taskpos + l);
                    bst->addr = op_addrs[l];
                    ASSERT(valid_b_pptr(op_addrs[l]));
                    bst->len = len;
                    for (int j = 0; j < len; j++) {
                        bst->vals[j] = keys[l + j];
                        bst->vals[j + len] = heights[l + j];
                        ASSERT(heights[l + j] == 1 ||
                               heights[l + j] == L2_HEIGHT);
                    }
                });
                io->finish_task_batch();
            });
        }

        time_nested("exec", [&]() { io->exec(); });

        time_nested("get result", [&]() {
            if (type == predecessor_insert && search) {
                printf("predecessor_insert_search\n");
                parfor_wrap(0, search_llen, [&](size_t i) {
                    int loop_l = search_ll[i];
                    int loop_r = rr[loop_l];
                    int len = loop_r - loop_l;
                    b_search_with_path_reply *bsr =
                        (b_search_with_path_reply *)
                            L2_search_with_path_batch->get_reply(
                                op_taskpos[loop_l], op_addrs[loop_l].id);
                    ASSERT(len <= bsr->len);
                    offset_pptr *task_op = bsr->ops;
                    for (int j = 0; j < bsr->len; j++) {
                        int off = task_op[j].offset + loop_l;
                        pptr ad = (pptr){.id = task_op[j].id,
                                         .addr = task_op[j].addr};
                        ASSERT_EXEC(in_pbuffer(ad.addr) || in_bbuffer(ad.addr),
                                    {
                                        printf("loop_l=%d off=%d pptr=%llx\n",
                                               loop_l, off, pptr_to_int64(ad));
                                    });

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
                                ASSERT_EXEC(k != 0, {
                                    for (int kk = 0; kk < bsr->len; kk++) {
                                        int off = task_op[kk].offset + loop_l;
                                        pptr ad = ((pptr){.id = task_op[kk].id,
                                                   .addr = task_op[kk].addr});
                                        printf("j=%d ad=%lx\n", off,
                                               pptr_to_int64(ad));
                                    }
                                    for (int kk = L2_HEIGHT - 1; kk >= 0; kk--) {
                                        printf("j=%d kk=%d path=%lx\n", off, kk,
                                               pptr_to_int64(paths[off][kk]));
                                    }
                                    printf("j=%d path=%lx\n", off, pptr_to_int64(ad));
                                    fflush(stdout);
                                    dpu_control::print_log([](size_t i) {
                                        (void)i;
                                        return true;
                                    });
                                });
                            }
                        }
                    }
#ifdef KHB_CPU_DEBUG
                    for (int j = loop_l; j < loop_r; j++) {
                        ASSERT_EXEC(valid_pptr(op_addrs[j]), {
                            mut.lock();
                            for (int xx = loop_l; xx < loop_r; xx++) {
                                printf("%d %lx\n", xx,
                                       pptr_to_int64(op_addrs[xx]));
                            }
                            fflush(stdout);
                            mut.unlock();
                        });
                    }
#endif
                });
            }
            parfor_wrap(0, active_pos_len, [&](size_t x) {
                int i = active_pos[x];
                int l = this_ll[i];
                int j = i - l;
                if (search_start[l] && fixed_length_search) {
                    ASSERT(!split_start[l] && search &&
                           type == predecessor_only);
                    b_fixed_search_reply *bfsr =
                        (b_fixed_search_reply *)L2_search_batch->get_reply(
                            op_taskpos[i], op_addrs[i].id);
                    op_addrs[i] = bfsr->addr;
                } else if (search_start[l]) {
                    ASSERT(!split_start[l] && search &&
                           type == predecessor_only);
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
                    ASSERT(valid_pptr(op_addrs[i]));
                } else {
                    ASSERT(false);
                }
            });
        });
        io->reset();

        return true;
    }

    void predecessor_l1(int ht, int length, int64_t *keys, pptr *op_addrs) {
        assert(ht == 0);
        double bias = query_bias(length, op_addrs, nr_of_dpus);
#ifdef VERBOSE
        printf("bias: %lf\n", bias);
#endif
        if (bias < bias_limit) {
            auto io = alloc_io_manager();
            io->init();
            auto L1_search_batch = io->alloc_task_batch(
                direct, fixed_length, fixed_length, B_FIXED_SEARCH_TSK,
                sizeof(b_fixed_search_task), sizeof(b_fixed_search_reply));

            time_nested("taskgen", [&]() {
                auto batch = L1_search_batch;
                auto targets = [&](const b_fixed_search_task &x) {
                    return x.addr.id;
                };
                // another buffer filling method
                // parlay::parallel_for(0, length, [&](size_t i) {
                //     auto bfst =
                //         (b_fixed_search_task *)batch->push_task_zero_copy(
                //             op_addrs[i].id, -1, true, op_taskpos + i);
                //     *bfst = (b_fixed_search_task){
                //         {.addr = op_addrs[i], .key = keys[i]}};
                // });
                batch->push_task_from_array_by_isort<true>(
                    length,
                    [&](size_t i) {
                        return (b_fixed_search_task){
                            {.addr = op_addrs[i], .key = keys[i]}};
                    },
                    targets, make_slice(op_taskpos, op_taskpos + length));
                io->finish_task_batch();
            });

            time_nested("exec", [&]() { ASSERT(io->exec()); });

            time_nested("get result", [&]() {
                auto batch = L1_search_batch;
                parfor_wrap(0, length, [&](size_t i) {
                    b_fixed_search_reply *bfsr =
                        (b_fixed_search_reply *)batch->ith(op_addrs[i].id,
                                                           op_taskpos[i]);
                    op_addrs[i] = bfsr->addr;
                });
            });
            io->reset();
        } else {
            predecessor_l2_one_round(0, length, keys, op_addrs, L2_SIZE, true,
                                     true, false, predecessor_only, NULL, NULL);
            return;
        }
    }

    // block local counts
    uint32_t query_bias_block_local_counts[BATCH_SIZE];
    uint32_t query_bias_counts[NR_DPUS];

    double query_bias(size_t n, pptr *op_addrs, size_t num_buckets) {
        using namespace parlay;

        auto ids = parlay::delayed_seq<uint32_t>(
            n, [&](size_t i) { return op_addrs[i].id; });

        size_t num_blocks = 1 + n * sizeof(uint32_t) /
                                    std::max<size_t>(num_buckets * 500, 5000);
        num_blocks = max(num_blocks, parlay::num_workers());
        size_t block_size = ((n - 1) / num_blocks) + 1;

        auto Keys = make_slice(ids);

        size_t m = num_blocks * num_buckets;

        assert(m < BATCH_SIZE);

        auto block_local_counts = parlay::make_slice(
            query_bias_block_local_counts, query_bias_block_local_counts + m);

        // sort each block
        parallel_for(
            0, num_blocks,
            [&](size_t i) {
                size_t start = (std::min)(i * block_size, n);
                size_t end = (std::min)(start + block_size, n);
                internal::seq_count_(
                    Keys.cut(start, end), Keys.cut(start, end),
                    block_local_counts.begin() + i * num_buckets, num_buckets);
            },
            1);

        auto counts = parlay::make_slice(query_bias_counts,
                                         query_bias_counts + num_buckets);

        parallel_for(
            0, num_buckets,
            [&](size_t i) {
                auto v = 0;
                for (size_t j = 0; j < num_blocks; j++) {
                    v += block_local_counts[j * num_buckets + i];
                }
                counts[i] = v;
            },
            1 + 1024 / num_blocks);

        uint32_t max_count = *(parlay::max_element(counts));
        return (double)max_count * nr_of_dpus / n;
    }

    uint32_t *predecessor_core_ll = predecessor_l2_one_round_ll;  // reuse
    void predecessor_core(predecessor_type type, int32_t *heights = NULL,
                          pptr **paths = NULL) {
        int64_t *keys = i64_input;

#ifdef VERBOSE
        printf("%d\n", length);
        printf("\n********** START PREDECESSOR **********\n");
#endif

        dpu_binary_switch_to(dpu_binary::query_binary);

        papi_start_global_counters(l3counters);
        time_nested("L3", [&]() {
            auto io = alloc_io_manager();
            // ASSERT(io == io_managers[0]);
            io->init();

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

            io->reset();
        });
        papi_stop_global_counters(l3counters);

#ifdef KHB_CPU_DEBUG
        for (int i = 0; i < length; i++) {
            ASSERT_EXEC(valid_b_pptr(op_addrs[i]), {
                print_array("op_addrs", (int64_t *)op_addrs, length, true);
            });
        }
#endif

        papi_start_global_counters(l2counters);

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

        double bias;
        time_nested("count bias", [&]() {
            bias = query_bias(length, op_addrs, nr_of_dpus);
#ifdef VERBOSE
            printf("bias: %lf\n", bias);
#endif
        });

        time_nested("L2", [&]() {
#ifdef SHADOW_SUBTREE
            if (bias < bias_limit) {
                time_nested("Layer2", [&]() {
                    predecessor_l2_one_round(2, length, keys, op_addrs,
                                             push_pull_limit_dynamic * 2, false,
                                             true, true, type, heights,
                                             paths);  // no split
                });
            } else {
                time_nested("Layer2-1/2", [&]() {
                    predecessor_l2_one_round(
                        2, length, keys, op_addrs, push_pull_limit_dynamic,
                        true, false, false, type, heights, paths);
                });
                time_nested("Layer2-2/2", [&]() {
                    predecessor_l2_one_round(1, length, keys, op_addrs,
                                             push_pull_limit_dynamic, true,
                                             true, false, type, heights, paths);
                });
            }
#else
            {
                time_nested("Layer2-1/2", [&]() {
                    predecessor_l2_one_round(2, length, keys, op_addrs, push_pull_limit_dynamic,
                                             true, true, false, type, heights,
                                             paths);
                });
                time_nested("Layer2-2/2", [&]() {
                    predecessor_l2_one_round(1, length, keys, op_addrs, push_pull_limit_dynamic,
                                             true, true, false, type, heights,
                                             paths);
                });
            }
#endif
            papi_stop_global_counters(l2counters);

            papi_start_global_counters(l1counters);
            if (type == predecessor_only) {
                time_nested("Layer1", [&]() {
                    predecessor_l1(0, length, keys, op_addrs);
                });
            }
            papi_stop_global_counters(l1counters);
        });

#ifdef KHB_CPU_DEBUG
        if (type == predecessor_insert) {
            for (int i = 0; i < length; i++) {
                for (int j = 0; j < heights[i]; j++) {
                    ASSERT(valid_pptr(paths[i][j]));
                }
            }
        }
#endif

        papi_start_global_counters(datacounters);
        if (type == predecessor_only) {  // not insert_predecessor
            papi_start_global_counters(datacounters1);

            time_nested("data nodes", [&]() {
                time_start("init");
                auto task_starts =
                    parlay::delayed_seq<bool>(length, [&](size_t i) {
                        return i == 0 ||
                               not_equal_pptr(op_addrs[i], op_addrs[i - 1]);
                    });
                auto ll = parlay::make_slice(predecessor_core_ll,
                                             predecessor_core_ll + BATCH_SIZE);
                uint32_t llen = parlay::pack_index_into(task_starts, ll);
                ll = ll.cut(0, llen);
                time_end("init");

                auto io = alloc_io_manager();
                // ASSERT(io == io_managers[0]);
                io->init();
                auto L1_search_batch =
                    io->alloc<p_get_key_task, p_get_key_reply>(direct);

                time_nested("taskgen", [&]() {
                    auto batch = L1_search_batch;
                    // buffer filling method
                    // auto targets = [&](const p_get_key_task &x) {
                    //     return x.addr.id;
                    // };
                    // batch->push_task_from_array_by_isort<true>(
                    //     llen,
                    //     [&](size_t i) {
                    //         return (p_get_key_task){.addr = op_addrs[ll[i]]};
                    //     },
                    //     targets, make_slice(op_taskpos, op_taskpos + llen));

                    parfor_wrap(0, llen, [&](size_t i) {
                        p_get_key_task *pgkt =
                            (p_get_key_task *)batch->push_task_zero_copy(
                                op_addrs[ll[i]].id, -1, true, op_taskpos + i);
                        *pgkt = (p_get_key_task){.addr = op_addrs[ll[i]]};
                        ASSERT(valid_p_pptr(op_addrs[i]));
                    });
                    io->finish_task_batch();
                });
                papi_stop_global_counters(datacounters1);

                papi_start_global_counters(datacounters2);
                time_nested("exec", [&]() { io->exec(); });
                papi_stop_global_counters(datacounters2);

                papi_start_global_counters(datacounters3);
                auto off = parlay::delayed_seq<int>(
                    length_before_deduplication, [&](size_t i) {
                        return back_trace_offset[i];
                    });
                time_nested("get result", [&]() {
                    auto batch = L1_search_batch;
                    parfor_wrap(0, llen, [&](size_t i) {
                        int l = ll[i];
                        p_get_key_reply *pgkr =
                            (p_get_key_reply *)batch->get_reply(op_taskpos[i],
                                                                op_addrs[l].id);
                        int result_l = back_trace_offset_startpos[ll[i]];
                        int result_r =
                            (i == llen - 1)
                                ? length_before_deduplication
                                : back_trace_offset_startpos[ll[i + 1]];
                        parfor_wrap(result_l, result_r, [&](size_t x) {
                            kv_output[off[x]] = (key_value){
                                .key = pgkr->key, .value = pgkr->value};
                        });
                    });
                });
                io->reset();
            });
            papi_stop_global_counters(datacounters3);
        }
        papi_stop_global_counters(datacounters);
    }

    void get_load(slice<int64_t *, int64_t *> keys) {
        int n = keys.size();
        length = n;

        parlay::parallel_for(0, n, [&](size_t i) {
            keys_with_offset_sorted[i] = make_pair(i, keys[i]);
        });

        auto kwos_slice = parlay::make_slice(keys_with_offset_sorted,
                                             keys_with_offset_sorted + n);
        time_nested("sort", [&]() {
            parlay::sort_inplace(kwos_slice,
                                 [](const auto &t1, const auto &t2) {
                                     return t1.second < t2.second;
                                 });
        });
        parlay::parallel_for(0, n, [&](size_t i) {
            i64_input[i] = kwos_slice[i].second;
            back_trace_offset[i] = kwos_slice[i].first;
        });
    }

    void get() {
        int64_t *keys_sorted = i64_input;
        int n = length;

        auto task_starts = parlay::delayed_seq<bool>(n, [&](size_t i) {
            return i == 0 || keys_sorted[i] != keys_sorted[i - 1];
        });

        auto ll = parlay::pack_index(task_starts);
        int llen = ll.size();

        dpu_binary_switch_to(dpu_binary::query_binary);
        auto io = alloc_io_manager();
        // ASSERT(io == io_managers[0]);
        io->init();
        auto target = parlay::tabulate(llen, [&](int i) {
            return hash_to_dpu(keys_sorted[ll[i]], 0, nr_of_dpus);
        });
        auto get_batch = io->alloc<p_get_task, p_get_reply>(direct);
        time_nested("taskgen", [&]() {
            get_batch->push_task_from_array_by_isort<false>(
                llen,
                [&](size_t i) {
                    return (p_get_task){.key = keys_sorted[ll[i]]};
                },
                make_slice(target), make_slice(op_taskpos, op_taskpos + llen));
            // filling method
            // parfor_wrap(0, length, [&](size_t i) {
            //     int64_t key = keys[i];
            //     // int target = hash_to_dpu(key, 0, nr_of_dpus);
            //     p_get_task *pgt = (p_get_task *)batch->push_task_zero_copy(
            //         target_addr[i], -1, true, op_taskpos + i);
            //     *pgt = (p_get_task){.key = key};
            // });
            io->finish_task_batch();
        });

        time_nested("exec", [&]() { ASSERT(io->exec()); });

        time_nested("fill result", [&]() {
            auto idx = parlay::delayed_seq<int>(
                n, [&](size_t i) { return back_trace_offset[i]; });
            parlay::parallel_for(0, llen, [&](size_t i) {
                auto reply =
                    (p_get_reply *)get_batch->ith(target[i], op_taskpos[i]);
                int l = ll[i];
                int r = (i == llen - 1) ? n : ll[i + 1];
                parlay::parallel_for(l, r, [&](size_t x) {
                    kv_output[idx[x]] =
                        (key_value){.key = reply->key, .value = reply->value};
                });
            });
        });

        io->reset();

        // assert(false);
        // return result;
    }

    void update(slice<key_value *, key_value *> ops) {
        (void)ops;
        assert(false);
    }

    void predecessor_load(slice<int64_t *, int64_t *> keys) {
        int n = keys.size();
        length = n;
        parlay::parallel_for(0, n, [&](size_t i) {
            keys_with_offset_sorted[i] = make_pair(i, keys[i]);
        });

        auto kwos_slice = parlay::make_slice(keys_with_offset_sorted,
                                             keys_with_offset_sorted + n);
        time_nested("sort", [&]() {
            parlay::sort_inplace(kwos_slice,
                                 [](const auto &t1, const auto &t2) {
                                     return t1.second < t2.second;
                                 });
        });

        time_nested("deduplication", [&]() {
            length_before_deduplication = length;
            auto different = parlay::delayed_seq<bool>(n, [&](size_t i) {
                return (i == 0) || (keys_with_offset_sorted[i].second !=
                                    keys_with_offset_sorted[i - 1].second);
            });
            parlay::parallel_for(0, n, [&](size_t i) {
                back_trace_offset[i] = keys_with_offset_sorted[i].first;
            });
            auto btos_slice =
                parlay::make_slice(back_trace_offset_startpos,
                                   back_trace_offset_startpos + length);
            n = length = parlay::pack_index_into(parlay::make_slice(different),
                                                 btos_slice);
            parlay::parallel_for(0, n, [&](size_t i) {
                i64_input[i] = kwos_slice[btos_slice[i]].second;
            });
        });
    }

    void predecessor() {
        time_nested("core",
                    [&]() { predecessor_core(predecessor_only, NULL, NULL); });
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
        }
        ASSERT(node_id[L2_HEIGHT + 1] <=
               newnode_offset_buffer + BATCH_SIZE * 2);
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

    void insert_load(slice<key_value *, key_value *> kvs) {
        int n = kvs.size();
        length = n;
        parlay::sort_inplace(kvs, [](auto t1, auto t2) { return t1 < t2; });
        auto kv_input_slice = parlay::make_slice(kv_input, kv_input + length);
        n = length = parlay::pack_into(
            kvs,
            parlay::make_slice(parlay::delayed_seq<bool>(
                n,
                [&](size_t i) {
                    return (i == 0) || (kvs[i].key != kvs[i - 1].key);
                })),
            kv_input_slice);
    }

    void insert() {
        dpu_binary_switch_to(dpu_binary::query_binary);
        time_start("init");
        {
            auto kv_input_slice =
                parlay::make_slice(kv_input, kv_input + length);

            auto io = alloc_io_manager();
            io->init();
            auto target = parlay::tabulate(length, [&](size_t i) {
                return hash_to_dpu(kv_input[i].key, 0, nr_of_dpus);
            });
            auto update_batch =
                io->alloc<p_update_task, p_update_reply>(direct);
            time_nested("taskgen", [&]() {
                update_batch->push_task_from_array_by_isort<false>(
                    length,
                    [&](size_t i) {
                        return (p_update_task){{.key = kv_input[i].key,
                                                .value = kv_input[i].value}};
                    },
                    make_slice(target),
                    make_slice(op_taskpos, op_taskpos + length));
                io->finish_task_batch();
            });

            time_nested("exec", [&]() { ASSERT(io->exec()); });

            auto not_existed = parlay::tabulate(length, [&](size_t i) {
                auto reply = (p_update_reply *)update_batch->ith(target[i],
                                                                 op_taskpos[i]);
                return (reply->valid == 0);
            });

            io->reset();

            auto tmp = parlay::pack(kv_input_slice, not_existed);
            ASSERT(tmp.size() == length);
            length = tmp.size();
            parlay::parallel_for(0, length,
                                 [&](size_t i) { kv_input[i] = tmp[i]; });
        }
        int n = length;
        parlay::parallel_for(0, n,
                             [&](size_t i) { i64_input[i] = kv_input[i].key; });
        int64_t *keys = i64_input;
        auto values = parlay::delayed_seq<int64_t>(
            n, [&](size_t i) { return kv_input[i].value; });

        auto heights = parlay::sequence<int>::from_function(
            length,
            [&](size_t i) -> int {
                (void)&i;
                int t = rn_gen::parallel_rand();
                t = __builtin_ctz(t) + 1;
                if (t <= L2_HEIGHT * L2_SIZE_LOG) {
                    t = (t - 1) / L2_SIZE_LOG + 1;
                } else {
                    t = t - L2_HEIGHT * L2_SIZE_LOG + L2_HEIGHT;
                    t = min(t, L2_HEIGHT + max_l3_height);
                }
                return t;
            },
            (PARALLEL_ON) ? 0 : INT32_MAX);
            // 1111(chunking L1)
            // 2222, 3333(chunking L2)
            // 4, 5, 6(L3), ...

        auto predecessor_record = parlay::map(heights, [&](int32_t x) {
            return (x >= CACHE_HEIGHT) ? L2_HEIGHT : 1;
        });

        time_nested("horizontal reduce",
                    [&]() { horizontal_reduce(heights.data(), length); });

        auto predecessor_record_sum_pair = parlay::scan(predecessor_record);

        auto predecessor_record_total = predecessor_record_sum_pair.second;
        auto predecessor_record_prefix_sum = predecessor_record_sum_pair.first;

        auto insert_path_addrs =
            parlay::map(predecessor_record_prefix_sum,
                        [&](int32_t x) { return insert_path_addrs_buf + x; });

        ASSERT(predecessor_record_total < BATCH_SIZE * 1.2);

        time_end("init");

        printf("\n**** INSERT PREDECESSOR ****\n");
        time_nested("predecessor", [&]() {
            predecessor_core(predecessor_insert, predecessor_record.data(),
                             insert_path_addrs.data());
        });

        dpu_binary_switch_to(dpu_binary::insert_binary);

        auto keys_target = parlay::tabulate(length, [&](uint32_t i) {
            return hash_to_dpu(keys[i], 0, nr_of_dpus);
        });

        auto l2_root_target = parlay::sequence<int>(length, -1);
        for (int x = 0; x < newnode_count[L2_HEIGHT]; x++) {
            int i = node_id[L2_HEIGHT][x];
            int cnc = INT32_MAX;
            for (int KK = 0; KK < 4; KK++) {
                int t = abs(rn_gen::parallel_rand()) % nr_of_dpus;
                if (l2_root_count[t] < cnc) {
                    cnc = l2_root_count[t];
                    l2_root_target[i] = t;
                }
            }
            l2_root_count[l2_root_target[i]]++;
        }

        {
            int total_l2_root = 0;
            for (int i = 0; i < NR_DPUS; i++) {
                total_l2_root += l2_root_count[i];
            }
            printf("total l2 root: %d\n", total_l2_root);
        }

        printf("\n**** INSERT L123 ****\n");

        time_start("newnode L123 + truncate L2");

        auto l1_addrs = parlay::sequence(length, null_pptr);
        auto l2_addrs =
            parlay::map(predecessor_record_prefix_sum, [&](int32_t x) {
                return l2_addrs_buf + x;
            });  // may waste little memory
        auto l2_addrs_taskpos =
            parlay::map(predecessor_record_prefix_sum,
                        [&](int32_t x) { return l2_addrs_taskpos_buf + x; });

        auto io = alloc_io_manager();
        // ASSERT(io == io_managers[0]);
        io->init();

        auto L2_newnode_batch = io->alloc_task_batch(
            direct, fixed_length, fixed_length, B_NEWNODE_TSK,
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
                    *bnt = (b_newnode_task){{.height = ht}};
                }
            });
            io->finish_task_batch();
        });

        auto L1_newnode_batch = io->alloc_task_batch(
            direct, fixed_length, fixed_length, P_NEWNODE_TSK,
            sizeof(p_newnode_task), sizeof(p_newnode_reply));
        time_nested("L1 taskgen", [&]() {
            L1_newnode_batch->push_task_from_array_by_isort<false>(
                length,
                [&](size_t i) {
                    return (p_newnode_task){{.key = keys[i],
                                             .height = heights[i],
                                             .value = values[i]}};
                },
                make_slice(keys_target),
                make_slice(op_taskpos, op_taskpos + length));
            // another filling method
            // parfor_wrap(0, length, [&](size_t i) {
            //     p_newnode_task *pnt =
            //         (p_newnode_task *)L1_newnode_batch->push_task_zero_copy(
            //             keys_target[i], -1, true, op_taskpos + i);
            //     *pnt = (p_newnode_task){
            //         {.key = keys[i], .height = heights[i], .value =
            //         values[i]}};
            // });
            io->finish_task_batch();
        });

        auto insert_truncate_taskpos = parlay::map(
            predecessor_record_prefix_sum,
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

        IO_Task_Batch *L2_truncate_batch;
        IO_Task_Batch *cache_truncate_batch;

        int L2_truncate_batch_erl = sizeof(b_truncate_reply) + S64(L2_SIZE);
        L2_truncate_batch = io->alloc_task_batch(
            direct, fixed_length, variable_length, B_TRUNCATE_TSK,
            sizeof(b_truncate_task), L2_truncate_batch_erl);
        time_nested("L2 truncate", [&]() {
            auto new_truncate_task = [&](int i, int ht) {
                pptr addr = insert_path_addrs[i][ht];
                b_truncate_task *btt =
                    (b_truncate_task *)L2_truncate_batch->push_task_zero_copy(
                        addr.id, -1, true, &(insert_truncate_taskpos[i][ht]));
                *btt = (b_truncate_task){
                    {.addr = insert_path_addrs[i][ht], .key = keys[i]}};
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

#ifdef SHADOW_SUBTREE
        cache_truncate_batch = io->alloc_task_batch(
            direct, fixed_length, fixed_length, CACHE_TRUNCATE_TSK,
            sizeof(cache_truncate_task), 0);
        time_nested("cache truncate", [&]() {
            auto new_cache_truncate_task = [&](int i, int ht) {
                ASSERT(CACHE_HEIGHT == ht + 1);
                pptr addr = insert_path_addrs[i][ht + 1];
                cache_truncate_task *ctt =
                    (cache_truncate_task *)cache_truncate_batch
                        ->push_task_zero_copy(addr.id, -1, true);
                *ctt = (cache_truncate_task){
                    {.addr = addr, .key = keys[i], .height = ht}};
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
            io->finish_task_batch();
        });
#endif

        time_nested("exec", [&]() { ASSERT(io->exec()); });

        ASSERT(length > 0);
        time_nested("get result", [&]() {
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

        time_end("newnode L123 + truncate L2");

        printf("\n**** INSERT L123 ud ****\n");

        auto io2 = alloc_io_manager();
        ASSERT(io2 == io_managers[1]);
        io2->init();

        IO_Task_Batch *L3_insert_batch = nullptr;
        int l3_length = 0;
        time_nested("L3 taskgen", [&]() {
            l3_length = newnode_count[L2_HEIGHT];
            if (l3_length == 0) {
                return;
            }
            L3_insert_batch =
                io2->alloc<L3_insert_task, empty_task_reply>(broadcast);
            for (int t = 0; t < l3_length; t++) {
                int i = node_id[L2_HEIGHT][t];
                L3_insert_task *tit =
                    (L3_insert_task *)L3_insert_batch->push_task_zero_copy(
                        -1, -1, false);
                *tit = (L3_insert_task){{.key = keys[i],
                                         .height = heights[i] - L2_HEIGHT,
                                         .down = l2_addrs[i][L2_HEIGHT - 1]}};
            }
            io2->finish_task_batch();
        });

        time_nested("build ud ptrs", [&]() {
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
                                    for (r = t + 1; r < newnode_count[ht];
                                         r++) {
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
                                    int tsklen = r - t;
                                    b_insert_task *bit =
                                        (b_insert_task *)L2_key_insert_batch
                                            ->push_task_zero_copy(
                                                insert_addr.id,
                                                S64(2 + tsklen * 2), true);
                                    bit->addr = insert_addr;
                                    bit->len = tsklen;
                                    int64_t *tskkeys = bit->vals;
                                    pptr *tskaddrs =
                                        (pptr *)(bit->vals + tsklen);
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

            auto L2_newnode_init_batch = io2->alloc_task_batch(
                direct, variable_length, fixed_length, B_INSERT_TSK, -1, 0);
            {
                for (int ht = 0; ht < L2_HEIGHT; ht++) {
                    time_nested("nodefill taskgen", [&]() {
                        int nodeht = ht + 1;
                        parfor_wrap(0, newnode_count[nodeht], [&](size_t t) {
                            if (is_truncator(nodeht, t, ht)) {
                                int i = node_id[nodeht][t];
                                pptr pred_addr = insert_path_addrs[i][ht];
                                int r;
                                for (r = t + 1; r < newnode_count[nodeht];
                                     r++) {
                                    int rid = node_id[nodeht][r];
                                    if (not_equal_pptr(
                                            insert_path_addrs[i][ht],
                                            insert_path_addrs[rid][ht])) {
                                        break;
                                    }
                                }

                                // init reply
                                b_truncate_reply *rep =
                                    (b_truncate_reply *)
                                        L2_truncate_batch->get_reply(
                                            insert_truncate_taskpos[i][ht],
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
                                                S64(2 + (tsklen + 1) * 2),
                                                true);
                                    bit->addr = newnode_addr;
                                    bit->len = tsklen + 1;
                                    int64_t *tskkeys = bit->vals;
                                    pptr *tskaddrs =
                                        (pptr *)(bit->vals + tsklen + 1);
                                    tskkeys[0] = keys[rid];
                                    tskaddrs[0] = (ht == 0)
                                                      ? l1_addrs[rid]
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
                            if (is_truncator(nodeht, t, ht)) {
                                int i = node_id[nodeht][t];
                                pptr pred_addr = insert_path_addrs[i][ht];
                                int r;
                                for (r = t + 1; r < newnode_count[nodeht];
                                     r++) {
                                    int rid = node_id[nodeht][r];
                                    if (not_equal_pptr(
                                            insert_path_addrs[i][ht],
                                            insert_path_addrs[rid][ht])) {
                                        break;
                                    }
                                }

                                // init reply
                                b_truncate_reply *rep =
                                    (b_truncate_reply *)
                                        L2_truncate_batch->get_reply(
                                            insert_truncate_taskpos[i][ht],
                                            pred_addr.id);
                                pptr right = rep->right;
                                pptr left = pred_addr;

                                for (int j = t; j < r; j++) {
                                    int rid = node_id[nodeht][j];
                                    pptr newnode_addr = l2_addrs[rid][ht];
                                    if (j == (int)t) {
                                        b_set_lr_task *bslt =
                                            (b_set_lr_task *)L2_set_lr_batch
                                                ->push_task_zero_copy(
                                                    pred_addr.id, -1, true);
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
                                                : l2_addrs[node_id[nodeht]
                                                                  [j + 1]][ht];
                                    }
                                    left = newnode_addr;
                                    if (j == r - 1 &&
                                        not_equal_pptr(right, null_pptr)) {
                                        b_set_lr_task *bslt =
                                            (b_set_lr_task *)L2_set_lr_batch
                                                ->push_task_zero_copy(right.id,
                                                                      -1, true);
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

        time_nested("build udlr exec", [&]() { ASSERT(!io2->exec()); });

        io2->reset();
        io2 = nullptr;

#ifdef SHADOW_SUBTREE
        time_nested("build cache", [&]() { build_cache(); });

        printf("\n**** INSERT CACHE ****\n");

        ASSERT(L2_HEIGHT == 3);

        io = alloc_io_manager();
        // ASSERT(io == io_managers[0]);
        io->init();

        ASSERT(CACHE_HEIGHT == 2);
        auto cache_insert_batch_1to3 = io->alloc_task_batch(
            direct, fixed_length, fixed_length, CACHE_INSERT_TSK,
            sizeof(cache_insert_task), 0);
        {
            auto batch = cache_insert_batch_1to3;
            // new L2-1 to L2-3
            parfor_wrap(0, newnode_count[1], [&](size_t nid) {
                int i = node_id[1][nid];
                ASSERT(heights[i] >= 2);
                if (heights[i] > 3) {  // can be 2 / 3
                    return;
                }
                ASSERT(predecessor_record[i] == L2_HEIGHT);
                pptr source = insert_path_addrs[i][2];

                if (nid > 0) {
                    int preid = nid - 1;
                    int pre = node_id[1][preid];
                    ASSERT(heights[pre] >= 2);
                    if (not_equal_pptr(source, insert_path_addrs[pre][2])) {
                        // pass
                    } else {
                        return;  // not the left most
                    }
                }

                // V2
                int succid = nid;
                int preheight = heights[i];
                pptr pre_source_1 = insert_path_addrs[i][1];
                auto valid_succ = [&](int succid, int succ) {
                    if (heights[succ] == 3) {
                        preheight = heights[succ];
                        pre_source_1 = insert_path_addrs[succ][1];
                    } else if (preheight == 2 ||
                               not_equal_pptr(
                                   insert_path_addrs[succ][1],
                                   pre_source_1)) {  // heights[succ] == 2
                        return true;
                    }
                    return false;
                };
                int count = 0;
                for (succid = nid; succid < newnode_count[1]; succid++) {
                    int succ = node_id[1][succid];
                    ASSERT(heights[succ] >= 2);
                    if (heights[succ] > 3 ||
                        not_equal_pptr(source, insert_path_addrs[succ][2])) {
                        break;
                    }
                    if (valid_succ(succid, succ)) {
                        count++;
                    }
                }
                auto *cit =
                    (cache_insert_task *)batch->push_multiple_tasks_zero_copy(
                        source.id, -1, count, true, NULL);
                int count_chk = count;
                count = 0;
                preheight = heights[i];
                pre_source_1 = insert_path_addrs[i][1];
                for (succid = nid; succid < newnode_count[1]; succid++) {
                    int succ = node_id[1][succid];
                    ASSERT(heights[succ] >= 2);
                    if (heights[succ] > 3 ||
                        not_equal_pptr(source, insert_path_addrs[succ][2])) {
                        break;
                    }
                    if (valid_succ(succid, succ)) {
                        cit[count++] =
                            (cache_insert_task){{.addr = source,
                                                 .key = keys[succ],
                                                 .t_addr = l2_addrs[succ][0],
                                                 .height = 1}};
                    }
                }
                ASSERT(count == count_chk);
            });
            io->finish_task_batch();
        }

        time_nested("cache insert", [&]() { ASSERT(!io->exec()); });
        io->reset();
#endif

        return;
    }

    void remove_load(slice<int64_t *, int64_t *> _keys) {
        length = _keys.size();
        int n = length;
        parlay::sort_inplace(_keys);
        auto i64_input_slice = make_slice(i64_input, i64_input + length);
        length = parlay::pack_into(
            _keys,
            parlay::delayed_seq<bool>(n,
                                      [&](size_t i) {
                                          return (i == 0) ||
                                                 (_keys[i] != _keys[i - 1]);
                                      }),
            i64_input_slice);
    }

    void remove() {
#ifndef SHADOW_SUBTREE
        throw "shadow subtree not removed from DELETE.";
#endif

        auto keys_slice = parlay::make_slice(i64_input, i64_input + length);

        cout << "********************************" << length
             << " after dedup ********************************" << endl;

        // auto heights = parlay::sequence<int>(length);
        auto heights = parlay::sequence<int>::uninitialized(length);

        time_nested("height", [&]() {
            // get height, also remove from hash tables
            dpu_binary_switch_to(dpu_binary::query_binary);

            auto io = alloc_io_manager();
            // ASSERT(io == io_managers[0]);
            io->init();

            auto keys_target = parlay::tabulate(length, [&](uint32_t i) {
                return hash_to_dpu(keys_slice[i], 0, nr_of_dpus);
            });

            auto get_height_batch = io->alloc_task_batch(
                direct, fixed_length, fixed_length, P_GET_HEIGHT_TSK,
                sizeof(p_get_height_task), sizeof(p_get_height_reply));
            time_nested("taskgen", [&]() {
                auto batch = get_height_batch;
                batch->push_task_from_array_by_isort<false>(
                    length,
                    [&](size_t i) {
                        return (p_get_height_task){.key = keys_slice[i]};
                    },
                    make_slice(keys_target),
                    make_slice(op_taskpos, op_taskpos + length));
                // another filling method
                // parfor_wrap(0, length, [&](size_t i) {
                //     int64_t key = keys[i];
                //     int target = keys_target[i];
                //     p_get_height_task *pght =
                //         (p_get_height_task *)batch->push_task_zero_copy(
                //             target, -1, true, op_taskpos + i);
                //     *pght = (p_get_height_task){.key = key};
                // });
                io->finish_task_batch();
            });

            time_nested("exec", [&]() { io->exec(); });

            time_nested("get result", [&]() {
                auto batch = get_height_batch;
                parfor_wrap(0, length, [&](size_t i) {
                    int64_t key = keys_slice[i];
                    int target = keys_target[i];
                    p_get_height_reply *pghr =
                        (p_get_height_reply *)batch->get_reply(op_taskpos[i],
                                                               target);
                    heights[i] = pghr->height;
                });
            });
            io->reset();
        });

        auto valid_remove = parlay::tabulate(
            length, [&](int i) -> bool { return heights[i] >= 0; });

        {
            auto tmp = parlay::pack(keys_slice, valid_remove);
            heights = parlay::pack(heights, valid_remove);
            ASSERT(tmp.size() == heights.size());
            length = tmp.size();
            parlay::parallel_for(0, length,
                                 [&](size_t i) { i64_input[i] = tmp[i]; });
        }
        auto keys = parlay::make_slice(i64_input, i64_input + length);

        cout << "********************************" << length
             << " to remove ********************************" << endl;
        ASSERT(keys.size() == heights.size());

#ifdef REMOVE_DEBUG
        for (int i = 0; i < length; i++) {
            printf("k[%d]=%llx\th[%d]=%d\n", i, keys[i], heights[i]);
        }
#endif

        time_nested("horizontal reduce",
                    [&]() { horizontal_reduce(heights.data(), length); });

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
            predecessor_core(predecessor_insert, predecessor_record.data(),
                             remove_path_addrs.data());
        });

#ifdef REMOVE_DEBUG
        for (int i = 0; i < length; i++) {
            int ht = min(L2_HEIGHT, predecessor_record[i]);
            for (int j = 0; j < ht; j++) {
                printf("remove_path_addrs[%d][%d]=%lx\n", i, j,
                       pptr_to_int64(remove_path_addrs[i][j]));
            }
        }
#endif

        dpu_binary_switch_to(dpu_binary::delete_binary);

        auto remove_truncate_taskpos = parlay::map(
            predecessor_record_prefix_sum,
            [&](int32_t x) { return insert_truncate_taskpos_buf + x; });

        time_nested("core", [&]() {
            auto io = alloc_io_manager();
            // ASSERT(io == io_managers[0]);
            io->init();

            auto L2_remove_batch = io->alloc_task_batch(
                direct, variable_length, fixed_length, B_REMOVE_TSK, -1, 0);
            time_nested("L2 remove", [&]() {
                for (int ht = 0; ht < L2_HEIGHT; ht++) {
                    int nodelen = newnode_count[ht];
                    auto task_starts =
                        parlay::delayed_seq<bool>(nodelen, [&](int t) -> bool {
                            if (t == 0) {
                                return true;
                            }
                            int i = node_id[ht][t];
                            int l = node_id[ht][t - 1];
                            ASSERT(l < i);
                            return not_equal_pptr(remove_path_addrs[i][ht],
                                                  remove_path_addrs[l][ht]);
                        });
                    auto l = parlay::pack_index(task_starts);
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
                    int sht =
                        ht + 1;  // (height-1) of the node to cause merging
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
            io2->init();

            auto io3 = alloc_io_manager();
            ASSERT(io3 == io_managers[2]);
            io3->init();

            IO_Task_Batch *L3_remove_batch = nullptr;
            time_nested("L3 remove", [&]() {
                int ht = L2_HEIGHT;
                int nodelen = newnode_count[ht];
                if (nodelen == 0) {
                    return;
                }
                L3_remove_batch = io2->alloc_task_batch(
                    broadcast, fixed_length, fixed_length, L3_REMOVE_TSK,
                    sizeof(L3_remove_task), 0);
                for (int t = 0; t < nodelen; t++) {
                    int i = node_id[ht][t];
                    L3_remove_task *trt =
                        (L3_remove_task *)L3_remove_batch->push_task_zero_copy(
                            -1, -1, false);
                    trt->key = keys[i];
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
                    int sht =
                        ht + 1;  // (height-1) of the node to cause merging
                    auto task_start =
                        parlay::sequence<bool>(newnode_count[sht], false);

                    auto siz = parlay::sequence<int>(newnode_count[sht], 0);
                    auto taskpos =
                        parlay::sequence<int>(newnode_count[sht], -1);

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

                    auto l = parlay::pack_index(task_start);
                    int llen = l.size();
                    if (llen == 0) continue;
                    auto taskptr =
                        parlay::sequence<b_insert_task *>(llen, nullptr);
                    parfor_wrap(0, llen, [&](size_t t) {
                        int ll = l[t];
                        int rr =
                            (t == llen - 1) ? newnode_count[sht] : l[t + 1];
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
                                        remove_truncate_taskpos[ri][ht],
                                        addrr.id);
                            brgnr->right = brgnrr->right;
#ifdef REMOVE_DEBUG
                            printf(
                                "L2 lr after: "
                                "key=%llx\tleft=%llx\tright=%llx\n",
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
                        int rr =
                            (t == llen - 1) ? newnode_count[sht] : l[t + 1];
                        int i = node_id[sht][ll];
                        b_insert_task *bit = taskptr[t];
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
                        printf("L2 cache insert: addr=%llx len=%lld\n",
                               cmit->addr, cmit->len);
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

            // left right pointers seem to be unused
            // could possibly be removed causing no error.
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
                            (b_set_lr_task *)L2_set_lr_batch
                                ->push_task_zero_copy(left.id, -1, true);
                        *bslt = (b_set_lr_task){
                            {.addr = left, .left = null_pptr, .right = right}};

                        if (not_equal_pptr(brgnr->right, null_pptr)) {
                            bslt = (b_set_lr_task *)
                                       L2_set_lr_batch->push_task_zero_copy(
                                           right.id, -1, true);
                            *bslt = (b_set_lr_task){{.addr = right,
                                                     .left = left,
                                                     .right = null_pptr}};
                        }
                    });
                }
                io2->finish_task_batch();
            });

            time_nested("L2 remove insert exec",
                        [&]() { ASSERT(!io2->exec()); });

            io->reset();
            io2->reset();

            ASSERT(CACHE_HEIGHT == 2);
            time_nested("build cache", [&]() { build_cache(); });

            // for a deletion with height 3, this will try to remove its h1 node
            // from the cache of its h3 predecessor, which doesn't exist. but
            // this shouldn't cause any problem
            time_nested("L2 cache remove", [&]() {
                auto cache_remove_batch = io3->alloc_task_batch(
                    direct, fixed_length, fixed_length, CACHE_REMOVE_TSK,
                    sizeof(cache_remove_task), 0);
                parfor_wrap(0, newnode_count[1], [&](size_t nid) {
                    int i = node_id[1][nid];
                    pptr target = remove_path_addrs[i][2];
                    if (nid > 0) {
                        int preid = nid - 1;
                        int pre = node_id[1][preid];
                        if (equal_pptr(target, remove_path_addrs[pre][2])) {
                            return;  // not start
                        }
                    }
                    int succid = nid;
                    for (succid = nid; succid < newnode_count[1]; succid++) {
                        int succ = node_id[1][succid];
                        if (!equal_pptr(target, remove_path_addrs[succ][2])) {
                            break;
                        }
                    }
                    auto *crt =
                        (cache_remove_task *)
                            cache_remove_batch->push_multiple_tasks_zero_copy(
                                target.id, -1, succid - nid, true);
                    for (succid = nid; succid < newnode_count[1]; succid++) {
                        int succ = node_id[1][succid];
                        if (!equal_pptr(target, remove_path_addrs[succ][2])) {
                            break;
                        }
                        crt[succid - nid] = (cache_remove_task){
                            {.addr = target, .key = keys[succ], .height = 1}};
                    }
                });
                io3->finish_task_batch();
            });

            time_nested("cache update", [&]() { ASSERT(!io3->exec()); });
            io3->reset();
        });
    }

    /********************** Range Scan ***********************************/
    template <class T>
    class range_end_more {
       public:
        T identity;
        range_end_more() {
            identity.lkey = INT64_MIN;
            identity.rkey = INT64_MIN;
        }
        static T f(T r, T R) {
            T res;
            res.lkey = R.lkey;
            res.rkey = max(r.rkey, R.rkey);
            return res;
        }
    };

    template <class T>
    class pull_count {
       public:
        T identity;
        pull_count() { identity.second = -1; }
        static T f(T a, T b) {
            T res = b;
            if (b.first.second.id == INVALID_DPU_ID)
                res.second = a.second;
            else
                res.second = a.second + 1;
            return res;
        }
    };

    template <class T = scan_operation>
    inline T make_scan_op(int64_t lkey, int64_t rkey) {
        T res;
        res.lkey = min(lkey, rkey);
        res.rkey = max(lkey, rkey);
        return res;
    }

    auto l2_scan_one_round(parlay::sequence<pptr> &addrs_in,
                           parlay::sequence<scan_operation> &ops_merged, int ht,
                           int limit = push_pull_limit) {
        IO_Task_Batch *L2_scan_search_batch;
        IO_Task_Batch *L2_scan_fetch_batch;
        IO_Task_Batch *L2_scan_pull_batch;
        int range_num = ops_merged.size();
        int prev_node_num = addrs_in.size();
        int res_node_num = (range_num << 1);
        parlay::sequence<pptr> addrs_out;

        auto task_start = parlay::tabulate(range_num, [&](int i) -> bool {
            return (i == 0) ||
                   not_equal_pptr(addrs_in[(i << 1) - 2], addrs_in[i << 1]) ||
                   not_equal_pptr(addrs_in[(i << 1) - 1], addrs_in[i << 1]) ||
                   not_equal_pptr(addrs_in[i << 1], addrs_in[(i << 1) + 1]);
        });
        auto lll = parlay::pack_index(task_start);
        int llen = lll.size();
        auto rr = parlay::tabulate(llen, [&](int i) -> size_t {
            return ((i == llen - 1) ? range_num : lll[i + 1]);
        });

        auto this_ll = parlay::sequence<int>(range_num, 0);
        parfor_wrap(0, llen, [&](size_t i) {
            if (rr[i] - lll[i] >= limit)
                this_ll[lll[i]] = 2;
            else {
                for (size_t j = lll[i]; j < rr[i]; j++) this_ll[j] = 1;
            }
        });

        auto pull_tab = parlay::tabulate(
            range_num, [&](int i) -> bool { return (this_ll[i] == 2); });
        auto pull_ll = parlay::pack_index(pull_tab);
        int pull_llen = pull_ll.size();

        auto push_tab = parlay::tabulate(
            range_num, [&](int i) -> bool { return (this_ll[i] == 1); });
        auto push_ll = parlay::pack_index(push_tab);
        int push_llen = push_ll.size();

        auto pull_rr_tab = parlay::tabulate(range_num, [&](int i) -> bool {
            return ((this_ll[i] == 0) &&
                    ((i == range_num - 1) || (this_ll[i + 1] != 0))) ||
                   ((this_ll[i] == 2) &&
                    ((i == range_num - 1) || (this_ll[i + 1] != 0)));
        });
        auto pull_rr = parlay::pack_index(pull_rr_tab);
        int pull_rren = pull_rr.size();
        ASSERT(pull_llen == pull_rren);
        auto scan_taskpos = parlay::sequence<int32_t>(prev_node_num);

        auto io = alloc_io_manager();
        io->init();

        time_nested("task gen", [&]() {
            int expected_get_node_reply = S64(L2_SIZE + 1);

            time_nested("L2 search", [&]() {
                time_nested("push", [&]() {
                    if (push_llen > 0) {
                        L2_scan_search_batch = io->alloc_task_batch(
                            direct, fixed_length, variable_length,
                            B_SCAN_SEARCH_TSK, sizeof(b_scan_search_task),
                            expected_get_node_reply);
                        parfor_wrap(0, push_llen, [&](int j) {
                            int i = push_ll[j];
                            pptr addrs1 = addrs_in[i << 1],
                                 addrs2 = addrs_in[(i << 1) + 1];
                            if (equal_pptr(addrs1, addrs2) &&
                                not_equal_pptr(addrs1, null_pptr)) {
                                // b_scan_search_task *bsst =
                                //     (b_scan_search_task
                                //     *)L2_scan_search_batch
                                //         ->push_task_zero_copy(
                                //             addrs1.id, -1, true,
                                //             op_taskpos + (i << 1));
                                b_scan_search_task *bsst =
                                    (b_scan_search_task *)L2_scan_search_batch
                                        ->push_task_zero_copy(
                                            addrs1.id, -1, true,
                                            &(scan_taskpos[i << 1]));
                                *bsst = (b_scan_search_task){
                                    {.addr = addrs1,
                                     .lkey = ops_merged[i].lkey,
                                     .rkey = ops_merged[i].rkey}};
                            } else {
                                b_scan_search_task *bsst;
                                if (not_equal_pptr(addrs1, null_pptr)) {
                                    // bsst = (b_scan_search_task *)
                                    //            L2_scan_search_batch
                                    //                ->push_task_zero_copy(
                                    //                    addrs1.id, -1, true,
                                    //                    op_taskpos + (i <<
                                    //                    1));
                                    bsst = (b_scan_search_task *)
                                               L2_scan_search_batch
                                                   ->push_task_zero_copy(
                                                       addrs1.id, -1, true,
                                                       &(scan_taskpos[i << 1]));
                                    *bsst = (b_scan_search_task){{
                                        .addr = addrs1,
                                        .lkey = ops_merged[i].lkey,
                                        .rkey = ops_merged[i].rkey,
                                    }};
                                }
                                if (not_equal_pptr(addrs2, null_pptr)) {
                                    // bsst =
                                    //     (b_scan_search_task *)
                                    //         L2_scan_search_batch
                                    //             ->push_task_zero_copy(
                                    //                 addrs2.id, -1, true,
                                    //                 op_taskpos + (i << 1) +
                                    //                 1);
                                    bsst =
                                        (b_scan_search_task
                                             *)L2_scan_search_batch
                                            ->push_task_zero_copy(
                                                addrs2.id, -1, true,
                                                &(scan_taskpos[(i << 1) + 1]));
                                    *bsst = (b_scan_search_task){{
                                        .addr = addrs2,
                                        .lkey = ops_merged[i].lkey,
                                        .rkey = ops_merged[i].rkey,
                                    }};
                                }
                            }
                        });
                        io->finish_task_batch();
                    }
                });

                time_nested("pull", [&]() {
                    if (pull_llen > 0) {
                        L2_scan_pull_batch = io->alloc_task_batch(
                            direct, fixed_length, variable_length,
                            B_GET_NODE_TSK, sizeof(b_get_node_task),
                            expected_get_node_reply);
                        parfor_wrap(0, pull_llen, [&](int j) {
                            int i = pull_ll[j];
                            pptr addr_pptr = addrs_in[i << 1];
                            if (not_equal_pptr(addr_pptr, null_pptr)) {
                                // b_get_node_task *bgnt =
                                //     (b_get_node_task *)
                                //         L2_scan_pull_batch->push_task_zero_copy(
                                //             addr_pptr.id, -1, true,
                                //             op_taskpos + (i << 1));
                                b_get_node_task *bgnt =
                                    (b_get_node_task *)
                                        L2_scan_pull_batch->push_task_zero_copy(
                                            addr_pptr.id, -1, true,
                                            &(scan_taskpos[i << 1]));
                                *bgnt = (b_get_node_task){.addr = addr_pptr};
                            }
                        });
                        io->finish_task_batch();
                    }
                });
            });

            time_nested("L2 fetch", [&]() {
                L2_scan_fetch_batch = io->alloc_task_batch(
                    direct, fixed_length, variable_length, B_FETCH_CHILD_TSK,
                    sizeof(b_fetch_child_task), expected_get_node_reply);
                parfor_wrap(range_num + range_num, prev_node_num, [&](int i) {
                    pptr addr_pptr = addrs_in[i];
                    if (not_equal_pptr(addr_pptr, null_pptr)) {
                        // b_fetch_child_task *bfct =
                        //     (b_fetch_child_task *)
                        //         L2_scan_fetch_batch->push_task_zero_copy(
                        //             addr_pptr.id, -1, true, op_taskpos + i);
                        b_fetch_child_task *bfct =
                            (b_fetch_child_task *)
                                L2_scan_fetch_batch->push_task_zero_copy(
                                    addr_pptr.id, -1, true, &(scan_taskpos[i]));
                        *bfct = (b_fetch_child_task){.addr = addr_pptr};
                    }
                });
                io->finish_task_batch();
            });
        });

        time_nested("exec", [&]() { io->exec(); });

        time_nested("get result", [&]() {
            // auto op_result_buffer = parlay::sequence<int>(prev_node_num);
            auto op_result_buffer =
                parlay::tabulate(prev_node_num, [&](int i) { return (int)0; });
            // Counting result nodes number
            time_nested("push count", [&]() {
                if (push_llen > 0) {
                    parfor_wrap(0, push_llen, [&](int j) {
                        int i = push_ll[j];
                        if (equal_pptr(addrs_in[i << 1],
                                       addrs_in[(i << 1) + 1]) &&
                            not_equal_pptr(addrs_in[i << 1], null_pptr)) {
                            // b_scan_search_reply *bssr =
                            //     (b_scan_search_reply *)L2_scan_search_batch
                            //         ->get_reply(op_taskpos[i << 1],
                            //                     addrs_in[i << 1].id);
                            b_scan_search_reply *bssr =
                                (b_scan_search_reply *)L2_scan_search_batch
                                    ->get_reply(scan_taskpos[i << 1],
                                                addrs_in[i << 1].id);
                            op_result_buffer[i << 1] =
                                max((int64_t)0, bssr->len - 2);
                            op_result_buffer[(i << 1) + 1] = 0;
                        } else {
                            b_scan_search_reply *bssr;
                            if (not_equal_pptr(addrs_in[i << 1], null_pptr)) {
                                // bssr = (b_scan_search_reply *)
                                //            L2_scan_search_batch->get_reply(
                                //                op_taskpos[i << 1],
                                //                addrs_in[i << 1].id);
                                bssr = (b_scan_search_reply *)
                                           L2_scan_search_batch->get_reply(
                                               scan_taskpos[i << 1],
                                               addrs_in[i << 1].id);
                                op_result_buffer[i << 1] =
                                    max((int64_t)0, bssr->len - 1);
                            }
                            if (not_equal_pptr(addrs_in[(i << 1) + 1],
                                               null_pptr)) {
                                // bssr = (b_scan_search_reply *)
                                //            L2_scan_search_batch->get_reply(
                                //                op_taskpos[(i << 1) + 1],
                                //                addrs_in[(i << 1) + 1].id);
                                bssr = (b_scan_search_reply *)
                                           L2_scan_search_batch->get_reply(
                                               scan_taskpos[(i << 1) + 1],
                                               addrs_in[(i << 1) + 1].id);
                                op_result_buffer[(i << 1) + 1] =
                                    max((int64_t)0, bssr->len - 1);
                            }
                        }
                    });
                }
            });

            time_nested("pull count", [&]() {
                if (pull_llen > 0) {
                    for (int j = 0; j < pull_llen; j++) {
                        int ll = (pull_ll[j] << 1);
                        int rr = ((pull_rr[j] + 1) << 1);
                        if (not_equal_pptr(addrs_in[ll], null_pptr)) {
                            // b_get_node_reply *bgnr =
                            //     (b_get_node_reply *)
                            //         L2_scan_pull_batch->get_reply(
                            //             op_taskpos[ll], addrs_in[ll].id);
                            b_get_node_reply *bgnr =
                                (b_get_node_reply *)
                                    L2_scan_pull_batch->get_reply(
                                        scan_taskpos[ll], addrs_in[ll].id);
                            int nnlen = bgnr->len;
                            int64_t *rep_keys = bgnr->vals;
                            pptr *rep_addrs = (pptr *)(rep_keys + nnlen);
                            auto rep_sort = parlay::sequence<
                                pair<pair<int64_t, pptr>, int>>(rr - ll +
                                                                nnlen);
                            parfor_wrap(0, nnlen, [&](int i) {
                                rep_sort[i].first.first = rep_keys[i];
                                rep_sort[i].first.second = rep_addrs[i];
                            });
                            parfor_wrap(ll, rr, [&](int i) {
                                if (((i >> 1) << 1) == i)
                                    rep_sort[i - ll + nnlen].first.first =
                                        ops_merged[i >> 1].lkey;
                                else
                                    rep_sort[i - ll + nnlen].first.first =
                                        ops_merged[i >> 1].rkey;
                                rep_sort[i - ll + nnlen].first.second =
                                    null_pptr;
                            });
                            parlay::sort_inplace(
                                rep_sort,
                                [&](pair<pair<int64_t, pptr>, int> p1,
                                    pair<pair<int64_t, pptr>, int> p2) -> bool {
                                    return (p1.first.first < p2.first.first) ||
                                           ((p1.first.first ==
                                             p2.first.first) &&
                                            (p1.first.second.id !=
                                             INVALID_DPU_ID));
                                });
                            parlay::scan_inclusive_inplace(
                                rep_sort,
                                pull_count<pair<pair<int64_t, pptr>, int>>());

                            // Rewrite sorted (key,addr) into Reply array
                            auto rep_tmp = parlay::tabulate(
                                rr - ll + nnlen, [&](size_t i) -> bool {
                                    return rep_sort[i].first.second.id !=
                                           INVALID_DPU_ID;
                                });
                            auto rep_sorted = parlay::pack_index(rep_tmp);
                            parfor_wrap(0, rep_sorted.size(), [&](size_t i) {
                                rep_keys[i] =
                                    rep_sort[rep_sorted[i]].first.first;
                                rep_addrs[i] =
                                    rep_sort[rep_sorted[i]].first.second;
                            });

                            // Mark the Predecessor & Number of Range Boundaries
                            parfor_wrap(0, rr - ll + nnlen, [&](size_t i) {
                                rep_tmp[i] = !rep_tmp[i];
                            });
                            rep_sorted = parlay::pack_index(rep_tmp);
                            parfor_wrap(
                                0, (rep_sorted.size() >> 1), [&](size_t i) {
                                    size_t ii = (i << 1);
                                    size_t k = ll + ii;
                                    // Predecessor of the key in Reply arrays
                                    op_result_buffer[k] =
                                        rep_sort[rep_sorted[ii]].second;
                                    // Number of keys returned
                                    int num = max(
                                        0, rep_sort[rep_sorted[ii + 1]].second -
                                               (int)op_result_buffer[k] - 1);
                                    op_result_buffer[k + 1] =
                                        num - op_result_buffer[k];
                                });
                        }
                    }
                }
            });

            time_nested("fetch count", [&]() {
                parfor_wrap(range_num << 1, prev_node_num, [&](int i) {
                    if (not_equal_pptr(addrs_in[i], null_pptr)) {
                        // b_fetch_child_reply *bfcr =
                        //     (b_fetch_child_reply *)L2_scan_fetch_batch
                        //         ->get_reply(op_taskpos[i], addrs_in[i].id);
                        b_fetch_child_reply *bfcr =
                            (b_fetch_child_reply *)L2_scan_fetch_batch
                                ->get_reply(scan_taskpos[i], addrs_in[i].id);
                        op_result_buffer[i] = bfcr->len;
                    }
                });
            });

            auto res_node_lens = parlay::make_slice(op_result_buffer);
            parlay::scan_inclusive_inplace(res_node_lens);
            res_node_num += res_node_lens[prev_node_num - 1];

            addrs_out = parlay::sequence<pptr>(res_node_num);
            parfor_wrap(0, res_node_num,
                        [&](int i) { addrs_out[i] = null_pptr; });
            // Getting results
            time_nested("push get", [&]() {
                if (push_llen > 0) {
                    parfor_wrap(0, push_llen, [&](int j) {
                        int i = push_ll[j];
                        if (equal_pptr(addrs_in[i << 1],
                                       addrs_in[(i << 1) + 1]) &&
                            not_equal_pptr(addrs_in[i << 1], null_pptr)) {
                            // b_scan_search_reply *bssr =
                            //     (b_scan_search_reply *)L2_scan_search_batch
                            //         ->get_reply(op_taskpos[i << 1],
                            //                     addrs_in[i << 1].id);
                            b_scan_search_reply *bssr =
                                (b_scan_search_reply *)L2_scan_search_batch
                                    ->get_reply(scan_taskpos[i << 1],
                                                addrs_in[i << 1].id);
                            int len = bssr->len;
                            addrs_out[i << 1] = bssr->addr[0];
                            if (len == 1)
                                addrs_out[(i << 1) + 1] = addrs_out[i << 1];
                            else if (len > 1)
                                addrs_out[(i << 1) + 1] = bssr->addr[len - 1];
                            int ss =
                                (i == 0 ? 0 : res_node_lens[(i << 1) - 1]) +
                                (range_num << 1);
                            for (int k = 0; k < len - 2; k++)
                                addrs_out[ss + k] = bssr->addr[k + 1];
                        } else {
                            b_scan_search_reply *bssr;
                            int len, ss;
                            if (not_equal_pptr(addrs_in[i << 1], null_pptr)) {
                                // bssr = (b_scan_search_reply *)
                                //            L2_scan_search_batch->get_reply(
                                //                op_taskpos[i << 1],
                                //                addrs_in[i << 1].id);
                                bssr = (b_scan_search_reply *)
                                           L2_scan_search_batch->get_reply(
                                               scan_taskpos[i << 1],
                                               addrs_in[i << 1].id);
                                len = bssr->len;
                                addrs_out[i << 1] = bssr->addr[0];
                                ss =
                                    (i == 0 ? 0 : res_node_lens[(i << 1) - 1]) +
                                    (range_num << 1);
                                for (int k = 0; k < len - 1; k++)
                                    addrs_out[ss + k] = bssr->addr[k + 1];
                            }
                            if (not_equal_pptr(addrs_in[(i << 1) + 1],
                                               null_pptr)) {
                                // bssr = (b_scan_search_reply *)
                                //            L2_scan_search_batch->get_reply(
                                //                op_taskpos[(i << 1) + 1],
                                //                addrs_in[(i << 1) + 1].id);
                                bssr = (b_scan_search_reply *)
                                           L2_scan_search_batch->get_reply(
                                               scan_taskpos[(i << 1) + 1],
                                               addrs_in[(i << 1) + 1].id);
                                len = bssr->len;
                                addrs_out[(i << 1) + 1] = bssr->addr[len - 1];
                                ss = res_node_lens[i << 1] + range_num +
                                     range_num;
                                for (int k = 0; k < len - 1; k++)
                                    addrs_out[ss + k] = bssr->addr[k];
                            }
                        }
                    });
                }
            });

            time_nested("pull get", [&]() {
                if (pull_llen > 0) {
                    parfor_wrap(0, pull_llen, [&](int j) {
                        int ll = pull_ll[j];
                        int rr = pull_rr[j] + 1;
                        if (not_equal_pptr(addrs_in[ll << 1], null_pptr)) {
                            // b_get_node_reply *bgnr =
                            //     (b_get_node_reply *)L2_scan_pull_batch
                            //         ->get_reply(op_taskpos[ll << 1],
                            //                     addrs_in[ll << 1].id);
                            b_get_node_reply *bgnr =
                                (b_get_node_reply *)L2_scan_pull_batch
                                    ->get_reply(scan_taskpos[ll << 1],
                                                addrs_in[ll << 1].id);
                            int nnlen = bgnr->len;
                            int64_t *rep_keys = bgnr->vals;
                            pptr *rep_addrs = (pptr *)(rep_keys + nnlen);

                            for (int i = ll; i < rr; i++) {
                                int ss =
                                    (i == 0 ? 0 : res_node_lens[(i << 1) - 1]);
                                int begin_in_reply = res_node_lens[i << 1] - ss;
                                int num_nodes_fetch =
                                    res_node_lens[(i << 1) + 1] - ss;
                                ss += (range_num << 1);
                                addrs_out[i << 1] = rep_addrs[begin_in_reply];
                                if (num_nodes_fetch > 0) {
                                    addrs_out[(i << 1) + 1] =
                                        rep_addrs[begin_in_reply +
                                                  num_nodes_fetch + 1];
                                    for (int k = 0; k < num_nodes_fetch; k++)
                                        addrs_out[ss + k] =
                                            rep_addrs[begin_in_reply + k + 1];
                                } else {
                                    if ((begin_in_reply == nnlen - 1) ||
                                        (rep_keys[begin_in_reply + 1] >
                                         ops_merged[i].rkey))
                                        addrs_out[(i << 1) + 1] =
                                            rep_addrs[begin_in_reply];
                                    else
                                        addrs_out[(i << 1) + 1] =
                                            rep_addrs[begin_in_reply + 1];
                                }
                            }
                        }
                    });
                }
            });

            time_nested("fetch get", [&]() {
                parfor_wrap(range_num << 1, prev_node_num, [&](int i) {
                    if (not_equal_pptr(addrs_in[i], null_pptr)) {
                        // b_fetch_child_reply *bfcr =
                        //     (b_fetch_child_reply *)L2_scan_fetch_batch
                        //         ->get_reply(op_taskpos[i], addrs_in[i].id);
                        b_fetch_child_reply *bfcr =
                            (b_fetch_child_reply *)L2_scan_fetch_batch
                                ->get_reply(scan_taskpos[i], addrs_in[i].id);
                        int len = bfcr->len;
                        int ss = op_result_buffer[i - 1] + (range_num << 1);
                        for (int k = 0; k < len; k++)
                            addrs_out[ss + k] = bfcr->addr[k];
                    }
                });
            });
        });
        io->reset();
        return addrs_out;
    }

    auto scan(slice<scan_operation *, scan_operation *> ops) {
        printf("\n********** Scan Test **********\n");
        int length = ops.size();
        int range_num = 0;
        int res_node_num = 0;
        int p_node_num = 0;
        int final_key_num = 0;

        parlay::sequence<scan_operation> ops_merged;
        time_nested("Merge Ranges", [&]() {
            auto seq = parlay::sort(
                ops, [&](scan_operation s1, scan_operation s2) -> bool {
                    return (s1.lkey < s2.lkey) ||
                           ((s1.lkey == s2.lkey) && (s1.rkey < s2.rkey));
                });
            auto seq_max_prefix_scan =
                parlay::scan(seq, range_end_more<scan_operation>());
            auto scan_start_arr = parlay::tabulate(length, [&](int i) {
                return (seq_max_prefix_scan.first[i].rkey < seq[i].lkey) ||
                       (i == 0);
            });
            auto scan_start = parlay::pack_index(scan_start_arr);
            range_num = scan_start.size();
            ops_merged = parlay::tabulate(range_num, [&](int i) {
                return (scan_operation){
                    .lkey = seq[scan_start[i]].lkey,
                    .rkey = ((i != range_num - 1)
                                 ? (seq_max_prefix_scan.first[scan_start[i + 1]]
                                        .rkey)
                                 : seq_max_prefix_scan.second.rkey)};
            });
            auto block_nums = parlay::sequence<int64_t>(range_num);
            uint64_t avg_len = ops[0].rkey - ops[0].lkey;
            parfor_wrap(0, range_num, [&](size_t i) {
                uint64_t merge_len;
                if (ops_merged[i].rkey >= 0 && ops_merged[i].lkey >= 0)
                    merge_len = ops_merged[i].rkey - ops_merged[i].lkey;
                else if (ops_merged[i].rkey < 0 && ops_merged[i].lkey < 0)
                    merge_len = ops_merged[i].rkey - ops_merged[i].lkey;
                else
                    merge_len = (uint64_t)(ops_merged[i].rkey) +
                                (uint64_t)(-(ops_merged[i].lkey));
                if (merge_len >= (uint64_t)5 * avg_len)
                    block_nums[i] = merge_len / avg_len + 1;
                else
                    block_nums[i] = 1;
            });
            auto block_idx_scan = parlay::scan(block_nums);
            auto range_tmp =
                parlay::sequence<scan_operation>(block_idx_scan.second);
            if (range_num < (length >> 4)) {
                for (int i = 0; i < range_num; i++) {
                    if (block_nums[i] > 1) {
                        int64_t start_idx = block_idx_scan.first[i];
                        int64_t block_size = ops[0].rkey - ops[0].lkey;
                        parfor_wrap(0, block_nums[i], [&](int64_t j) {
                            range_tmp[start_idx + j].lkey =
                                ops_merged[i].lkey + block_size * j;
                            range_tmp[start_idx + j].rkey =
                                ops_merged[i].lkey + block_size * (j + 1);
                            if (range_tmp[start_idx + j].rkey >
                                ops_merged[i].rkey)
                                range_tmp[start_idx + j].rkey =
                                    ops_merged[i].rkey;
                        });
                    } else
                        range_tmp[block_idx_scan.first[i]] = ops_merged[i];
                }
            } else {
                parfor_wrap(0, range_num, [&](int i) {
                    if (block_nums[i] > 1) {
                        int64_t start_idx = block_idx_scan.first[i];
                        int64_t block_size = ops[0].rkey - ops[0].lkey;
                        for (int64_t j = 0; j < block_nums[i]; j++) {
                            range_tmp[start_idx + j].lkey =
                                ops_merged[i].lkey + block_size * j;
                            range_tmp[start_idx + j].rkey =
                                ops_merged[i].lkey + block_size * (j + 1);
                            if (range_tmp[start_idx + j].rkey >
                                ops_merged[i].rkey)
                                range_tmp[start_idx + j].rkey =
                                    ops_merged[i].rkey;
                        }
                    } else
                        range_tmp[block_idx_scan.first[i]] = ops_merged[i];
                });
            }
            ops_merged = range_tmp;
            range_num = block_idx_scan.second;
        });
        printf("Total Range Num: %d\n", length);
        printf("Merged Range Num: %d\n", range_num);
        if (range_num == 1)
            printf("Merged Range: (%lld,%lld)\n", ops_merged[0].lkey,
                   ops_merged[0].rkey);
        dpu_binary_switch_to(dpu_binary::query_binary);

        parlay::sequence<pptr> l3_addrs;
        time_nested("L3 Scan", [&]() {
            auto io = alloc_io_manager();
            io->init();
            IO_Task_Batch *L3_scan_batch;

            L3_scan_batch = io->alloc<L3_scan_task, L3_scan_reply>(direct);
            time_nested("taskgen", [&]() {
                auto batch = L3_scan_batch;
                parfor_wrap(0, nr_of_dpus, [&](size_t i) {
                    int l = range_num * i / nr_of_dpus;
                    int r = range_num * (i + 1) / nr_of_dpus;
                    for (int j = l; j < r; j++) {
                        L3_scan_task *tst =
                            (L3_scan_task *)batch->push_task_zero_copy(i, -1,
                                                                       false);
                        *tst = (L3_scan_task){{.lkey = ops_merged[j].lkey,
                                               .rkey = ops_merged[j].rkey}};
                    }
                });
                io->finish_task_batch();
            });

            time_nested("exec", [&]() { io->exec(); });

            time_nested("get result", [&]() {
                auto batch = L3_scan_batch;
                auto l3_addr_size = parlay::sequence<int>(range_num);
                parfor_wrap(0, nr_of_dpus, [&](size_t i) {
                    int l = range_num * i / nr_of_dpus;
                    int r = range_num * (i + 1) / nr_of_dpus;
                    for (int j = l; j < r; j++) {
                        L3_scan_reply *tsr =
                            (L3_scan_reply *)batch->get_reply(j - l, i);
                        l3_addr_size[j] = max(0, (int)tsr->len - 2);
                    }
                });
                parlay::scan_inclusive_inplace(l3_addr_size);
                res_node_num = (range_num << 1) + l3_addr_size[range_num - 1];
                l3_addrs = parlay::sequence<pptr>(res_node_num);
                parfor_wrap(0, res_node_num,
                            [&](size_t i) { l3_addrs[i] = null_pptr; });
                parfor_wrap(0, nr_of_dpus, [&](size_t i) {
                    int l = range_num * i / nr_of_dpus;
                    int r = range_num * (i + 1) / nr_of_dpus;
                    for (int j = l; j < r; j++) {
                        L3_scan_reply *tsr =
                            (L3_scan_reply *)batch->get_reply(j - l, i);
                        int len = tsr->len;
                        if (len > 0) {
                            l3_addrs[j << 1] = tsr->addr[0];
                            l3_addrs[(j << 1) + 1] = tsr->addr[len - 1];
                            int ss = (range_num << 1) +
                                     ((j == 0) ? 0 : l3_addr_size[j - 1]);
                            for (int k = 0; k < len - 2; k++)
                                l3_addrs[k + ss] = tsr->addr[k + 1];
                        } else {
                            l3_addrs[j << 1] = null_pptr;
                            l3_addrs[(j << 1) + 1] = null_pptr;
                        }
                    }
                });
            });
            io->reset();
        });
#ifdef L3_SKIP_LIST
        printf("L3 skip list scan finished\n");
#else
        printf("L3 AB-Tree scan finished\n");
#endif

        parlay::sequence<pptr> l2_addrs_0;
        time_nested("L2 Scan", [&]() {
            parlay::sequence<pptr> l2_addrs_2, l2_addrs_1;
            time_nested("L2-2 Scan", [&]() {
                l2_addrs_2 = l2_scan_one_round(l3_addrs, ops_merged, 2);
            });
            printf("L2-2 scan finished\n");
            time_nested("L2-1 Scan", [&]() {
                l2_addrs_1 = l2_scan_one_round(l2_addrs_2, ops_merged, 1);
            });
            printf("L2-1 scan finished\n");
            time_nested("L2-0 Scan", [&]() {
                l2_addrs_0 = l2_scan_one_round(l2_addrs_1, ops_merged, 0);
            });
            printf("L2-0 scan finished\n");
        });

        auto l2_addrs_non_null = parlay::tabulate(
            l2_addrs_0.size(),
            [&](int i) { return not_equal_pptr(l2_addrs_0[i], null_pptr); });
        auto l2_addrs_non_null_idx = parlay::pack_index(l2_addrs_non_null);
        auto l1_addrs = parlay::tabulate(
            l2_addrs_non_null_idx.size(),
            [&](int i) { return l2_addrs_0[l2_addrs_non_null_idx[i]]; });
        auto kvs = parlay::sequence<key_value>(l1_addrs.size());

        printf("L1 scan starts\n");
        time_nested("L1", [&]() {
            auto scan_taskpos = parlay::sequence<int32_t>(kvs.size());
            auto io = alloc_io_manager();
            io->init();
            auto L1_search_batch =
                io->alloc<p_get_key_task, p_get_key_reply>(direct);
            time_nested("taskgen", [&]() {
                auto batch = L1_search_batch;
                auto targets = [&](const p_get_key_task &x) {
                    return x.addr.id;
                };
                batch->push_task_from_array_by_isort<true>(
                    kvs.size(),
                    [&](int64_t i) {
                        return (p_get_key_task){.addr = l1_addrs[i]};
                    },
                    // targets, make_slice(op_taskpos, op_taskpos +
                    // kvs.size()));
                    targets, make_slice(scan_taskpos));
                io->finish_task_batch();
            });

            time_nested("exec", [&]() { io->exec(); });

            time_nested("get result", [&]() {
                auto batch = L1_search_batch;
                parfor_wrap(0, kvs.size(), [&](int64_t i) {
                    p_get_key_reply *pgkr = (p_get_key_reply *)batch->get_reply(
                        // op_taskpos[i], l1_addrs[i].id);
                        scan_taskpos[i], l1_addrs[i].id);
                    kvs[i].key = pgkr->key;
                    kvs[i].value = pgkr->value;
                });
            });
            io->reset();
        });
        printf("L1 scan finished\n");

        time_start("reassemble results");
        parlay::sort_inplace(kvs, [&](key_value kv1, key_value kv2) -> bool {
            return (kv1.key < kv2.key) ||
                   ((kv1.key == kv2.key) && (kv1.value < kv2.value));
        });
        auto kv_set =
            parlay::unique(kvs, [&](key_value kv1, key_value kv2) -> bool {
                return (kv1.key == kv2.key);
            });
        int64_t kv_n = kv_set.size();
        auto index_set = parlay::tabulate(length, [&](int i) {
            int64_t ll = 0, rr = kv_n;
            int64_t lkey = ops[i].lkey, rkey = ops[i].rkey;
            int64_t mid, res_ll, res_rr;
            while (rr - ll > 1) {
                mid = (ll + rr) >> 1;
                if (lkey >= kv_set[mid].key)
                    ll = mid;
                else if (rkey < kv_set[mid].key)
                    rr = mid;
                else {
                    break;
                }
            }
            int64_t lll = ll, rrr = rr, mmm = mid;

            ll = lll;
            rr = mmm;
            while (rr - ll > 1) {
                mid = (ll + rr) >> 1;
                if (lkey < kv_set[mid].key)
                    rr = mid;
                else
                    ll = mid;
            }
            if (lkey <= kv_set[ll].key)
                res_ll = ll;
            else
                res_ll = ll + 1;

            ll = mmm;
            rr = rrr;
            while (rr - ll > 1) {
                mid = (ll + rr) >> 1;
                if (rkey <= kv_set[mid].key)
                    rr = mid;
                else
                    ll = mid;
            }
            if (ll >= kv_n - 1)
                res_rr = kv_n;
            else if (kv_set[ll + 1].key <= rkey)
                res_rr = ll + 2;
            else
                res_rr = ll + 1;
            if (res_ll == res_rr || res_ll + 1 == res_rr) {
                if (kv_set[res_ll].key >= lkey && kv_set[res_ll].key <= rkey)
                    res_rr = res_ll + 1;
                else
                    res_rr = res_ll;
            }
            return std::make_pair(res_ll, res_rr);
        });
        time_end("reassemble results");
        return std::make_pair(kv_set, index_set);
    }
};

pim_skip_list *pim_skip_list_drivers;
