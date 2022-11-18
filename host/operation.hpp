#pragma once

#include <iostream>
#include <cstdio>
#include <string>
#include "oracle.hpp"
#include "task.hpp"
#include "dpu_control.hpp"
#include "task_framework_host.hpp"
#include "value.hpp"

#include <parlay/primitives.h>
#include <parlay/range.h>
#include <parlay/sequence.h>

using namespace std;

namespace pim_skip_list {

bool init_state = false;
int64_t op_keys[BATCH_SIZE];
int64_t op_results[BATCH_SIZE];
int32_t op_heights[BATCH_SIZE];
int32_t insert_heights[BATCH_SIZE];
pptr op_addrs[BATCH_SIZE];
// pptr op_addrs2[BATCH_SIZE];
int32_t op_taskpos[BATCH_SIZE * 2];
int maxheight;  // setting max height
int L3_id[BATCH_SIZE];

int insert_offset_buffer[BATCH_SIZE * 2];

pptr search_path_addrs_buf[BATCH_SIZE * 2];
pptr insert_path_addrs_buf[BATCH_SIZE * 2];
pptr insert_path_rights_buf[BATCH_SIZE * 2];
int64_t insert_path_chks_buf[BATCH_SIZE * 2];
int insert_path_taskpos_buf[BATCH_SIZE * 2];

pptr *search_path_addrs[BATCH_SIZE];
pptr *insert_path_addrs[BATCH_SIZE];
pptr *insert_path_rights[BATCH_SIZE];
int64_t *insert_path_chks[BATCH_SIZE];
int *insert_path_taskpos[BATCH_SIZE];

template <class F> // [valid, invalid] [l, r)
inline int binary_search_local_l(int l, int r, F f) {
    // ASSERT(l >= 0 && r >= 0 && r > l);
    int mid = (l + r) >> 1;
    while (r - l > 1) {
        if (f(mid)) {
            l = mid;
        } else {
            r = mid;
        }
        mid = (l + r) >> 1;
    }
    return l;
}

template <class F> // [invalid, valid] (l, r]
inline int binary_search_local_r(int l, int r, F f) {
    // ASSERT(l >= 0 && r >= 0 && r > l);
    int mid = (l + r) >> 1;
    while (r - l > 1) {
        if (f(mid)) {
            r = mid;
        } else {
            l = mid;
        }
        mid = (l + r) >> 1;
    }
    return r;
}

void init_skiplist(uint32_t height) {
    ASSERT(height > LOWER_PART_HEIGHT);
    maxheight = height;

    printf("\n********** INIT SKIP LIST **********\n");

    printf("\n**** INIT L2 ****\n");
    pptr l2node = null_pptr;
    pptr l3node = null_pptr;

    {
        auto io = alloc_io_manager();
        ASSERT(io == io_managers[0]);
        io->init();
        auto batch = io->alloc<L2_init_task, L2_init_reply>(direct);

        int target = hash_to_dpu(LLONG_MIN, 0, nr_of_dpus);
        int location;

        auto sit = (L2_init_task*)batch->push_task_zero_copy(target, -1, true, &location);
        *sit = (L2_init_task){{
            .key = LLONG_MIN, .addr = null_pptr, .height = LOWER_PART_HEIGHT}};

        io->finish_task_batch();
        
        ASSERT(io->exec());

        {
            auto reply = (L2_init_reply*)batch->ith(target, location);
            ASSERT(l2node.id == INVALID_DPU_ID);
            l2node = reply->addr;
        }

        io->reset();
    }

    {
        auto io = alloc_io_manager();
        ASSERT(io == io_managers[0]);
        io->init();
        auto batch = io->alloc<L3_init_task, empty_task_reply>(broadcast);
        auto tit = (L3_init_task *)batch->push_task_zero_copy(-1, -1, false);
        *tit = (L3_init_task){{.key = LLONG_MIN,
                              .addr = l2node,
                              .height = height - LOWER_PART_HEIGHT}};
        io->finish_task_batch();
        ASSERT(!io->exec());
    }
}

void init() {
    time_nested("init", [&]() { init_skiplist(26); });
}

void predecessor_L3(int length, int64_t* keys) {
    time_nested("L3", [&]() {
        auto io = alloc_io_manager();
        ASSERT(io == io_managers[0]);
        io->init();
        IO_Task_Batch *L3_search_batch;
        time_nested("taskgen", [&]() {
            L3_search_batch =
                io->alloc<L3_search_task, L3_search_reply>(direct);

            parlay::parallel_for(0, nr_of_dpus, [&](size_t i) {
                int l = (int64_t)length * i / nr_of_dpus;
                int r = (int64_t)length * (i + 1) / nr_of_dpus;
                for (int j = l; j < r; j++) {
                    auto tst =
                        (L3_search_task *)L3_search_batch->push_task_zero_copy(
                            i, -1, true);
                    *tst = (L3_search_task){.key = keys[j]};
                }
            });
        });
        io->finish_task_batch();

        time_nested("exec", [&]() { ASSERT(io->exec()); });

        time_nested("get result", [&]() {
            parlay::parallel_for(0, nr_of_dpus, [&](size_t i) {
                int l = (int64_t)length * i / nr_of_dpus;
                int r = (int64_t)length * (i + 1) / nr_of_dpus;
                for (int j = l; j < r; j++) {
                    auto tsr =
                        (L3_search_reply *)L3_search_batch->ith(i, j - l);
                    op_addrs[j] = tsr->addr;
                    op_heights[j] = LOWER_PART_HEIGHT - 1;
                }
            });
        });
        io->reset();
    });
}

int basebit = 4;
int basew = (1 << basebit);

void init_paths(pptr** paths, int l, int w, int r) {
    int n = (r - l) / w + 1;
    parlay::parallel_for(0, n, [&](size_t x) {
        int i = l + w * x;
        paths[i] = search_path_addrs_buf + x * LOWER_PART_HEIGHT;
    });
}

void get_height(pptr **paths, int l, int w, int r) {
    int n = (r - l) / w + 1;
    parlay::parallel_for(0, n, [&](size_t x) {
        if ((x & 1) == 0) {
            return;
        }
        if ((x + 1) >= n) {
            return;
        }
        int ll = l + (x - 1) * w;
        int rr = l + (x + 1) * w;
        int ii = l + x * w;
        ASSERT(op_heights[ii] == LOWER_PART_HEIGHT - 1);
        ASSERT(op_heights[ll] == -1);
        ASSERT(op_heights[rr] == -1);
        int h = op_heights[ii];
        while (h >= 0 && equal_pptr(paths[ll][h], paths[rr][h])) {
            paths[ii][h] = paths[ll][h];
            h --;
        }
        if (h < op_heights[ii]) {
            op_addrs[ii] = paths[ll][h];
            op_heights[ii] = h;
        }
        // paths[ii][h] = paths[ll][h];
        
    }, 2000);
}

void init_tasks(IO_Task_Batch* L2_search_batch, int l, int w, int r, int64_t* keys) {
    int n = (r - l) / w + 1;
    parlay::parallel_for(
        0, n,
        [&](size_t x) {
            int i = l + w * x;
            if (op_heights[i] == -1) {
                return;
            }
            auto sst = (L2_search_task *)L2_search_batch->push_task_zero_copy(
                op_addrs[i].id, -1, true, op_taskpos + i);
            *sst = (L2_search_task){
                {.key = keys[i], .addr = op_addrs[i], .height = op_heights[i]}};
        },
        1000);
}

void record_paths(IO_Task_Batch* L2_search_batch, pptr** paths, int l, int w, int r) {
    int n = (r - l) / w + 1;
    parlay::parallel_for(
        0, n,
        [&](size_t x) {
            int i = l + w * x;
            ASSERT(i % basew == 0);
            if (op_heights[i] == -1) {
                return;
            }
            L2_search_reply* ssr = (L2_search_reply*)L2_search_batch->ith(
                op_addrs[i].id, op_taskpos[i]);
            pptr nxt_addr = ssr->addr;
            paths[i][op_heights[i]] = nxt_addr;
            if (equal_pptr(nxt_addr, op_addrs[i])) {
                op_heights[i]--;
            } else {
                op_addrs[i] = nxt_addr;
            }
        },
        1000);
}

void build_paths(int length, pptr **paths, int64_t* keys) {
    int bit = 1;
    while ((1 << (bit + 1)) < length) {
        bit += 1;
    }
    for (; bit >= basebit; bit --) {
        int w = 1 << bit;
        string roundname("Push ");
        roundname += std::to_string(bit);
        time_nested(roundname, [&]() {
            get_height(paths, 0, w, length);
            // for (int i = 0; i < length; i += w) {
            //     printf("%d\t%d\n", bit, op_heights[i]);
            // }
            int cnt = 0;
            while (true) {
                cnt ++;
                auto io = alloc_io_manager();
                ASSERT(io == io_managers[0]);
                io->init();

                IO_Task_Batch *L2_search_batch =
                    io->alloc<L2_search_task, L2_search_reply>(direct);
                init_tasks(L2_search_batch, 0, w, length, keys);
                io->finish_task_batch();

                bool finished = false;
                time_nested(
                    "exec", [&]() { finished = !io->exec(); }, false);
                if (finished) {
                    io->reset();
                    break;
                }
                record_paths(L2_search_batch, paths, 0, w, length);
                io->reset();
            }
            printf("%d %d\n", bit, cnt);
        });
    }
}

void predecessor_jump(int length, int64_t* keys, pptr **paths, int* heights) {
    ASSERT(LOWER_PART_HEIGHT == 12);
    pptr** pivot_paths = search_path_addrs;
    init_paths(pivot_paths, 0, basew, length);
    build_paths(length, pivot_paths, keys);

    // exit(1);
    parlay::parallel_for(0, length / basebit, [&](size_t i) {
        int l = i * basew;
        int r = (i + 1) * basew;
        int h = LOWER_PART_HEIGHT - 1;
        if (r >= length) {
            r = length;
        } else {
            while (h >= 0 && equal_pptr(pivot_paths[l][h], pivot_paths[r][h])) {
                h--;
            }
        }
        if (paths != NULL) { // need paths
            for (int ht = h + 1; ht < LOWER_PART_HEIGHT; ht ++) {
                ASSERT(equal_pptr(pivot_paths[l][ht], pivot_paths[r][ht]));
            }
            for (int i = l; i < r; i++) {
                for (int ht = h + 1; ht < heights[i]; ht ++) {
                    paths[i][ht] = pivot_paths[l][ht];
                }
            }
        }
        for (int i = l; i < r; i++) {
            op_addrs[i] = pivot_paths[l][h];
            op_heights[i] = h;
        }
    });

    // auto sorted_heights = parlay::sort(parlay::make_slice(op_heights, op_heights + length));
    // for (int i = 0; i < sorted_heights.size(); i ++) {
    //     printf("%d\t%d\n", i, sorted_heights[i]);
    // }
    // exit(0);
}

void predecessor_push(int length, int64_t* keys, pptr **paths, int* heights) {
    // for (int i = 0; i < length; i ++) {
    //     printf("%d\t%llx\t%d\n", i, pptr_to_int64(op_addrs[i]), op_heights[i]);
    // }
    // exit(1);
    time_nested("Push", [&]() {
        while (true) {
            auto io = alloc_io_manager();
            ASSERT(io == io_managers[0]);
            io->init();

            time_start("taskgen");
            IO_Task_Batch *L2_search_batch =
                io->alloc<L2_search_task, L2_search_reply>(direct);
            // atomic<int> count = 0;
            parlay::parallel_for(0, length, [&](size_t i) {
                if (op_heights[i] == -1) {
                    return;
                }
                auto sst =
                    (L2_search_task *)L2_search_batch->push_task_zero_copy(
                        op_addrs[i].id, -1, true, op_taskpos + i);
                *sst = (L2_search_task){{.key = keys[i],
                                         .addr = op_addrs[i],
                                         .height = op_heights[i]}};
                // count ++;
            });
            io->finish_task_batch();
            // cout<<count<<endl;
            time_end("taskgen");

            bool finished = false;
            time_nested(
                "exec", [&]() { finished = !io->exec(); }, false);
            if (finished) {
                io->reset();
                break;
            }

            time_start("get_result");
            parlay::parallel_for(0, length, [&](size_t i) {
                if (op_heights[i] == -1) {
                    return;
                }
                L2_search_reply *ssr = (L2_search_reply *)L2_search_batch->ith(
                    op_addrs[i].id, op_taskpos[i]);
                pptr nxt_addr = ssr->addr;
                if (equal_pptr(nxt_addr, op_addrs[i])) {
                    if (paths != NULL && op_heights[i] < heights[i]) {
                        paths[i][op_heights[i]] =
                            nxt_addr;  // only record when going downward
                    }
                    op_heights[i]--;
                } else {
                    op_addrs[i] = nxt_addr;
                }
            });
            time_end("get_result");
            io->reset();
        }
    });
}

void predecessor_data(int length, key_value *results) {
    time_nested("Data Node", [&]() {
        auto io = alloc_io_manager();
        ASSERT(io == io_managers[0]);
        io->init();

        IO_Task_Batch *data_get_node_batch =
            io->alloc<L2_get_node_task, L2_get_node_reply>(direct);
        parlay::parallel_for(0, length, [&](size_t i) {
            ASSERT(op_heights[i] == -1);
            auto sgnt =
                (L2_get_node_task *)data_get_node_batch->push_task_zero_copy(
                    op_addrs[i].id, -1, true, op_taskpos + i);
            *sgnt = (L2_get_node_task){
                {.addr = op_addrs[i], .height = op_heights[i]}};
        });
        io->finish_task_batch();

        time_nested(
            "exec", [&]() { ASSERT(io->exec()); }, false);

        parlay::parallel_for(0, length, [&](size_t i) {
            L2_get_node_reply *sgnr =
                (L2_get_node_reply *)data_get_node_batch->ith(op_addrs[i].id,
                                                              op_taskpos[i]);
            results[i] = (key_value){.key = sgnr->chk,
                                     .value = pptr_to_int64(sgnr->right)};
        });
        io->reset();
    });
}

// void build_right_chks(int length, int *heights, pptr **paths, pptr **rights,
//                       int64_t **chks) {
//     auto io = alloc_io_manager();
//     ASSERT(io == io_managers[0]);
//     io->init();

//     IO_Task_Batch *batch =
//         io->alloc<L2_get_node_task, L2_get_node_reply>(direct);
//     parlay::parallel_for(0, length, [&](size_t i) {
//         int offset = paths[i] - insert_path_addrs_buf;
//         for (int j = 0; j < heights[i]; j++) {
//             auto sgnt = (L2_get_node_task *)batch->push_task_zero_copy(
//                 paths[i][j].id, -1, true, op_taskpos + offset + j);
//             *sgnt = (L2_get_node_task){{.addr = paths[i][j], .height = j}};
//         }
//     });
//     io->finish_task_batch();

//     time_nested("exec", [&]() { ASSERT(io->exec()); });

//     parlay::parallel_for(0, length, [&](size_t i) {
//         int offset = paths[i] - insert_path_addrs_buf;
//         for (int j = 0; j < heights[i]; j++) {
//             L2_get_node_reply *sgnr = (L2_get_node_reply *)batch->ith(
//                 paths[i][j].id, op_taskpos[offset + j]);
//             rights[i][j] = sgnr->right;
//             chks[i][j] = sgnr->chk;
//         }
//     });
//     io->reset();
// }

void predecessor_jump_push(int length, int *heights = NULL, pptr **paths = NULL,
                           pptr **rights = NULL, int64_t **chks = NULL,
                           int64_t *keys = NULL, key_value *results = NULL) {
    (void)rights;
    (void)chks;
    predecessor_L3(length, keys);
    predecessor_jump(length, keys, paths, heights);
    predecessor_push(length, keys, paths, heights);
    if (results != NULL) {
        predecessor_data(length, results);
    }
    // if (rights != NULL) {
    //     build_right_chks
    // }
}

void predecessor_core(int length, int32_t *heights = NULL, pptr **paths = NULL,
                      pptr **rights = NULL, int64_t **chks = NULL,
                      int64_t *keys = NULL,
                      key_value *results = NULL) {  // assert keys sorted
    // bool pull_only = (chks != NULL);
    bool pull_only = false;
    bool predecessor_insert = (chks != NULL);
    ASSERT(predecessor_insert == (results == NULL));
    ASSERT(keys != NULL);
    // printf("START PREDECESSOR\n");

    time_nested("L3", [&]() {
        auto io = alloc_io_manager();
        ASSERT(io == io_managers[0]);
        io->init();
        IO_Task_Batch *L3_search_batch;
        time_nested("taskgen", [&]() {
            L3_search_batch =
                io->alloc<L3_search_task, L3_search_reply>(direct);

            parlay::parallel_for(0, nr_of_dpus, [&](size_t i) {
                int l = (int64_t)length * i / nr_of_dpus;
                int r = (int64_t)length * (i + 1) / nr_of_dpus;
                for (int j = l; j < r; j++) {
                    auto tst =
                        (L3_search_task *)L3_search_batch->push_task_zero_copy(
                            i, -1, true);
                    *tst = (L3_search_task){.key = keys[j]};
                }
            });
        });
        io->finish_task_batch();

        time_nested("exec", [&]() { ASSERT(io->exec()); });

        time_nested("get result", [&]() {
            parlay::parallel_for(0, nr_of_dpus, [&](size_t i) {
                int l = (int64_t)length * i / nr_of_dpus;
                int r = (int64_t)length * (i + 1) / nr_of_dpus;
                for (int j = l; j < r; j++) {
                    auto tsr =
                        (L3_search_reply *)L3_search_batch->ith(i, j - l);
                    op_addrs[j] = tsr->addr;
                }
            });
        });
        io->reset();
    });

    time_nested("L2", [&]() {
        parlay::parallel_for(0, length, [&](size_t i) {
            op_heights[i] = LOWER_PART_HEIGHT - 1;
        });

        int cnt = 0;
        auto task_start = parlay::tabulate(length, [](int i) -> bool {
            return not_equal_pptr(op_addrs[i - 1], op_addrs[i]);
        });
        task_start[0] = true;

        auto ll = parlay::pack_index(task_start);
        int llen = ll.size();
        auto rr = parlay::sequence<int>(length, -1);

        parlay::parallel_for(0, llen, [&](size_t i) {
            rr[ll[i]] = (i == llen - 1) ? length : ll[i + 1];
        });

        while (true) {
            cnt++;
            time_start("taskgen");

            time_start("pack");

            auto io = alloc_io_manager();
            ASSERT(io == io_managers[0]);
            io->init();

            auto ll = parlay::pack_index(task_start);
            int llen = ll.size();
            time_end("pack", false);

            IO_Task_Batch *L2_get_node_batch =
                io->alloc<L2_get_node_task, L2_get_node_reply>(direct);
            parlay::parallel_for(0, llen, [&](size_t x) {
                int l = ll[x];
                int r = rr[l];
                if ((r - l == 1) && (op_heights[l] >= 0) &&
                    !pull_only) {  // push
                    return;
                }
                int i = l;
                auto sgnt =
                    (L2_get_node_task *)L2_get_node_batch->push_task_zero_copy(
                        op_addrs[i].id, -1, true, op_taskpos + i);
                *sgnt = (L2_get_node_task){
                    {.addr = op_addrs[i], .height = op_heights[i]}};
            });
            io->finish_task_batch();

            IO_Task_Batch *L2_search_batch;
            if (!pull_only) {
                time_nested("search taskgen", [&]() {
                    L2_search_batch =
                        io->alloc<L2_search_task, L2_search_reply>(direct);
                    parlay::parallel_for(0, llen, [&](size_t x) {
                        int l = ll[x];
                        int r = rr[l];
                        if ((r - l == 1) && (op_heights[l] >= 0)) {
                            int i = l;
                            auto sst = (L2_search_task *)
                                           L2_search_batch->push_task_zero_copy(
                                               op_addrs[i].id, -1, true,
                                               op_taskpos + i);
                            *sst = (L2_search_task){{.key = keys[i],
                                                     .addr = op_addrs[i],
                                                     .height = op_heights[i]}};
                        }
                    });
                    io->finish_task_batch();
                });
            }

            time_end("taskgen", false);

            bool finished = false;
            time_nested(
                "exec", [&]() { finished = !io->exec(); }, false);
            if (finished) {
                io->reset();
                break;
            }

            // parlay::parallel_for(0, length, [&](size_t i) {
            time_nested(
                "get result",
                [&]() {
                    parlay::parallel_for(0, llen, [&](size_t i) {
                        int loop_l = ll[i];
                        // int loop_r = (i == llen - 1) ? length : ll[i +
                        // 1];
                        int loop_r = rr[loop_l];
                        if (loop_r - loop_l == 1 && !pull_only &&
                            op_heights[loop_l] >= 0) {  // push method
                            L2_search_reply *ssr =
                                (L2_search_reply *)L2_search_batch->ith(
                                    op_addrs[loop_l].id, op_taskpos[loop_l]);
                            pptr nxt_addr = ssr->addr;
                            if (equal_pptr(nxt_addr, op_addrs[loop_l])) {
                                if (predecessor_insert) {
                                    int ht = op_heights[loop_l];
                                    if (ht < heights[loop_l]) {
                                        paths[loop_l][ht] = op_addrs[loop_l];
                                        rights[loop_l][ht] =
                                            op_addrs[loop_l];  // mark as
                                                               // invalid
                                    }
                                }
                                op_heights[loop_l]--;
                            } else {
                                op_addrs[loop_l] = nxt_addr;
                            }
                        } else {  // pull method
                            L2_get_node_reply *sgnr =
                                (L2_get_node_reply *)L2_get_node_batch->ith(
                                    op_addrs[loop_l].id, op_taskpos[loop_l]);

                            int64_t chk = sgnr->chk;
                            pptr r = sgnr->right;

                            if (op_heights[loop_l] == -1) {
                                task_start[loop_l] = false;
                                op_heights[loop_l] = -2;
                                if (!predecessor_insert) {
                                    parlay::parallel_for(
                                        loop_l, loop_r,
                                        [&](size_t j) {
                                            results[j] = (key_value){
                                                .key = chk,
                                                .value = pptr_to_int64(r)};
                                        },
                                        1000);
                                }
                            } else {
                                int dividr = binary_search_local_r(
                                    loop_l - 1, loop_r - 1,
                                    [&](int i) { return keys[i] >= chk; });
                                if (keys[dividr] < chk) dividr++;

                                auto additional_update = [&](int lend,
                                                             int rend) {
                                    int ht = op_heights[lend];
                                    // parlay::parallel_for(lend, rend,
                                    // [&](size_t j) {
                                    for (int j = lend; j < rend; j++) {
                                        ASSERT(j < length);
                                        if (ht < heights[j]) {
                                            paths[j][ht] = op_addrs[lend];
                                            rights[j][ht] = r;
                                            chks[j][ht] = chk;
                                        }
                                    }
                                    // });
                                };
                                if (r.id == INVALID_DPU_ID ||
                                    dividr == loop_r) {  // all go down
                                    if (predecessor_insert) {
                                        additional_update(loop_l, loop_r);
                                    }
                                    op_heights[loop_l]--;
                                } else if (dividr == loop_l) {  // all go right
                                    op_addrs[loop_l] = r;
                                } else {  // split
                                    ASSERT(dividr > loop_l && dividr < loop_r);
                                    task_start[dividr] = true;
                                    rr[dividr] = rr[loop_l];
                                    rr[loop_l] = dividr;
                                    // printf("%d*%d*%d*%d\n", loop_l,
                                    // loop_r, dividr, rr[loop_l]);
                                    op_heights[dividr] = op_heights[loop_l];
                                    op_addrs[dividr] = r;
                                    if (predecessor_insert) {
                                        additional_update(loop_l, dividr);
                                    }
                                    op_heights[loop_l]--;
                                }
                            }
                        }
                    });
                },
                false);
            io->reset();
        }
        cout << "Rounds: " << cnt << endl;
    });
}

template <class TT>
struct copy_scan {
    using T = TT;
    copy_scan() : identity(0) {}
    T identity;
    static T f(T a, T b) { return (b == -1) ? a : b; }
};

void insert(slice<key_value *, key_value *> kvs) {
    int n = kvs.size();

    time_start("init");

    parlay::sequence<key_value> kv_sorted;
    time_nested("sort", [&]() {
        kv_sorted =
            parlay::sort(kvs, [](auto t1, auto t2) { return t1.key < t2.key; });
    });

    auto kv_sorted_dedup = parlay::pack(
        move(kv_sorted), parlay::delayed_seq<bool>(n, [&](size_t i) {
            return (i == 0) || (kv_sorted[i] != kv_sorted[i - 1]);
        }));

    n = kv_sorted_dedup.size();

    auto keys =
        parlay::tabulate(n, [&](size_t i) { return kv_sorted_dedup[i].key; });
    auto values = parlay::delayed_seq<int64_t>(
        n, [&](size_t i) { return kv_sorted_dedup[i].value; });

    int length = n;

    printf("\n**** INIT HEIGHT ****\n");

    parlay::parallel_for(0, length, [&](size_t i) {
        int64_t t = parlay::hash64(keys[i]);
        // int32_t t = rn_gen::parallel_rand();
        t = t & (-t);
        int h = __builtin_ctz(t) + 1;
        h = min(h, maxheight);
        insert_heights[i] = h;
    });

    parlay::slice ins_heights_slice =
        parlay::make_slice(insert_heights, insert_heights + length);

    auto height_prefix_sum_pair = parlay::scan(ins_heights_slice);
    auto height_total = height_prefix_sum_pair.second;
    auto height_prefix_sum = height_prefix_sum_pair.first;

    auto insert_path_addrs = parlay::map(height_prefix_sum, [&](int32_t x) {
        return insert_path_addrs_buf + x;
    });
    auto insert_path_rights = parlay::map(height_prefix_sum, [&](int32_t x) {
        return insert_path_rights_buf + x;
    });
    auto insert_path_chks = parlay::map(
        height_prefix_sum, [&](int32_t x) { return insert_path_chks_buf + x; });
    auto insert_path_taskpos = parlay::map(height_prefix_sum, [&](int32_t x) {
        return insert_path_taskpos_buf + x;
    });

    ASSERT(height_total < BATCH_SIZE * 2);
    time_end("init");

    printf("\n**** INSERT PREDECESSOR ****\n");
    time_nested("predecessor", [&]() {
        if (init_state) {
            predecessor_core(length, insert_heights, insert_path_addrs.data(),
                             insert_path_rights.data(), insert_path_chks.data(),
                             keys.data());
        } else {
            #if defined(JUMP_PUSH)
            predecessor_jump_push(length, insert_heights,
                                  insert_path_addrs.data(),
                                  insert_path_rights.data(),
                                  insert_path_chks.data(), keys.data());
            #elif defined(PUSH_PULL)
            predecessor_core(length, insert_heights, insert_path_addrs.data(),
                             insert_path_rights.data(), insert_path_chks.data(),
                             keys.data());
            #else
            printf("Invalid algorithm type. neither jump push nor push pull.");
            assert(false);
            #endif
        }
    });

    time_start("prepare");
    const int BLOCK = 128;

    int node_count[LOWER_PART_HEIGHT + 1];
    int node_count_threadlocal[BLOCK + 1][LOWER_PART_HEIGHT + 1];
    int *node_offset_threadlocal[BLOCK + 1][LOWER_PART_HEIGHT + 1];

    memset(node_count, 0, sizeof(node_count));
    memset(node_count_threadlocal, 0, sizeof(node_count_threadlocal));

    int *node_id[LOWER_PART_HEIGHT + 1];

    std::mutex reduce_mutex;

    parlay::parallel_for(
        0, BLOCK,
        [&](size_t i) {
            int l = (int64_t)length * i / BLOCK;
            int r = (int64_t)length * (i + 1) / BLOCK;
            for (int j = l; j < r; j++) {
                for (int ht = 0; ht < insert_heights[j]; ht++) {
                    if (ht >= LOWER_PART_HEIGHT) break;
                    node_count_threadlocal[i][ht]++;
                }
            }
            reduce_mutex.lock();
            for (int ht = 0; ht < LOWER_PART_HEIGHT; ht++) {
                node_count[ht] += node_count_threadlocal[i][ht];
            }
            reduce_mutex.unlock();
        },
        1);

    node_id[0] = insert_offset_buffer;
    for (int i = 1; i <= LOWER_PART_HEIGHT; i++) {
        node_id[i] = node_id[i - 1] + node_count[i - 1];
        // printf("*1 %d %ld\n", i, node_id[i] - node_id[i - 1]);
    }
    ASSERT(node_id[LOWER_PART_HEIGHT] <= insert_offset_buffer + BATCH_SIZE * 2);

    for (int i = 0; i <= BLOCK; i++) {
        for (int ht = 0; ht < LOWER_PART_HEIGHT; ht++) {
            if (i == 0) {
                node_offset_threadlocal[0][ht] = node_id[ht];
            } else {
                node_offset_threadlocal[i][ht] =
                    node_offset_threadlocal[i - 1][ht] +
                    node_count_threadlocal[i - 1][ht];
            }
            if (i == BLOCK) {
                ASSERT(node_offset_threadlocal[i][ht] == node_id[ht + 1]);
            }
            // printf("*2 %d %d %ld\n", i, ht,
            //        node_offset_threadlocal[i][ht] -
            //        insert_offset_buffer);
        }
    }

    parlay::parallel_for(0, BLOCK, [&](size_t i) {
        int l = (int64_t)length * i / BLOCK;
        int r = (int64_t)length * (i + 1) / BLOCK;
        for (int j = l; j < r; j++) {
            for (int ht = 0; ht < insert_heights[j]; ht++) {
                if (ht >= LOWER_PART_HEIGHT) continue;
                *(node_offset_threadlocal[i][ht]++) = j;
            }
        }
    });
    time_end("prepare");

    time_nested("get r & chk", [&]() {
        auto is_truncator = [&](int ht, int j) {
            if (j == 0) return true;
            int i = node_id[ht][j];
            int l = node_id[ht][j - 1];
            return not_equal_pptr(insert_path_addrs[i][ht],
                                  insert_path_addrs[l][ht]);
        };

        auto io = alloc_io_manager();
        ASSERT(io == io_managers[0]);
        io->init();
        IO_Task_Batch *L2_get_node_batch;

        time_nested("taskgen", [&]() {
            L2_get_node_batch =
                io->alloc<L2_get_node_task, L2_get_node_reply>(direct);
            for (int ht = 0; ht < LOWER_PART_HEIGHT; ht++) {
                parlay::parallel_for(0, node_count[ht], [&](size_t j) {
                    int id = node_id[ht][j];
                    pptr addr = insert_path_addrs[id][ht];
                    pptr right = insert_path_rights[id][ht];
                    // always push in jump-push-search
                    if (is_truncator(ht, j)) {
                        auto sgnt = (L2_get_node_task *)
                                        L2_get_node_batch->push_task_zero_copy(
                                            addr.id, -1, true,
                                            &insert_path_taskpos[id][ht]);
                        *sgnt =
                            (L2_get_node_task){{.addr = addr, .height = ht}};
                    }
                });
            }
        });
        io->finish_task_batch();

        io->exec();
        // ASSERT(io->exec());

        for (int ht = 0; ht < LOWER_PART_HEIGHT; ht++) {
            auto pred = parlay::tabulate(node_count[ht], [&](int j) {
                return is_truncator(ht, j) ? j : -1;
            });
            parlay::scan_inclusive_inplace(pred, copy_scan<int>());
            parlay::parallel_for(0, node_count[ht], [&](size_t j) {
                int l_id = node_id[ht][pred[j]];
                pptr l_addr = insert_path_addrs[l_id][ht];
                pptr l_right = insert_path_rights[l_id][ht];

                L2_get_node_reply *sgnr =
                    (L2_get_node_reply *)L2_get_node_batch->ith(
                        l_addr.id, insert_path_taskpos[l_id][ht]);
                pptr r = sgnr->right;
                int64_t chk = sgnr->chk;
                {
                    int id = node_id[ht][j];
                    insert_path_rights[id][ht] = r;
                    insert_path_chks[id][ht] = chk;
                }
            });
        }
        io->reset();
    });

    printf("\n**** INSERT L2 ****\n");
    auto key_targets = parlay::tabulate(
        length, [&](size_t i) { return hash_to_dpu(keys[i], 0, nr_of_dpus); });

    time_nested("L2", [&]() {
        auto io = alloc_io_manager();
        ASSERT(io == io_managers[0]);
        io->init();
        IO_Task_Batch *L2_insert_batch;

        time_nested("taskgen", [&]() {
            L2_insert_batch =
                io->alloc<L2_insert_task, L2_insert_reply>(direct);
            parlay::parallel_for(0, length, [&](size_t i) {
                auto sit =
                    (L2_insert_task *)L2_insert_batch->push_task_zero_copy(
                        key_targets[i], -1, true, op_taskpos + i);
                *sit = (L2_insert_task){{.key = keys[i],
                                         .addr = int64_to_pptr(values[i]),
                                         .height = insert_heights[i]}};
            });
        });
        io->finish_task_batch();

        ASSERT(io->exec());

        parlay::parallel_for(0, length, [&](size_t i) {
            L2_insert_reply *sir = (L2_insert_reply *)L2_insert_batch->ith(
                key_targets[i], op_taskpos[i]);
            op_addrs[i] = sir->addr;
        });
        io->reset();
    });

    printf("\n**** INSERT L3 ****\n");
    bool reach_L3 = false;
    time_nested("L3", [&]() {
        atomic<int> cnt = 0;
        parlay::parallel_for(0, length, [&](size_t i) {
            if (insert_heights[i] > LOWER_PART_HEIGHT) {
                L3_id[cnt++] = i;
                reach_L3 = true;
            }
        });
        sort(L3_id, L3_id + cnt.load());

        if (!reach_L3) {
            return;
        }

        auto io = alloc_io_manager();
        ASSERT(io == io_managers[0]);
        io->init();
        IO_Task_Batch *L3_insert_batch =
            io->alloc<L3_insert_task, empty_task_reply>(broadcast);

        for (int t = 0; t < cnt; t++) {
            int i = L3_id[t];
            // printf("T3: %d %d %ld\n", t, i, keys[i]);
            auto tit = (L3_insert_task *)L3_insert_batch->push_task_zero_copy(
                -1, -1, false);
            *tit = (L3_insert_task){
                {.key = keys[i],
                 .addr = op_addrs[i],
                 .height = insert_heights[i] - LOWER_PART_HEIGHT}};
        }
        io->finish_task_batch();

        ASSERT(!io->exec());

        // parlay::parallel_for(0, length, [&](size_t i) {
        //     L3_insert_reply *tir =
        //         (L3_insert_reply *)L3_insert_batch->ith(-1,
        //         op_taskpos[i]);
        //     op_addrs2[i] = tir->addr;
        // });
        // io->reset();
    });

    printf("\n**** BUILD L2 LR ****\n");
    time_nested("insert lr", [&]() {
        auto io = alloc_io_manager();
        ASSERT(io == io_managers[0]);
        io->init();
        IO_Task_Batch *L2_build_lr_batch =
            io->alloc<L2_build_lr_task, empty_task_reply>(direct);

        time_nested("taskgen", [&]() {
            bool print = false;
            for (int ht = 0; ht < LOWER_PART_HEIGHT; ht++) {
                time_nested(
                    string("insert_lr_taskgen_details_") + std::to_string(ht),
                    [&]() {
                        parlay::parallel_for(0, node_count[ht], [&](size_t j) {
                            int id = node_id[ht][j];
                            int l = (j == 0) ? -1 : node_id[ht][j - 1];
                            int r = ((int)j == node_count[ht] - 1)
                                        ? -1
                                        : node_id[ht][j + 1];
                            ASSERT(insert_heights[id] > ht);
                            ASSERT(l == -1 || insert_heights[l] > ht);
                            ASSERT(r == -1 || insert_heights[r] > ht);

                            auto adcur = op_addrs[id];

                            if (l == -1 ||
                                not_equal_pptr(insert_path_addrs[id][ht],
                                               insert_path_addrs[l][ht])) {
                                // no new node on the left,
                                // build right of the
                                // predecessor, build left
                                // to the predecessor
                                // ASSERT(l == -1);
                                pptr adl = insert_path_addrs[id][ht];
                                {  // l -> ins
                                    auto sblt =
                                        (L2_build_lr_task *)L2_build_lr_batch
                                            ->push_task_zero_copy(adl.id, -1,
                                                                  true);
                                    *sblt = (L2_build_lr_task){{.addr = adl,
                                                                .chk = keys[id],
                                                                .height = ht,
                                                                .val = adcur}};
                                    if (print) {
                                        printf("%d-%x %ld %ld %d-%x\n",
                                               sblt->addr.id, sblt->addr.addr,
                                               sblt->chk, sblt->height,
                                               sblt->val.id, sblt->val.addr);
                                    }
                                }
                                {  // l <- ins
                                    auto sblt =
                                        (L2_build_lr_task *)L2_build_lr_batch
                                            ->push_task_zero_copy(adcur.id, -1,
                                                                  true);
                                    *sblt =
                                        (L2_build_lr_task){{.addr = adcur,
                                                            .chk = -1,
                                                            .height = -1 - ht,
                                                            .val = adl}};
                                    if (print) {
                                        printf("%d-%x %ld %ld %d-%x\n",
                                               sblt->addr.id, sblt->addr.addr,
                                               sblt->chk, sblt->height,
                                               sblt->val.id, sblt->val.addr);
                                    }
                                }
                            } else {  // insl <- ins
                                auto sblt =
                                    (L2_build_lr_task *)
                                        L2_build_lr_batch->push_task_zero_copy(
                                            adcur.id, -1, true);
                                *sblt =
                                    (L2_build_lr_task){{.addr = adcur,
                                                        .chk = -1,
                                                        .height = -1 - ht,
                                                        .val = op_addrs[l]}};
                                if (print) {
                                    printf("%d-%x %ld %ld %d-%x\n",
                                           sblt->addr.id, sblt->addr.addr,
                                           sblt->chk, sblt->height,
                                           sblt->val.id, sblt->val.addr);
                                }
                            }

                            if (r == -1 ||
                                not_equal_pptr(insert_path_addrs[id][ht],
                                               insert_path_addrs[r][ht])) {
                                // ASSERT(r == -1);
                                pptr adr = insert_path_rights[id][ht];

                                {  // ins -> r
                                    auto sblt =
                                        (L2_build_lr_task *)L2_build_lr_batch
                                            ->push_task_zero_copy(adcur.id, -1,
                                                                  true);
                                    *sblt = (L2_build_lr_task){
                                        {.addr = adcur,
                                         .chk = insert_path_chks[id][ht],
                                         .height = ht,
                                         .val = adr}};
                                    if (print) {
                                        printf("%d-%x %ld %ld %d-%x\n",
                                               sblt->addr.id, sblt->addr.addr,
                                               sblt->chk, sblt->height,
                                               sblt->val.id, sblt->val.addr);
                                    }
                                }

                                if (insert_path_rights[id][ht].id !=
                                    INVALID_DPU_ID) {  // ins <- r
                                    auto sblt =
                                        (L2_build_lr_task *)L2_build_lr_batch
                                            ->push_task_zero_copy(adr.id, -1,
                                                                  true);
                                    *sblt =
                                        (L2_build_lr_task){{.addr = adr,
                                                            .chk = -1,
                                                            .height = -1 - ht,
                                                            .val = adcur}};
                                    if (print) {
                                        printf("%d-%x %ld %ld %d-%x\n",
                                               sblt->addr.id, sblt->addr.addr,
                                               sblt->chk, sblt->height,
                                               sblt->val.id, sblt->val.addr);
                                    }
                                }
                            } else {  // ins -> insr
                                auto sblt =
                                    (L2_build_lr_task *)
                                        L2_build_lr_batch->push_task_zero_copy(
                                            adcur.id, -1, true);
                                *sblt =
                                    (L2_build_lr_task){{.addr = adcur,
                                                        .chk = keys[r],
                                                        .height = ht,
                                                        .val = op_addrs[r]}};
                                if (print) {
                                    printf("%d-%x %ld %ld %d-%x\n",
                                           sblt->addr.id, sblt->addr.addr,
                                           sblt->chk, sblt->height,
                                           sblt->val.id, sblt->val.addr);
                                }
                            }
                        });
                    });
                // for (int j = 0; j < node_count[i]; j ++) {
                //     printf("%d ", node_id[i][j]);
                // }
                // printf("\n");
            }
        });
        io->finish_task_batch();
        time_nested("exec", [&]() { ASSERT(!io->exec()); });
        io->reset();
    });
    // cout<<"FINISHED!"<<endl;

    // for (int i = 0; i < length; i++) {
    //     printf("%d %d ", i, insert_heights[i]);
    //     if (insert_heights[i] > LOWER_PART_HEIGHT) {
    //         print_pptr(op_addrs[i], " ");
    //         print_pptr(op_addrs2[i], "\n");
    //     } else {
    //         print_pptr(op_addrs[i], "\n");
    //     }
    // }
    // exit(-1);
}

auto get(slice<int64_t *, int64_t *> ops) {
    assert(false);
    key_value empty;
    return parlay::sequence<key_value>(ops.size(), empty);
    // return false;
}

void update(slice<key_value *, key_value *> ops) {
    assert(false);
    // return false;
}

auto scan(slice<scan_operation *, scan_operation *> ops) {
    assert(false);
    parlay::sequence<key_value> kvset;
    parlay::sequence<pair<int, int>> indexset;
    return make_pair(kvset, indexset);
    // return false;
}

// bool stop = false;
static auto predecessor(slice<int64_t *, int64_t *> keys) {
    // if (!stop) {
    //     stop = true;
    // } else {
    //     dpu_control::print_log([](int i) { return i == 0; });
    //     exit(0);
    // }
    int n = keys.size();
    auto keys_with_offset = parlay::delayed_seq<pair<int, int64_t>>(
        n, [&](size_t i) { return make_pair(i, keys[i]); });
    parlay::sequence<pair<int, int64_t>> keys_with_offset_sorted;

    time_nested("sort", [&]() {
        keys_with_offset_sorted = parlay::sort(
            keys_with_offset,
            [](auto t1, auto t2) { return t1.second < t2.second; });
    });

    auto keys_sorted = parlay::tabulate(
        n, [&](size_t i) { return keys_with_offset_sorted[i].second; });

    parlay::sequence<key_value> sorted_results(n);
    time_nested("core", [&]() {
        #if defined(JUMP_PUSH)
            predecessor_jump_push(n, NULL, NULL, NULL, NULL, keys_sorted.data(),
                              sorted_results.data());
        #elif defined(PUSH_PULL)
            predecessor_core(n, NULL, NULL, NULL, NULL, keys_sorted.data(),
                            sorted_results.data());
        #else
        printf("Invalid algorithm type. neither jump push nor push pull.");
        assert(false);
        #endif
    });

    auto result = parlay::sequence<key_value>::uninitialized(n);
    time_nested("fill result", [&]() {
        auto idx = parlay::delayed_seq<int>(
            n, [&](size_t i) { return keys_with_offset_sorted[i].first; });
        parlay::parallel_for(
            0, n, [&](size_t i) { result[idx[i]] = sorted_results[i]; });
    });
    return result;
}

void remove(slice<int64_t *, int64_t *> keys) { assert(false); }

}  // namespace pim_skip_list