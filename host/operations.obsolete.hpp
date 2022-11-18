#pragma once

#include <iostream>
#include <cstdio>
#include <string>
#include <libcuckoo/cuckoohash_map.hh>
#include "task_host.hpp"
#include "timer.hpp"
#include "host.hpp"
#include "util.hpp"
#include <parlay/primitives.h>
#include <parlay/range.h>
#include <parlay/sequence.h>

using namespace std;

extern int64_t epoch_number;

int maxheight; // setting max height

enum predecessor_type { predecessor_insert, predecessor_only };



// void get(int length, int64_t* keys) {
//     assert(false);
//     epoch_number++;
//     // printf("START GET\n");

//     init_io_buffer(false);
//     set_io_buffer_type(L2_GET_TSK, L2_GET_REP);
//     parlay::parallel_for(0, nr_of_dpus, [&](size_t i) {
//         int l = length * i / nr_of_dpus;
//         int r = length * (i + 1) / nr_of_dpus;
//         for (int j = l; j < r; j++) {
//             L2_get_task tgt = (L2_get_task){.key = keys[j]};
//             push_task(&tgt, sizeof(L2_get_task), sizeof(L2_get_task), i);
//         }
//     });
//     ASSERT(exec());
//     parlay::
// }

timer predecessor_L3_task_generate("predecessor_L3_task_generate");
timer predecessor_L3("predecessor_L3");
timer predecessor_exec("predecessor_exec");
timer predecessor_L3_get_result("predecessor_L3_get_result");

void predecessor(predecessor_type type, int length, int64_t* keys) {
    (void)type;

    epoch_number++;
    // printf("START PREDECESSOR\n");
    // auto id = parlay::tabulate(length, [&](int i) { return i; });
    // parlay::sort_inplace(id, [&](int i, int j) { return keys[i] < keys[j]; });
    // for (int i = 0; i < length; i ++) {
    //     printf("%ld\n", keys[i]);
    // }
    // exit(-1);

    predecessor_L3.start();
    {
        predecessor_L3_task_generate.start();
        init_io_buffer(false);
        set_io_buffer_type(L3_SEARCH_TSK, L3_SEARCH_REP);

        auto ll = parlay::sequence<int>(nr_of_dpus);
        auto rr = parlay::sequence<int>(nr_of_dpus);

        parlay::parallel_for(0, nr_of_dpus, [&](size_t i) {
            int64_t l = key_split[i];
            int64_t r =
                ((int)i == nr_of_dpus - 1) ? INT64_MAX : key_split[i + 1];

            int loop_l = binary_search_local_r(
                -1, length, [&](int x) { return l <= keys[x]; });
            int loop_r = binary_search_local_r(
                -1, length, [&](int x) { return r <= keys[x]; });

            // if (r >= keys[id[loop_r]]) loop_r++;
            // printf("%ld\t%ld\t%d\t%d\n", l, r, loop_l, loop_r);
            for (int j = loop_l; j < loop_r; j++) {
                L3_search_task tst = (L3_search_task){.key = keys[j]};
                push_task(&tst, sizeof(L3_search_task), sizeof(L3_search_reply),
                          i);
            }
            ll[i] = loop_l;
            rr[i] = loop_r;
        });
        predecessor_L3_task_generate.end();

        time_nested("exec", [&]() { ASSERT(exec()); });

        predecessor_L3_get_result.start();
        L3_search_reply _;
        apply_to_all_reply(true, _, [&](L3_search_reply &tsr, int i, int j) {
            ASSERT(buffer_state == receive_direct);
            int offset = ll[i] + j;
            ASSERT(offset < rr[i]);
            op_results[offset] = tsr.result_key;
        });
        predecessor_L3_get_result.end();
    }
    predecessor_L3.end();
}

int insert_offset_buffer[BATCH_SIZE * 2];
timer insert_init("insert_init");
timer insert_sort("insert_sort");
timer insert_height("insert_height");
timer insert_taskgen("insert_taskgen");
timer insert_exec("insert_exec");

void insert(int length, insert_task* tasks) {
    // printf("\n********** INIT SKIP LIST **********\n");

    // insert_init.start();

    printf("\n**** INIT HEIGHT ****\n");
    epoch_number++;

    // for (int i = 0; i < length; i += length / 10) {
    //     printf("%lld\n", insert_keys[i]);
    // }
    // {
    //     int s = keys.size();
    //     for (int i = 0; i < s; i++) {
    //         printf("%ld\n", keys[i]);
    //     }
    // }

    auto keys = parlay::delayed_tabulate(length, [&](int i) {
        return tasks[i].key;
    });

    auto values = parlay::delayed_tabulate(length, [&](int i) {
        return tasks[i].value;
    });

    // insert_height.start();
    time_nested("init height", [&]() {
        parlay::parallel_for(0, length, [&](size_t i) {
            int32_t t = randint64(parlay::worker_id());
            t = t & (-t);
            int h = __builtin_ctz(t) + 1;
            h = min(h, maxheight);
            insert_heights[i] = h;
        });
    });

    // insert_height.end();

    // insert_init.end();

    time_nested("taskgen", [&]() {
        printf("\n**** INSERT L3 ****\n");
        init_io_buffer(false);
        set_io_buffer_type(L3_INSERT_TSK, L3_INSERT_REP);

        parlay::parallel_for(0, nr_of_dpus, [&](size_t i) {
            int64_t l = key_split[i];
            int64_t r =
                ((int)i == nr_of_dpus - 1) ? INT64_MAX : key_split[i + 1];
            int ll = binary_search_local_r(-1, length,
                                           [&](int x) { return l <= keys[x]; });
            int rr = binary_search_local_r(-1, length,
                                           [&](int x) { return r <= keys[x]; });
            // if (r >= keys[rr]) rr++;
            ASSERT((int)i != 0 || ll == 0);
            ASSERT((int)i != (nr_of_dpus - 1) || rr == length);
            // printf("%ld\t%ld\t%d\t%d\t%d\n", l, r, ll, rr, rr - ll);
            for (int j = ll; j < rr; j++) {
                if (!(l <= keys[j] && keys[j] < r)) {
                    printf("%ld %ld %ld %d:%d\n", l, r, keys[j], j, length);
                    ASSERT(false);
                }
                L3_insert_task tit =
                    (L3_insert_task){.key = keys[j],
                                    //  .addr = null_pptr,
                                     .height = insert_heights[j],
                                     .value = values[j]};
                push_task(&tit, sizeof(L3_insert_task), 0,
                          i);
            }
        });
    });

    time_nested("exec", [&]() { !exec(); });
    buffer_state = idle;
}

timer remove_task_generate("remove_task_generate");

auto deduplication(int64_t *arr, int &length) {  // assume sorted
    auto seq = parlay::make_slice(arr, arr + length);
    parlay::sort_inplace(seq);

    auto dup = parlay::delayed_tabulate(
        length, [&](int i) -> bool { return i == 0 || seq[i] != seq[i - 1]; });
    auto packed = parlay::pack(seq, dup);
    length = packed.size();
    return packed;
}

void remove(int length, int64_t* remove_keys) {
    remove_task_generate.start();
    epoch_number++;

    // auto keys = deduplication(remove_keys, length);
    auto keys = parlay::make_slice(remove_keys, length);

    // for (int i = 0; i < length; i += length / 10) {
    //     printf("%lld\n", remove_keys[i]);
    // }

    init_io_buffer(false);
    set_io_buffer_type(L3_REMOVE_TSK, EMPTY);
    parlay::parallel_for(0, nr_of_dpus, [&](size_t i) {
        int64_t l = key_split[i];
        int64_t r = ((int)i == nr_of_dpus - 1) ? INT64_MAX : key_split[i + 1];
        int ll = binary_search_local_r(-1, length,
                                       [&](int x) { return l <= keys[x]; });
        int rr = binary_search_local_r(-1, length,
                                       [&](int x) { return r <= keys[x]; });
        // if (r >= keys[rr]) rr++;
        ASSERT((int)i != 0 || ll == 0);
        ASSERT((int)i != (nr_of_dpus - 1) || rr == length);
        for (int j = ll; j < rr; j++) {
            L3_remove_task trt = (L3_remove_task){.key = keys[j]};
            push_task(&trt, sizeof(L3_remove_task), 0, i);
        }
    });
    // exit(-1);
    // for (int i = 0; i < length; i++) {
    //     L3_remove_task trt = (L3_remove_task){.key = keys[i]};
    //     push_task(&trt, sizeof(L3_remove_task), 0, -1);
    // }
    remove_task_generate.end();

    ASSERT(!exec());
}
