#pragma once
#include <parlay/primitives.h>

#include "dpu_control.hpp"
#include "oracle.hpp"
#include "task.hpp"
#include "task_framework_host.hpp"
#include "value.hpp"

// Range Scan
#include "host_util.hpp"

using namespace std;
using namespace parlay;

namespace pim_skip_list {

bool init_state = false;

parlay::sequence<int64_t> key_split;
parlay::sequence<int64_t> min_key;
int max_height = MAX_L3_HEIGHT - 1;

const int64_t KEY_RANGE_MIN = INT64_MIN;
const uint64_t KEY_RANGE_SIZE = UINT64_MAX;

void init_splits() {
    printf("\n********** INIT SPLITS **********\n");
    key_split = parlay::sequence<int64_t>(nr_of_dpus);
    min_key = parlay::sequence<int64_t>(nr_of_dpus);
    uint64_t split = KEY_RANGE_SIZE / nr_of_dpus;
    key_split[0] = KEY_RANGE_MIN;
    for (int i = 1; i < nr_of_dpus; i++) {
        key_split[i] = key_split[i - 1] + split;
    }
    for (int i = 0; i < nr_of_dpus - 1; i++) {
        min_key[i] = key_split[i + 1];
    }
    min_key[nr_of_dpus - 1] = INT64_MAX;
}

void init_skiplist() {
    printf("\n********** INIT SKIP LIST **********\n");

    auto io = alloc_io_manager();
    io->init();
    auto batch = io->alloc<L3_init_task, empty_task_reply>(direct);
    parlay::sequence<int> location(nr_of_dpus);
    batch->push_task_sorted(
        nr_of_dpus, nr_of_dpus,
        [&](size_t i) {
            return (L3_init_task){{.key = key_split[i],
                                   .value = INT64_MIN,
                                   .height = max_height}};
        },
        [&](size_t i) { return i; }, make_slice(location));
    io->finish_task_batch();
    time_nested("exec", [&]() { ASSERT(!io->exec()); });
    io->reset();
}

void init() {
    time_nested("init splits", init_splits);
    time_nested("init_skiplist", init_skiplist);
}

// Range Scan
template <typename I64Iterator>
int find_target(int64_t key, slice<I64Iterator, I64Iterator> target, int ll = 0,
                int rr = -1) {
    int l = ((ll < 0) ? 0 : ll), r = ((rr <= ll) ? target.size() : rr);
    while (r - l > 1) {
        int mid = (l + r) >> 1;
        if (target[mid] <= key) {
            l = mid;
        } else {
            r = mid;
        }
    }
    return l;
}

// Range Scan
template <typename I64Iterator>
auto find_range_target(scan_operation scan_op,
                       slice<I64Iterator, I64Iterator> target, int ll = 0,
                       int rr = -1) {
    int64_t lkey = scan_op.lkey, rkey = scan_op.rkey;
    int l = ((ll < 0) ? 0 : ll), r = ((rr <= ll) ? target.size() : rr);
    int mid;
    while (r - l > 10) {
        mid = (l + r) >> 1;
        if (target[mid] <= lkey) {
            l = mid;
        } else if (target[mid - 1] >= rkey) {
            r = mid;
        } else {
            break;
        }
    }
    int res_l = find_target(lkey, target, l, r);
    int res_r = find_target(rkey, target, l, r);
    if (res_r >= target.size()) res_r = target.size() - 1;
    return std::make_pair(res_l, res_r);
}

template <typename IntIterator1, typename IntIterator2, typename IntIterator3>
void find_targets(slice<IntIterator1, IntIterator1> in,
                  slice<IntIterator2, IntIterator2> target,
                  slice<IntIterator3, IntIterator3> split) {
    int n = in.size();
    parlay::sequence<int> starts(nr_of_dpus, 0);
    time_nested("bs", [&]() {
        parallel_for(
            0, nr_of_dpus,
            [&](size_t i) {
                starts[i] = find_target(split[i], in);
                while (starts[i] < n && in[starts[i]] < split[i]) {
                    starts[i]++;
                }
            },
            1024 / log2_up(n));
    });
    for (int i = 0; i < nr_of_dpus; i ++) {
        if (starts[i] < n) {
            target[starts[i]] = i;
        }
    }
    parlay::scan_inclusive_inplace(target, parlay::maxm<int>());
}

auto get(slice<int64_t*, int64_t*> keys) {
    int n = keys.size();
    auto splits = make_slice(min_key);

    time_start("find_target");
    auto target = parlay::tabulate(
        n, [&](size_t i) { return find_target(keys[i], splits); });
    time_end("find_target");

    IO_Manager* io;
    IO_Task_Batch* batch;

    auto location = parlay::sequence<int>(n);
    time_nested("taskgen", [&]() {
        io = alloc_io_manager();
        io->init();
        batch = io->alloc<L3_get_task, L3_get_reply>(direct);
        time_nested("push_task", [&]() {
            batch->push_task_from_array_by_isort<false>(
                n, [&](size_t i) { return (L3_get_task){.key = keys[i]}; },
                make_slice(target), make_slice(location));
        });
        io->finish_task_batch();
    });

    time_nested("exec", [&]() { ASSERT(io->exec()); });
    time_start("get_result");
    auto result = parlay::tabulate(n, [&](size_t i) {
        auto reply = (L3_get_reply*)batch->ith(target[i], location[i]);
        if (reply->valid == 1) {
            return (key_value){.key = keys[i], .value = reply->value};
        } else {
            return (key_value){.key = INT64_MIN, .value = INT64_MIN};
        }
    });
    time_end("get_result");
    io->reset();
    return result;
}

void update(slice<key_value*, key_value*> ops) {
    (void)ops;
    assert(false);
}

auto predecessor(slice<int64_t*, int64_t*> keys) {
    int n = keys.size();
    auto splits = make_slice(min_key);

    time_start("find_target");
    auto target = parlay::tabulate(
        n, [&](size_t i) { return find_target(keys[i], splits); });
    time_end("find_target");

    IO_Manager* io;
    IO_Task_Batch* batch;

    auto location = parlay::sequence<int>(n);
    time_nested("taskgen", [&]() {
        io = alloc_io_manager();
        io->init();
        batch = io->alloc<L3_search_task, L3_search_reply>(direct);
        time_nested("push_task", [&]() {
            batch->push_task_from_array_by_isort<false>(
                n, [&](size_t i) { return (L3_search_task){.key = keys[i]}; },
                make_slice(target), make_slice(location));
        });
        io->finish_task_batch();
    });

    time_nested("exec", [&]() { ASSERT(io->exec()); });
    time_start("get_result");
    auto result = parlay::tabulate(n, [&](size_t i) {
        auto reply = (L3_search_reply*)batch->ith(target[i], location[i]);
        return (key_value){.key = reply->key, .value = reply->value};
    });
    time_end("get_result");
    io->reset();
    return result;
}

void insert(slice<key_value*, key_value*> kvs) {
    int n = kvs.size();

    time_nested("sort", [&]() {
        parlay::sort_inplace(kvs);
    });

    auto kv_sorted =
        parlay::pack(kvs, parlay::delayed_seq<bool>(n, [&](size_t i) {
                         return (i == 0) || (kvs[i].key != kvs[i - 1].key);
                     }));
    
    n = kv_sorted.size();
    printf("n=%d\n", n);
    printf("kvs.size=%d\n", (int)kvs.size());

    auto keys_sorted = parlay::delayed_seq<int64_t>(
        n, [&](size_t i) { return kv_sorted[i].key; });

    IO_Manager* io;
    IO_Task_Batch* batch;

    auto target = parlay::sequence<int>(n, 0);
    time_nested("find", [&]() {
        find_targets(make_slice(keys_sorted), make_slice(target),
                     make_slice(key_split));
    });

    auto heights = parlay::sequence<int>(n);
    time_nested("init height", [&]() {
        parlay::parallel_for(0, n, [&](size_t i) {
            int32_t t = rn_gen::parallel_rand();
            t = t & (-t);
            int h = __builtin_ctz(t) + 1;
            h = min(h, max_height);
            heights[i] = h;
        });
    });

    auto location = parlay::sequence<int>(n);
    time_nested("taskgen", [&]() {
        io = alloc_io_manager();
        io->init();
        batch = io->alloc<L3_insert_task, empty_task_reply>(direct);
        batch->push_task_sorted(
            n, nr_of_dpus,
            [&](size_t i) {
                return (L3_insert_task){{.key = kv_sorted[i].key,
                                         .value = kv_sorted[i].value,
                                         .height = heights[i]}};
            },
            [&](size_t i) { return target[i]; }, make_slice(location));
        io->finish_task_batch();
    });

    time_nested("taskgen2", [&]() {
        batch = io->alloc<L3_get_min_task, L3_get_min_reply>(direct);
        batch->push_task_sorted(
            nr_of_dpus, nr_of_dpus,
            [&](size_t i) {
                (void)i;
                return (L3_get_min_task){.key = 0};
            },
            [&](size_t i) { return i; },
            make_slice(location).cut(0, nr_of_dpus));
        io->finish_task_batch();
    });

    time_nested("exec", [&]() { ASSERT(io->exec()); });

    time_nested("result", [&]() {
        for (int i = 0; i < nr_of_dpus; i++) {
            auto rep = (L3_get_min_reply*)batch->ith(i, 0);
            if (!((rep->key <= min_key[i]) || (rep->key == INT64_MAX))) {
                printf("%d\t%lld\t%lld\t%lld\n", i, rep->key, min_key[i], key_split[i]);
                assert(false);
            }
            min_key[i] = min(min_key[i], rep->key);
        }
    });

    io->reset();
    return;
}

void remove(slice<int64_t*, int64_t*> keys) {
    int n = keys.size();

    time_nested("sort", [&]() { parlay::sort_inplace(keys); });

    auto keys_sorted =
        parlay::pack(keys, parlay::delayed_seq<bool>(n, [&](size_t i) {
                         return (i == 0) || (keys[i] != keys[i - 1]);
                     }));

    n = keys_sorted.size();

    IO_Manager* io;
    IO_Task_Batch* batch;

    auto target = parlay::sequence<int>(n, 0);
    time_nested("find", [&]() {
        find_targets(make_slice(keys_sorted), make_slice(target),
                     make_slice(key_split));
    });

    auto location = parlay::sequence<int>(n);
    time_nested("taskgen", [&]() {
        io = alloc_io_manager();
        io->init();
        batch = io->alloc<L3_remove_task, empty_task_reply>(direct);
        batch->push_task_sorted(
            n, nr_of_dpus,
            [&](size_t i) { return (L3_remove_task){.key = keys_sorted[i]}; },
            [&](size_t i) { return target[i]; }, make_slice(location));
        io->finish_task_batch();
    });

    time_nested("taskgen2", [&]() {
        batch = io->alloc<L3_get_min_task, L3_get_min_reply>(direct);
        batch->push_task_sorted(
            nr_of_dpus, nr_of_dpus,
            [&](size_t i) {
                (void)i;
                return (L3_get_min_task){.key = 0};
            },
            [&](size_t i) { return i; },
            make_slice(location).cut(0, nr_of_dpus));
        io->finish_task_batch();
    });

    time_nested("exec", [&]() { ASSERT(io->exec()); });

    time_nested("result", [&]() {
        for (int i = 0; i < nr_of_dpus; i++) {
            auto rep = (L3_get_min_reply*)batch->ith(i, 0);
            assert(rep->key >= min_key[i]);
            min_key[i] = rep->key;
        }
    });

    io->reset();
    cout << "remove finished!" << endl;
    return;
}

// Range Scan
auto scan(slice<scan_operation*, scan_operation*> op_set) {
    int nn = op_set.size();
    int n = op_set.size();
    auto ops_merged = op_set;

    time_start("find_target");
    auto splits = parlay::make_slice(min_key);
    auto target = parlay::tabulate(
        nn, [&](size_t i) { return find_range_target(ops_merged[i], splits); });

    IO_Manager* io;
    IO_Task_Batch* batch;
    auto node_nums = parlay::tabulate(
        nn, [&](size_t i) { return (target[i].second - target[i].first + 1); });
    auto node_num_prefix_scan = parlay::scan(node_nums);
    auto node_num = node_num_prefix_scan.second;
    auto target_scan = parlay::sequence<int>(node_num);
    auto location = parlay::sequence<int>(node_num);
    auto ops_scan = parlay::sequence<L3_scan_task>(node_num);
    auto node_num_prefix_sum = parlay::make_slice(node_num_prefix_scan.first);
    time_end("find_target");

    time_nested("taskgen", [&]() {
        parlay::parallel_for(0, nn, [&](size_t i) {
            for (int j = 0; j < node_nums[i]; j++) {
                target_scan[node_num_prefix_sum[i] + j] = target[i].first + j;
                ops_scan[node_num_prefix_sum[i] + j] =
                    make_scan_op<L3_scan_task>(ops_merged[i].lkey,
                                               ops_merged[i].rkey);
            }
        });
        io = alloc_io_manager();
        io->init();
        batch = io->alloc<L3_scan_task, L3_scan_reply>(direct);
        time_nested("push_task", [&]() {
            batch->push_task_from_array_by_isort<false>(
                node_num, [&](size_t i) { return ops_scan[i]; },
                make_slice(target_scan), make_slice(location));
        });
        io->finish_task_batch();
    });

    time_nested("exec", [&]() { ASSERT(io->exec()); });

    time_start("get_result");
    auto kv_nums = parlay::tabulate(node_num, [&](size_t i) {
        auto reply = (L3_scan_reply*)batch->ith(target_scan[i], location[i]);
        return (reply->length);
    });
    auto kv_nums_prefix_scan = parlay::scan(kv_nums);
    auto kv_num = kv_nums_prefix_scan.second;
    auto kv_nums_prefix_sum = parlay::make_slice(kv_nums_prefix_scan.first);
    auto kv_set1 = parlay::sequence<key_value>(kv_num);
    parlay::parallel_for(0, node_num, [&](size_t i) {
        auto reply = (L3_scan_reply*)batch->ith(target_scan[i], location[i]);
        for (int j = 0; j < kv_nums[i]; j++) {
            kv_set1[kv_nums_prefix_sum[i] + j].key = reply->vals[j];
            kv_set1[kv_nums_prefix_sum[i] + j].value =
                reply->vals[j + reply->length];
        }
    });
    io->reset();
    auto index_set = parlay::tabulate(n, [&](size_t i) {
        if(i != n - 1)
            return make_pair(
                kv_nums_prefix_sum[node_num_prefix_sum[i]],
                kv_nums_prefix_sum[node_num_prefix_sum[i] + node_nums[i] - 1] + kv_nums[node_num_prefix_sum[i] + node_nums[i] - 1]
            );
        else
            return make_pair(
                kv_nums_prefix_sum[node_num_prefix_sum[n - 1]],
                kv_num
            );
    });
    time_end("get_result");

    return make_pair(kv_set1, index_set);
}
};  // namespace pim_skip_list
