#pragma once

#include <iostream>
#include <cstdio>
#include <set>
#include "operations.hpp"
#include "timer.hpp"
#include "util.hpp"
using namespace std;

// const int VALUE_LIMIT = ;

extern bool print_debug;

static set<int64_t> golden_L3;

void init_test_framework() {
    epoch_number = 0;
    golden_L3.clear();
    golden_L3.insert(LLONG_MIN);
    golden_L3.insert(LLONG_MAX);
}

bool get_test(int length, bool check_result) {
    for (int i = 0; i < length; i++) {
        if (randint64(parlay::worker_id()) & 1) {
            auto it = golden_L3.begin();
            int rnd =
                (randint64(parlay::worker_id()) % (golden_L3.size() - 2)) +
                1;  // don't ask -INF or INF
            advance(it, rnd);
            op_keys[i] = *it;
        } else {
            op_keys[i] = randint64(parlay::worker_id());
        }
    }

    // sort(op_keys, op_keys + length);
    cout << "\n*** Start Get Test ***" << endl;
    time_nested("get", [&]() { get(length); });

    // if (print_debug) {
    //     for_each(golden_L3.begin(), golden_L3.end(),
    //              [&](int64_t v) { cout << "*" << v << endl; });
    //     cout << endl;
    // }

    if (check_result) {
        for (int i = 0; i < length; i++) {
            int64_t golden_result =
                (golden_L3.find(op_keys[i]) != golden_L3.end()) ? 1 : 0;
            if (op_results[i] != golden_result) {
                cout << op_keys[i] << ' ' << op_results[i] << ' '
                     << golden_result << endl;
                return false;
            }
        }
    }

    cout << endl << "\n*** End Get Test ***" << endl;
    return true;
}

bool predecessor_test(int length, bool check_result) {
    parlay::parallel_for(0, length, [&](size_t i) {
        op_keys[i] = randint64(parlay::worker_id());
    });

    sort(op_keys, op_keys + length);
    // sort(op_keys, op_keys + length);
    cout << "\n*** Start Predecessor Test ***" << endl;
    time_nested("predecessor", [&]() { predecessor(length); });

    // if (print_debug) {
    //     for_each(golden_L3.begin(), golden_L3.end(),
    //              [&](int64_t v) { cout << "*" << v << endl; });
    // }

    cout << endl;
    if (check_result) {
        for (int i = 0; i < length; i++) {
            set<int64_t>::iterator it = golden_L3.upper_bound(op_keys[i]);
            it--;
            if (op_results[i] != *it) {
                cout << op_keys[i] << ' ' << op_results[i] << ' ' << *it
                     << endl;
                return false;
            }
        }
    }
    cout << endl << "\n*** End Predecessor Test ***" << endl;
    return true;
}

bool insert_test(int length, bool check_result) {
    cout << "\n*** Start Insert Test ***" << endl;
    if (check_result) {
        for (int i = 0; i < length; i++) {
            op_keys[i] = randint64(parlay::worker_id());
            golden_L3.insert(op_keys[i]);
        }
    } else {
        for (int i = 0; i < length; i++) {
            op_keys[i] = randint64(parlay::worker_id());
        }
    }

    time_nested("insert", [&]() { insert(length); });
    cout << endl << "\n*** End Insert Test ***" << endl;
    return true;
}

void remove_test(int length, bool check_result) {
    if (check_result) {
        assert(golden_L3.size() > 2);
    }
    cout << "\n*** Start Remove Test ***" << endl;
    for (int i = 0; i < length; i++) {
        if (randint64(parlay::worker_id()) & 1) {
            auto it = golden_L3.begin();
            int rnd =
                (randint64(parlay::worker_id()) % (golden_L3.size() - 2)) +
                1;  // don't ask -INF or INF
            advance(it, rnd);
            op_keys[i] = *it;
        } else {
            op_keys[i] = randint64(parlay::worker_id());
        }
    }

    time_nested("remove", [&]() { remove(length); });

    if (check_result) {
        auto dedup_keys = deduplication(op_keys, length);
        int l = dedup_keys.size();
        for (int i = 0; i < l; i++) {
            golden_L3.erase(dedup_keys[i]);
        }
    }

    cout << endl << "\n*** End Remove Test ***" << endl;
}

void L3_sancheck() {
    cout << "\n*** Start L3 Sancheck ***" << endl;
    init_io_buffer(true);
    set_io_buffer_type(L3_SANCHECK_TSK, EMPTY);
    L3_sancheck_task tst;
    push_task(&tst, sizeof(L3_sancheck_task), 0, -1);
    ASSERT(!exec());
    print_log(0);
    cout << endl << "\n*** End L3 Sancheck ***" << endl;
}

void L2_print_all_nodes() {
    cout << "\n*** Start L2 Print All Nodes ***" << endl;
    init_io_buffer(true);
    set_io_buffer_type(L2_PRINT_ALL_NODES_TSK, EMPTY);
    L2_print_all_nodes_task spant;
    push_task(&spant, sizeof(L2_print_all_nodes_task), 0, -1);
    ASSERT(!exec());
    print_log(0, true);
    cout << endl << "\n*** End L3 Print All Nodes ***" << endl;
}