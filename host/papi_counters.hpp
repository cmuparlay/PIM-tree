#pragma once

#include <parlay/papi/papi_util_impl.h>
#include <cstring>
#include <string>
#include <cstdio>
using namespace std;

class papi_global_counters {
public:
    #ifdef USE_PAPI
    int64_t startval[nall_cpu_counters];
    int64_t totval[nall_cpu_counters];
    string name;
    #endif
    papi_global_counters(char* _name) {
        #ifdef USE_PAPI
        name = string(_name);
        memset(startval, 0, sizeof(startval));
        memset(totval, 0, sizeof(totval));
        #endif
    }

    void print() {
        #ifdef USE_PAPI
        std::cout<<std::endl<<name<<":"<<std::endl;
        for (int i = 0; i < nall_cpu_counters; i ++) {
            int c = all_cpu_counters[i];
            if (PAPI_query_event(c) != PAPI_OK) {
                std::cout << all_cpu_counters_strings[i] << "=-1" << std::endl;
                continue;
            }
            std::cout<<all_cpu_counters_strings[i]<<totval[i]<<std::endl;
        }
        #endif
    }
};

papi_global_counters l3counters("l3"), l2counters("l2"), l1counters("l1");
papi_global_counters datacounters1("data1"), datacounters2("data2"), datacounters3("data3");
papi_global_counters datacounters("data");

void papi_start_global_counters(papi_global_counters& c) {
#if (defined USE_PAPI) && (defined PAPI_BREAKDOWN)
    // stop all counters
    bool counter_switch = papi_turn_counters(false);
    if (counter_switch) {
        papi_check_counters(parlay::worker_id());
        papi_wait_counters(false, parlay::num_workers());

        memcpy(c.startval, counter_values, sizeof(c.startval));

        // start all counters
        papi_turn_counters(true);
        papi_check_counters(parlay::worker_id());
        papi_wait_counters(true, parlay::num_workers());
    }
#endif
}

void papi_stop_global_counters(papi_global_counters& c) {
#if (defined USE_PAPI) && (defined PAPI_BREAKDOWN)
    // stop all counters
    bool counter_switch = papi_turn_counters(false);
    if (counter_switch) {
        papi_check_counters(parlay::worker_id());
        papi_wait_counters(false, parlay::num_workers());

        for (int i = 0; i < nall_cpu_counters; i ++) {
            int cc = all_cpu_counters[i];
            if (PAPI_query_event(cc) != PAPI_OK) {
                continue;
            }
            assert(c.startval[i] <= counter_values[i]);
            c.totval[i] += counter_values[i] - c.startval[i];
        }
        // memcpy(c.startval, counter_values, sizeof(c.startval));

        // start all counters
        papi_turn_counters(true);
        papi_check_counters(parlay::worker_id());
        papi_wait_counters(true, parlay::num_workers());
    }
#endif
}