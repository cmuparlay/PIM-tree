#pragma once
#include <parlay/primitives.h>
#include "oracle.hpp"
#include "task.hpp"
#include "dpu_control.hpp"
#include "task_framework_host.hpp"
#include "value.hpp"
using namespace std;
using namespace parlay;

// Range Scan

template<class T = scan_operation>
inline T make_scan_op(int64_t lkey, int64_t rkey) {
    T res;
    res.lkey = min(lkey, rkey);
    res.rkey = max(lkey, rkey);
    return res;
}

template<class T>
class scan_op_rkey_nlt {
public:
    T identity;
    scan_op_rkey_nlt() {identity.rkey = INT64_MIN;}
    static T f(T r, T R) {
        return make_scan_op<T>(R.lkey, max(r.rkey, R.rkey));
    }
};

template<class T>
class copy_scan {
public:
    T identity;
    copy_scan() {identity = 0;}
    static T f(T a, T b) {
        return ((b == -1) ? a : b);
    }
};
