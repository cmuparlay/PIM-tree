#pragma once

#include "dpu_control.hpp"
#include <mutex>
#include <shared_mutex>

#define NULL_pt(type) ((type)-1)

#ifdef SHADOW_SUBTREE
#ifdef DPU_ENERGY
const string dpu_insert_binary = "build/pim_tree_dpu_insert_energy";
const string dpu_delete_binary = "build/pim_tree_dpu_delete_energy";
const string dpu_scan_binary = "build/pim_tree_dpu_scan_energy";
const string dpu_predecessor_binary = "build/pim_tree_dpu_predecessor_energy";
const string dpu_get_update_binary = "build/pim_tree_dpu_get_update_energy";
const string dpu_query_binary = "build/pim_tree_dpu_query_energy";
const string dpu_build_binary = "build/pim_tree_dpu_build_energy";
const string dpu_init_binary = "build/pim_tree_dpu_init_energy";
#else
const string dpu_insert_binary = "build/pim_tree_dpu_insert";
const string dpu_delete_binary = "build/pim_tree_dpu_delete";
const string dpu_scan_binary = "build/pim_tree_dpu_scan";
const string dpu_predecessor_binary = "build/pim_tree_dpu_predecessor";
const string dpu_get_update_binary = "build/pim_tree_dpu_get_update";
const string dpu_query_binary = "build/pim_tree_dpu_query";
const string dpu_build_binary = "build/pim_tree_dpu_build";
const string dpu_init_binary = "build/pim_tree_dpu_init";
#endif
#else
const string dpu_insert_binary = "build/pim_tree_dpu_insert_no_shadow_subtree";
const string dpu_delete_binary = "build/pim_tree_dpu_delete_no_shadow_subtree";
const string dpu_scan_binary = "build/pim_tree_dpu_scan_no_shadow_subtree";
const string dpu_predecessor_binary = "build/pim_tree_dpu_predecessor_no_shadow_subtree";
const string dpu_get_update_binary = "build/pim_tree_dpu_get_update_no_shadow_subtree";
const string dpu_query_binary = "build/pim_tree_dpu_query_no_shadow_subtree";
const string dpu_build_binary = "build/pim_tree_dpu_build_no_shadow_subtree";
const string dpu_init_binary = "build/pim_tree_dpu_init_no_shadow_subtree";
#endif

uint32_t wram_save_pos[NR_DPUS];

inline void init_wram_save_pos() {
    for(int i=0; i<NR_DPUS; i++)
        wram_save_pos[i] = NULL_pt(uint32_t);
}

void dpu_heap_load() {
    time_nested("DPU WRAM recovery", [&]() {
        DPU_FOREACH(dpu_set, dpu, each_dpu) {
            DPU_ASSERT(dpu_prepare_xfer(dpu, &wram_save_pos[each_dpu]));
        }
        DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU,
            "wram_heap_save_addr", 0, sizeof(uint32_t), DPU_XFER_ASYNC));
    });
}

void dpu_heap_save() {
    time_nested("DPU WRAM backup", [&]() {
        DPU_FOREACH(dpu_set, dpu, each_dpu) {
            DPU_ASSERT(dpu_prepare_xfer(dpu, &wram_save_pos[each_dpu]));
        }
        DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_FROM_DPU, "wram_heap_save_addr", 0, sizeof(uint32_t), DPU_XFER_DEFAULT));
    });
}

enum dpu_binary {
    empty,
    insert_binary,
    predecessor_binary,
    get_update_binary,
    query_binary,
    delete_binary,
    scan_binary,
    build_binary,
    init_binary
};
dpu_binary current_dpu_binary = dpu_binary::empty;

inline void dpu_binary_switch_core(dpu_binary target) {
    switch (target) {
        case dpu_binary::insert_binary: {
            dpu_control::load(dpu_insert_binary);
            break;
        }
        case dpu_binary::get_update_binary: {
            dpu_control::load(dpu_get_update_binary);
            break;
        }
        case dpu_binary::delete_binary: {
            dpu_control::load(dpu_delete_binary);
            break;
        }
        case dpu_binary::predecessor_binary: {
            dpu_control::load(dpu_predecessor_binary);
            break;
        }
        case dpu_binary::scan_binary: {
            dpu_control::load(dpu_scan_binary);
            break;
        }
        case dpu_binary::query_binary: {
            dpu_control::load(dpu_query_binary);
            break;
        }
        case dpu_binary::build_binary: {
            dpu_control::load(dpu_build_binary);
            break;
        }
        case dpu_binary::init_binary: {
            dpu_control::load(dpu_init_binary);
            break;
        }
        default: {
            ASSERT(false);
            break;
        }
    }
}

mutex switch_mutex;

inline void dpu_binary_switch_to(dpu_binary target) {
    unique_lock wLock(switch_mutex);
    time_nested("switchto" + std::to_string(target), [&]() {
        if (current_dpu_binary != target) {
            cpu_coverage_timer->end();
            time_nested("lock", [&]() {
                dpu_control::dpu_mutex.lock();
            });
            cpu_coverage_timer->start();
            if (current_dpu_binary != dpu_binary::empty) {
                dpu_heap_save();
                dpu_binary_switch_core(target);
                current_dpu_binary = target;
                dpu_heap_load();
            } else {
                dpu_binary_switch_core(target);
                current_dpu_binary = target;
            }
            time_nested("unlock", [&]() {
                dpu_control::dpu_mutex.unlock();
            });
        }
    });
}