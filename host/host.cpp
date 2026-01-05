/*
 * Copyright (c) 2014-2019 - UPMEM
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @file host.c
 * @brief Template for a Host Application Source File.
 */

#define __mram_ptr 

bool print_debug = false;

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdbool>
#include <iostream>

#include "task.hpp"
#include "task_framework_host.hpp"
#include "operation.hpp"
#include "compile.hpp"
#include "driver.hpp"

using namespace std;

#ifdef DIRECT_INTERFACE
const char* interface_type = "direct";
#else
const char* interface_type = "UPMEM";
#define SCHEDULER_DEACTIVATE
#endif

// used for debug
Driver<PIMTreeIndex>* driver;

/**
 * @brief Main of the Host Application.
 */
int main(int argc, char *argv[]) {
    dpu_control::alloc(DPU_ALLOCATE_ALL);
    namespace_pim_interface::pim_interface_init(dpu_set, interface_type);
    std::cout << "Using Interface: " << interface_type << std::endl;
    namespace_pim_interface::do_not_free_dpu_set_when_delete();

    driver = new Driver<PIMTreeIndex>();
    driver->exec(argc, argv);
    delete driver;

    namespace_pim_interface::pim_interface_delete();
    dpu_control::free();
    return 0;
}