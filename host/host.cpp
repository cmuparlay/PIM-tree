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
#include "driver.hpp"

using namespace std;

#ifdef DIRECT_INTERFACE
const char* interface_type = "direct";
#else
const char* interface_type = "UPMEM";
#define SCHEDULER_DEACTIVATE
#endif

#ifdef DPU_ENERGY
const string dpu_binary = "build/range_partitioning_skip_list_dpu_energy";
#else
const string dpu_binary = "build/range_partitioning_skip_list_dpu";
#endif

/**
 * @brief Main of the Host Application.
 */
int main(int argc, char *argv[]) {
    dpu_control::alloc(DPU_ALLOCATE_ALL);
    namespace_pim_interface::pim_interface_init(dpu_set, interface_type);
    std::cout << "Using Interface: " << interface_type << std::endl;
    namespace_pim_interface::do_not_free_dpu_set_when_delete();

    dpu_control::load(dpu_binary);

    Driver<pim_skip_list> driver;
    driver.exec(argc, argv);

    namespace_pim_interface::pim_interface_delete();
    dpu_control::free();
    return 0;
}