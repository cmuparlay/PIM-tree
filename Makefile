DPU_DIR := dpu
HOST_DIR := host
BUILDDIR ?= build
NR_TASKLETS ?= 12
NR_DPUS ?= 2048
CC = g++

define conf_filename
	${BUILDDIR}/.NR_DPUS_$(1)_NR_TASKLETS_$(2).conf
endef
CONF := $(call conf_filename,${NR_DPUS},${NR_TASKLETS})

JUMP_PUSH_TARGET := ${BUILDDIR}/jump_push_skip_list_host
PUSH_PULL_TARGET := ${BUILDDIR}/push_pull_skip_list_host
DPU_TARGET := ${BUILDDIR}/jumppush_pushpull_skip_list_dpu

COMMON_INCLUDES := common
COMMON_INCLUDE_SOURCES := $(wildcard ${COMMON_INCLUDES}/*.h)
HOST_INCLUDES := $(wildcard ${HOST_DIR}/*.hpp) 
HOST_SOURCES := $(wildcard ${HOST_DIR}/*.cpp) 
DPU_INCLUDES := $(wildcard ${DPU_DIR}/*.h)
DPU_SOURCES := $(wildcard ${DPU_DIR}/*.c)

.PHONY: all clean test

__dirs := $(shell mkdir -p ${BUILDDIR})

OLD_COMMON_FLAGS := -Wall -Wextra -Werror -g -I${COMMON_INCLUDES}
COMMON_FLAGS := -Wall -Wextra -g -I${COMMON_INCLUDES} -Ipim_base/include/common
HOST_LIB_FLAGS := -isystem pim_base/argparse/include -isystem pim_base/parlaylib/include  -Ipim_base/include/host -Ipim_base/timer_tree/include
HOST_FLAGS := ${COMMON_FLAGS} -std=c++17 -lpthread -O3 -I${HOST_DIR} ${HOST_LIB_FLAGS} `dpu-pkg-config --cflags --libs dpu` -DNR_TASKLETS=${NR_TASKLETS} -DNR_DPUS=${NR_DPUS}
DPU_FLAGS := ${COMMON_FLAGS} -I${DPU_DIR} -Ipim_base/include/dpu -O2 -DNR_TASKLETS=${NR_TASKLETS}

all: ${PUSH_PULL_TARGET} ${JUMP_PUSH_TARGET} ${DPU_TARGET}

${CONF}:
	$(RM) $(call conf_filename,*,*)
	touch ${CONF}


${JUMP_PUSH_TARGET}: ${HOST_SOURCES} ${HOST_INCLUDES} ${COMMON_INCLUDES} ${COMMON_INCLUDE_SOURCES} ${CONF}
	$(CC) -o $@ ${HOST_SOURCES} ${HOST_FLAGS} -DJUMP_PUSH=1

${PUSH_PULL_TARGET}: ${HOST_SOURCES} ${HOST_INCLUDES} ${COMMON_INCLUDES} ${COMMON_INCLUDE_SOURCES} ${CONF}
	$(CC) -o $@ ${HOST_SOURCES} ${HOST_FLAGS} -DPUSH_PULL=1

${DPU_TARGET}: ${DPU_SOURCES} ${DPU_INCLUDES} ${COMMON_INCLUDES} ${COMMON_INCLUDE_SOURCES} ${CONF}
	dpu-upmem-dpurte-clang ${DPU_FLAGS} -o $@ ${DPU_SOURCES}

clean:
	$(RM) -r $(BUILDDIR)

# test_c: ${HOST_TARGET} ${DPU_TARGET}
# 	./${HOST_TARGET}

# test: test_c

# debug: ${HOST_TARGET} ${DPU_TARGET}
# 	dpu-lldb ${HOST_TARGET}