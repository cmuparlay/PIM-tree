#pragma once

#ifdef L3_SKIP_LIST
#include "l3_skip_list.h"
#else
#include "l3_ab_tree.h"
#endif

#include "bnode.h"
#include "hashtable_l3size.h"
#include "statistics.h"
#include "macro.h"

__host mpuint8_t wram_heap_save_addr = NULL_pt(mpuint8_t);  // IRAM friendly

__mram_noinit Bnode bbuffer_tmp[B_BUFFER_SIZE / sizeof(Bnode)];
__mram_noinit cache_init_record cirbuffer_tmp[CACHE_INIT_RECORD_SIZE / sizeof(cache_init_record)];
__mram_noinit data_block dbbuffer_tmp[DB_BUFFER_SIZE / sizeof(data_block)];

#ifdef L3_SKIP_LIST
__mram_noinit uint8_t l3buffer_tmp[L3_BUFFER_SIZE];
#else
__mram_noinit L3Bnode l3bbuffer_tmp[L3_BUFFER_SIZE / sizeof(L3Bnode)];
#endif

__mram_noinit Pnode pbuffer_tmp[P_BUFFER_SIZE / sizeof(Pnode)];
__mram_noinit ht_slot ht_tmp[LX_HASHTABLE_SIZE];
__mram_noinit int64_t send_varlen_offset_tmp[NR_TASKLETS][MAX_TASK_COUNT_PER_TASKLET_PER_BLOCK];
__mram_noinit uint8_t send_varlen_buffer_tmp[NR_TASKLETS][MAX_TASK_BUFFER_SIZE_PER_TASKLET];

// dpu.c
extern int64_t DPU_ID; // = -1;
#ifdef L3_SKIP_LIST
extern mL3ptr root;
#else
extern mL3Bptr root;
#endif
extern bool wram_init_flag;

// bnode.h
extern uint32_t bcnt; // = 1;
extern mBptr bbuffer_start, bbuffer_end;
extern mBptr bbuffer;
// __mram_noinit Bnode bbuffer[B_BUFFER_SIZE / sizeof(Bnode)];

// cache.h
extern uint32_t circnt; // = 1;
extern mcirptr cirbuffer;
extern mcirptr cirbuffer_start, cirbuffer_end;
// __mram_noinit cache_init_record cirbuffer[CACHE_INIT_RECORD_SIZE / sizeof(cache_init_record)];

// data_block.h
extern uint32_t dbcnt; // = 1;
extern mdbptr dbbuffer_start, dbbuffer_end;
extern mdbptr dbbuffer;
// __mram_noinit data_block dbbuffer[DB_BUFFER_SIZE / sizeof(data_block)];

// l3.h
#ifdef L3_SKIP_LIST
extern uint32_t l3cnt; // = 8;
extern mpuint8_t l3buffer;
// __mram_noinit uint8_t l3buffer[L3_BUFFER_SIZE];
#else
extern uint32_t l3bcnt; // = 1;
extern mL3Bptr l3bbuffer;
// __mram_noinit L3Bnode l3bbuffer[L3_BUFFER_SIZE / sizeof(L3Bnode)];
#endif

// pnode.h
extern uint32_t pcnt; // = 1;
extern mPptr pbuffer_start, pbuffer_end;
extern mPptr pbuffer;
// __mram_noinit Pnode pbuffer[P_BUFFER_SIZE / sizeof(Pnode)];

// statistics.h
#ifdef DPU_STATISTICS
extern int num_of_node[]; // [NODE_NUM_CNT];
#endif

// storage.h
extern int htcnt; // = 0;
extern __mram_ptr ht_slot * ht;
// __mram_noinit ht_slot ht[LX_HASHTABLE_SIZE];

// task_dpu.h
extern mpint64_t send_varlen_offset[];
extern mpuint8_t send_varlen_buffer[];
// __mram_noinit int64_t send_varlen_offset[NR_TASKLETS][MAX_TASK_COUNT_PER_TASKLET_PER_BLOCK];
// __mram_noinit uint8_t send_varlen_buffer[NR_TASKLETS][MAX_TASK_BUFFER_SIZE_PER_TASKLET];
// extern mpuint8_t recv_buffer;
// extern mpuint8_t send_buffer;

extern gcnode free_list_bnode;
extern gcnode free_list_pnode;
extern gcnode free_list_bnode_tmp;
extern gcnode free_list_l3bnode;
extern gcnode free_list_data_block;

typedef struct WRAMHeap {

    int64_t DPU_ID;
    #ifdef L3_SKIP_LIST
    mL3ptr root;
    #else
    mL3Bptr root;
    #endif

    uint32_t bcnt;
    mBptr bbuffer;
    mBptr bbuffer_start;
    mBptr bbuffer_end;

    uint32_t circnt;
    mcirptr cirbuffer;
    mcirptr cirbuffer_start;
    mcirptr cirbuffer_end;

    uint32_t dbcnt;
    mdbptr dbbuffer;
    mdbptr dbbuffer_start;
    mdbptr dbbuffer_end;
    
    #ifdef L3_SKIP_LIST
    uint32_t l3cnt;
    mpuint8_t l3buffer;
    #else
    uint32_t l3bcnt;
    mL3Bptr l3bbuffer;
    mL3Bptr l3bbuffer_start;
    mL3Bptr l3bbuffer_end;
    #endif

    uint32_t pcnt;
    mPptr pbuffer;
    mPptr pbuffer_start;
    mPptr pbuffer_end;

    #ifdef DPU_STATISTICS
    int num_of_node[NODE_NUM_CNT];
    #endif

    int htcnt;
    __mram_ptr ht_slot * ht;

    mpint64_t send_varlen_offset[NR_TASKLETS];
    mpuint8_t send_varlen_buffer[NR_TASKLETS];

    gcnode free_list_bnode;
    gcnode free_list_bnode_tmp;
    gcnode free_list_pnode;
    gcnode free_list_l3bnode;
    gcnode free_list_data_block;

#ifdef DPU_ENERGY
    uint64_t op_cnt;
    uint64_t db_size_cnt;
    uint64_t cycle_cnt;
#endif

} WRAMHeap; //` __attribute__((aligned (8)));

__mram_noinit uint8_t wram_heap_save_addr_tmp[sizeof(WRAMHeap) << 1];

void wram_heap_save() {
    mpuint8_t saveAddr = wram_heap_save_addr;
    WRAMHeap heapInfo = (WRAMHeap){
        .DPU_ID = DPU_ID,
        .root = root,
        .bcnt = bcnt,
        .bbuffer = bbuffer,
        .bbuffer_start = bbuffer_start,
        .bbuffer_end = bbuffer_end,
        .circnt = circnt,
        .cirbuffer = cirbuffer,
        .cirbuffer_start = cirbuffer_start,
        .cirbuffer_end = cirbuffer_end,
        .dbcnt = dbcnt,
        .dbbuffer = dbbuffer,
        .dbbuffer_start = dbbuffer_start,
        .dbbuffer_end = dbbuffer_end,
        #ifdef L3_SKIP_LIST
        .l3cnt = l3cnt,
        .l3buffer = l3buffer,
        #else
        .l3bcnt = l3bcnt,
        .l3bbuffer = l3bbuffer,
        .l3bbuffer_start = l3bbuffer_start,
        .l3bbuffer_end = l3bbuffer_end,
        #endif
        .pcnt = pcnt,
        .pbuffer = pbuffer,
        .pbuffer_start = pbuffer_start,
        .pbuffer_end = pbuffer_end,
        .htcnt = htcnt,
        .ht = ht,
        .free_list_bnode = free_list_bnode,
        .free_list_bnode_tmp = free_list_bnode_tmp,
        .free_list_data_block = free_list_data_block,
        .free_list_l3bnode = free_list_l3bnode,
        .free_list_pnode = free_list_pnode,
#ifdef DPU_ENERGY
        .op_cnt = op_count,
        .db_size_cnt = db_size_count,
        .cycle_cnt = cycle_count,
#endif
    };
    #ifdef DPU_STATISTICS
    for(int i=0; i<NODE_NUM_CNT; i++)
        heapInfo.num_of_node[i] = num_of_node[i];
    #endif
    for(int i=0; i<NR_TASKLETS; i++){
        heapInfo.send_varlen_offset[i] = send_varlen_offset[i];
        heapInfo.send_varlen_buffer[i] = send_varlen_buffer[i];
    }

    if(saveAddr == NULL_pt(mpuint8_t)) saveAddr = wram_heap_save_addr_tmp;
    mram_write(&heapInfo, (mpuint8_t)saveAddr, sizeof(WRAMHeap));
    wram_heap_save_addr = saveAddr;
}

void wram_heap_init() {
    bbuffer = bbuffer_tmp;
    cirbuffer = cirbuffer_tmp;
    dbbuffer = dbbuffer_tmp;
    pbuffer = pbuffer_tmp;
    
    #ifdef L3_SKIP_LIST
    l3buffer = l3buffer_tmp;
    #else
    l3bbuffer = l3bbuffer_tmp;
    #endif
    ht = ht_tmp;
    
    statistic_init();
    for(int i=0; i<NR_TASKLETS; i++) {
        send_varlen_offset[i] = &(send_varlen_offset_tmp[i][0]);
        send_varlen_buffer[i] = &(send_varlen_buffer_tmp[i][0]);
    }
#ifdef DPU_ENERGY
    op_count = 0;
    db_size_count = 0;
    cycle_count = 0;
#endif
}

void wram_heap_load() {
    mpuint8_t saveAddr = wram_heap_save_addr;
    if(saveAddr == NULL_pt(mpuint8_t)) wram_heap_init();
    else {
        WRAMHeap heapInfo;
        mram_read((mpuint8_t)saveAddr, &heapInfo, sizeof(WRAMHeap));

        DPU_ID = heapInfo.DPU_ID;
        root = heapInfo.root;
        bcnt = heapInfo.bcnt;
        bbuffer = heapInfo.bbuffer;
        bbuffer_start = heapInfo.bbuffer_start;
        bbuffer_end = heapInfo.bbuffer_end;
        circnt = heapInfo.circnt;
        cirbuffer = heapInfo.cirbuffer;
        cirbuffer_start = heapInfo.cirbuffer_start;
        cirbuffer_end = heapInfo.cirbuffer_end;
        dbcnt = heapInfo.dbcnt;
        dbbuffer = heapInfo.dbbuffer;
        dbbuffer_start = heapInfo.dbbuffer_start;
        dbbuffer_end = heapInfo.dbbuffer_end;
        #ifdef L3_SKIP_LIST
        l3cnt = heapInfo.l3cnt;
        l3buffer = heapInfo.l3buffer;
        #else
        l3bcnt = heapInfo.l3bcnt;
        l3bbuffer = heapInfo.l3bbuffer;
        l3bbuffer_start = heapInfo.l3bbuffer_start;
        l3bbuffer_end = heapInfo.l3bbuffer_end;
        #endif
        pcnt = heapInfo.pcnt;
        pbuffer = heapInfo.pbuffer;
        pbuffer_start = heapInfo.pbuffer_start;
        pbuffer_end = heapInfo.pbuffer_end;
        htcnt = heapInfo.htcnt;
        ht = heapInfo.ht;
        free_list_bnode = heapInfo.free_list_bnode;
        free_list_bnode_tmp = heapInfo.free_list_bnode_tmp;
        free_list_data_block = heapInfo.free_list_data_block;
        free_list_l3bnode = heapInfo.free_list_l3bnode;
        free_list_pnode = heapInfo.free_list_pnode;

        #ifdef DPU_STATISTICS
        for(int i=0; i<NODE_NUM_CNT; i++)
            num_of_node[i] = heapInfo.num_of_node[i];
        #endif
        for(int i=0; i<NR_TASKLETS; i++){
            send_varlen_offset[i] = heapInfo.send_varlen_offset[i];
            send_varlen_buffer[i] = heapInfo.send_varlen_buffer[i];
        }
#ifdef DPU_ENERGY
        op_count = heapInfo.op_cnt;
        db_size_count = heapInfo.db_size_cnt;
        cycle_count = heapInfo.cycle_cnt;
#endif
    }
}
