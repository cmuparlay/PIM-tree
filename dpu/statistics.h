#pragma once

// OBSOLETE
// MAY NOT WORK.

#include <stdio.h>
#include <string.h>
#include <mutex.h>
#include "data_block.h"

#ifdef DPU_STATISTICS
MUTEX_INIT(statistic_lock);
extern uint32_t l3cnt, bcnt, dbcnt, circnt;

/* -------------------------- Definitions -------------------------- */

#ifdef DPU_STAT_BNODE_LENGTH
#define NODE_NUM_CNT (100)
int num_of_node[NODE_NUM_CNT];
#endif


/* -------------------------- Init -------------------------- */
void statistic_init() {
#ifdef DPU_STAT_BNODE_LENGTH
    memset(num_of_node, 0, sizeof(num_of_node));
#endif
}

/* -------------------------- Base -------------------------- */
void add_i(int *x, int a) {
    mutex_lock(statistic_lock);
    *x += a;
    mutex_unlock(statistic_lock);
}

void mul_i(int *x, int a) {
    mutex_lock(statistic_lock);
    *x *= a;
    mutex_unlock(statistic_lock);
}

void add_f(float *x, float a) {
    mutex_lock(statistic_lock);
    *x += a;
    mutex_unlock(statistic_lock);
}

void mul_f(float *x, float a) {
    mutex_lock(statistic_lock);
    *x *= a;
    mutex_unlock(statistic_lock);
}

/* -------------------------- Functions -------------------------- */
#ifdef DPU_STAT_BNODE_LENGTH
void node_count_add(int n, int a) {
    add_i(&num_of_node[MIN(n, NODE_NUM_CNT - 1)], a);
}
#endif
 
/* -------------------------- Print -------------------------- */
void print_statistics() {
#ifdef DPU_STAT_BNODE_LENGTH
    for (int i = 0; i < NODE_NUM_CNT; i++) {
        printf("#size%d = %d\n", i, num_of_node[i]);
    }
#endif
    // printf("l3cnt=%d bcnt=%d dbcnt=%d pcnt=%d circnt=%d\n", l3cnt, bcnt, dbcnt, pcnt, circnt);
    // printf("circnt=%d pcnt=%d\n", circnt, pcnt);
    // printf("bcnt=%d pcnt=%d\n", circnt, pcnt);
    printf("l3_size=%d b_header_size=%d db_size=%d p_size=%d\n", l3cnt, bcnt * sizeof(Bnode), dbcnt * sizeof(data_block), pcnt * sizeof(Pnode));
}

#else
void statistic_init() {}

void add_i(int *x, int a) {
    (void)x;
    (void)a;
}

void mul_i(int *x, int a) {
    (void)x;
    (void)a;
}

void add_f(float *x, float a) {
    (void)x;
    (void)a;
}

void mul_f(float *x, float a) {
    (void)x;
    (void)a;
}

void print_statistics() {}
#endif