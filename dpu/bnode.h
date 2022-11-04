#pragma once

#include <alloc.h>
// #include <barrier.h>
// #include <stdlib.h>

#include "cache.h"
#include "pnode.h"
#include "common.h"
#include "gc.h"

MUTEX_INIT(b_lock);

extern int64_t DPU_ID;

// Bnode
#ifdef IRAM_FRIENDLY
mBptr bbuffer;
#else
__mram_noinit Bnode bbuffer[B_BUFFER_SIZE / sizeof(Bnode)];
#endif
__host uint32_t bcnt = 1;
mBptr bbuffer_start, bbuffer_end;

static inline mBptr alloc_bn() {
    mutex_lock(b_lock);
    pptr recycle = alloc_node(&free_list_bnode, 1);
    mBptr ret;
    if (recycle.id == 0) {
        ret = bbuffer + bcnt;
        bcnt ++;
    } else {
        ret = (mBptr)recycle.addr;
    }
    mutex_unlock(b_lock);
    SPACE_IN_DPU_ASSERT(bcnt < (B_BUFFER_SIZE / sizeof(Bnode)), "rsb! of\n");
    return ret;
}

static inline bool in_bbuffer(mBptr addr) {
    // IN_DPU_ASSERT(
    //     bbuffer_start == bbuffer + 1 &&
    //         bbuffer_end == (bbuffer_start + (B_BUFFER_SIZE / sizeof(Bnode)) -
    //         1),
    //     "ib! inv\n");
    return addr >= bbuffer_start && addr < bbuffer_end;
}

static inline bool b_length_check(mBptr nn) {
    if (!(in_bbuffer(nn))) {
#ifdef BLC_DETAIL
        printf("blc! nn=%x\n", (uint32_t)nn);
        printf("bbuffer=%x dbbuffer=%x pbuffer=%x\n", (uint32_t)bbuffer,
               (uint32_t)dbbuffer, (uint32_t)pbuffer);
#endif
        return false;
    }
    Bnode bn;
    m_read_single(nn, &bn, sizeof(Bnode));
    int nnlen = (int)bn.len;
    mdbptr keysdb = bn.keys, addrsdb = bn.addrs, caddrsdb = bn.caddrs;
    len_addr keysla = keysdb->la, addrsla = addrsdb->la,
             caddrsla = caddrsdb->la;
    if (!(in_dbbuffer(keysdb) && in_dbbuffer(addrsdb) &&
          in_dbbuffer(caddrsdb))) {
#ifdef BLC_DETAIL
        printf("blc! nn=%x keys=%x addrs=%x caddrs=%x\n", (uint32_t)nn,
               (uint32_t)keysdb, (uint32_t)addrsdb, (uint32_t)caddrsdb);
#endif
        return false;
    }
    if ((keysla.len != nnlen) || (addrsla.len != nnlen) ||
        (caddrsla.len != nnlen)) {
#ifdef BLC_DETAIL
        printf("blc! nn=%x keylen=%d addrlen=%d caddrlen=%d nnlen=%d\n",
               (uint32_t)nn, keysla.len, addrsla.len, caddrsla.len, nnlen);
#endif
        return false;
    }
    return true;
}

static inline void b_newnode(mBptr newnode, mdbptr keys, mdbptr addrs,
                             mdbptr caddrs, int64_t height) {
    // IN_DPU_ASSERT(bcnt == 1, "bi! l1cnt\n");
    // IN_DPU_ASSERT((sizeof(Bnode) & 7) == 0, "bn! invlen\n");
    Bnode nn;
    nn.len = 0;
    nn.height = height;
    nn.up = nn.left = nn.right = null_pptr;
    nn.keys = keys;
    nn.addrs = addrs;
    nn.caddrs = caddrs;
    nn.padding = (mdbptr)INVALID_DPU_ADDR;
    data_block_init(keys);
    data_block_init(addrs);
    data_block_init(caddrs);

    m_write_single(&nn, newnode, sizeof(Bnode));
}

static inline void b_insert(mBptr nn, int req_len, int64_t* keys,
                            int64_t* addrs) { // always insert to the end. not ordered.
    // IN_DPU_ASSERT(b_length_check(nn) && req_len > 0, "bi! len\n");
    Bnode bn;
    m_read_single(nn, &bn, sizeof(Bnode));
    int nnlen = (int)bn.len;
    int nnht = (int)bn.height;
    mdbptr keysdb = bn.keys, addrsdb = bn.addrs, caddrsdb = bn.caddrs;

    pptr addr = PPTR(DPU_ID, nn);
    struct data_block wram_keys, wram_addrs, wram_caddrs;
    int64_t* nnkeys = wram_keys.data;
    int64_t* nnaddrs = wram_addrs.data;
    int64_t* nncaddrs = wram_caddrs.data;

    #ifdef SHADOW_SUBTREE
    mcirptr cir = (mcirptr)INVALID_DPU_ADDR;
    if (nnht == CACHE_HEIGHT) {
        cir = reserve_space_cache_init_record(req_len);
    }
    #endif

    for (int i = 0; i <= nnlen; i += DB_SIZE) {
        int curlen = MIN(DB_SIZE, nnlen - i);
        // IN_DPU_ASSERT(in_dbbuffer(keysdb) && in_dbbuffer(addrsdb) &&
        //                   in_dbbuffer(caddrsdb),
        //               "bst! inv2\n");
        if (i + DB_SIZE <= nnlen) {
            len_addr la = keysdb->la;
            keysdb->la = (len_addr){.len = la.len + req_len, .nxt = la.nxt};
            keysdb = la.nxt;
            la = addrsdb->la;
            addrsdb->la = (len_addr){.len = la.len + req_len, .nxt = la.nxt};
            addrsdb = la.nxt;
            la = caddrsdb->la;
            caddrsdb->la = (len_addr){.len = la.len + req_len, .nxt = la.nxt};
            caddrsdb = la.nxt;
        } else {
            nn->len += req_len;
            while (req_len > 0) {
                m_read_single(keysdb, &wram_keys, sizeof(data_block));
                m_read_single(addrsdb, &wram_addrs, sizeof(data_block));
                m_read_single(caddrsdb, &wram_caddrs, sizeof(data_block));
                for (; (curlen < DB_SIZE) && (req_len > 0); curlen++) {
                    if (*keys != INT64_MAX) {
                        nnkeys[curlen] = *keys;
                        nnaddrs[curlen] = *addrs;
                        nncaddrs[curlen] = PPTR_TO_I64(null_pptr);
                        #ifdef SHADOW_SUBTREE
                        if (nnht == CACHE_HEIGHT) {
                            // IN_DPU_ASSERT(in_cirbuffer(cir), "b1! cirinv\n");
                            *cir =
                                (cache_init_record){.source = addr,
                                                    .addr = addr,
                                                    .request = I64_TO_PPTR(*addrs)};
                            cir++;
                        }
                        #endif
                    }
                    req_len--;
                    keys++;
                    addrs++;
                }
                wram_keys.la.len = curlen + req_len;
                wram_addrs.la.len = curlen + req_len;
                wram_caddrs.la.len = curlen + req_len;
                if ((curlen == DB_SIZE) &&
                    (wram_keys.la.nxt == (mdbptr)INVALID_DPU_ADDR)) {
                    // IN_DPU_ASSERT(
                    //     wram_addrs.la.nxt == (mdbptr)INVALID_DPU_ADDR &&
                    //         wram_caddrs.la.nxt == (mdbptr)INVALID_DPU_ADDR,
                    //     "bi! allocerror\n");
                    // IN_DPU_ASSERT(wram_addrs.la.nxt ==
                    // (mdbptr)INVALID_DPU_ADDR,
                    //               "bi! allocerror\n");
                    wram_keys.la.nxt = data_block_allocate();
                    wram_addrs.la.nxt = data_block_allocate();
                    wram_caddrs.la.nxt = data_block_allocate();
                }
                m_write_single(&wram_keys, keysdb, sizeof(data_block));
                m_write_single(&wram_addrs, addrsdb, sizeof(data_block));
                m_write_single(&wram_caddrs, caddrsdb, sizeof(data_block));
                keysdb = wram_keys.la.nxt;
                addrsdb = wram_addrs.la.nxt;
                caddrsdb = wram_caddrs.la.nxt;
                curlen = 0;
            }
            return;
        }
    }
    // IN_DPU_ASSERT(false, "bi! error\n");
}

static inline void b_remove(mBptr nn, int len, int64_t* keys) {
    int tid = me();

    Bnode bn;
    m_read_single(nn, &bn, sizeof(Bnode));
    int nnlen = (int)bn.len;

    mpint64_t tmpbuf = (mpint64_t)push_variable_reply_head(tid);
    mpint64_t mpnnkeys = tmpbuf;
    mppptr mpnnaddrs = (mppptr)(tmpbuf + nnlen);
    mppptr mpnncaddrs = (mppptr)(tmpbuf + nnlen * 2);

    int l1 = data_block_to_mram(bn.keys, mpnnkeys);
    int l2 = data_block_to_mram(bn.addrs, mpnnaddrs);
    int l3 = data_block_to_mram(bn.caddrs, mpnncaddrs);

//     IN_DPU_ASSERT(l1 == nnlen && l2 == nnlen && l3 == nnlen, "br! len\n");
//     for (int k = 0; k < len; k ++) {
//         bool succeed = false;
//         for (int i = 0; i < nnlen; i ++) {
//             if (keys[k] == mpnnkeys[i]) {
//                 succeed = true;
//                 break;
//             }
//         }
//         IN_DPU_ASSERT_EXEC(succeed, {
//             for (int k = 0; k < nnlen; k ++) {
//                 printf("%d %llx %llx\n", k, mpnnkeys[k], pptr_to_int64(mpnnaddrs[k]));
//             }
//             printf("\n");
//             for (int k = 0; k < len; k ++) {
//                 printf("%d %llx\n", k, keys[k]);
//             }
//             pptr ad = PPTR(DPU_ID, nn);
//             printf("br! inv\t%llx\t%lld\t%d\n", pptr_to_int64(ad), nn->height, k);
//         });
//     }

    for (int i = 0; i < nnlen; i ++) {
        int64_t nnkey = mpnnkeys[i];
        for (int k = 0; k < len; k ++) {
            if (keys[k] > nnkey) {
                break;
            }
            if (keys[k] == nnkey) {
                mpnnaddrs[i] = (null_pptr);
            }
        }
    }
    for (int i = 0; i < nnlen; i ++) {
        while (i < nnlen && !valid_pptr(mpnnaddrs[i])) {
            nnlen --;
            mpnnkeys[i] = mpnnkeys[nnlen];
            mpnnaddrs[i] = mpnnaddrs[nnlen];
            mpnncaddrs[i] = mpnncaddrs[nnlen];
        }
    }
    data_block_from_mram(bn.keys, mpnnkeys, nnlen);
    data_block_from_mram(bn.addrs, mpnnaddrs, nnlen);
    data_block_from_mram(bn.caddrs, mpnncaddrs, nnlen);
    nn->len = nnlen;
}

// #define MIN(a, b) (((a) < (b)) ? (a) : (b))
static inline void b_search(mBptr nn, int len, int64_t* keys, int64_t* repkeys,
                            pptr* repaddrs, uint64_t* heights) {
    // IN_DPU_ASSERT(b_length_check(nn), "bsc! len\n");

    Bnode bn;
    m_read_single(nn, &bn, sizeof(Bnode));
    int nnlen = (int)bn.len;
    mdbptr keysdb = bn.keys, addrsdb = bn.addrs, caddrsdb = bn.caddrs;
    int nnht = (int)bn.height;

    struct data_block wram_keys, wram_addrs, wram_caddrs;
    int64_t* nnkeys = wram_keys.data;
    pptr* nnaddrs = (pptr*)wram_addrs.data;
    pptr* nncaddrs = (pptr*)wram_caddrs.data;

    for (int i = 0; i < len; i++) {
        repkeys[i] = INT64_MIN;
    }

// #ifdef KHB_DEBUG
//     bool hasmin = false;
// #endif

    // bool exitt = false;

    for (int i = 0; i < nnlen; i += DB_SIZE) {
        int curlen = MIN(DB_SIZE, nnlen - i);
        // IN_DPU_ASSERT(in_dbbuffer(keysdb) && in_dbbuffer(addrsdb),
        //               "bsc! inv2\n");
        m_read_single(keysdb, &wram_keys, sizeof(data_block));
        m_read_single(addrsdb, &wram_addrs, sizeof(data_block));
        m_read_single(caddrsdb, &wram_caddrs, sizeof(data_block));

        for (int j = 0; j < curlen; j++) {
            int64_t nnkey = nnkeys[j];

            // IN_DPU_ASSERT(valid_pptr(nnaddrs[j], NR_DPUS), "bsc! inv
            // addr!\n");
            for (int k = 0; k < len; k++) {
                if (nnkey <= keys[k] && nnkey >= repkeys[k]) {
                    repkeys[k] = nnkey;
                    pptr actual_addr = nnaddrs[j];
                    pptr local_addr = nncaddrs[j];

                    // IN_DPU_ASSERT(equal_pptr(local_addr, null_pptr),
                    //               "bs! invlocal\n");

                    pptr nxt_step = equal_pptr(local_addr, null_pptr)
                                        ? actual_addr
                                        : local_addr;
                    repaddrs[k] = nxt_step;

                    // IN_DPU_ASSERT(
                    //     (nnht > 0 && in_bbuffer((mBptr)actual_addr.addr) &&
                    //      in_bbuffer((mBptr)nxt_step.addr)) ||
                    //         (nnht == 0 && in_pbuffer((mPptr)actual_addr.addr)
                    //         &&
                    //          in_pbuffer((mPptr)nxt_step.addr)),
                    //     "bsc! nnht\n");

                    if (heights != NULL) {
                        bool leave = (nxt_step.id != DPU_ID) ||
                                     (!in_bbuffer((mBptr)nxt_step.addr));
                        bool record = (heights[k] >= L2_HEIGHT) ||
                                      (heights[k] == 1 && nnht <= 1);
                        if (leave || record) {
                            heights[k] = PPTR_TO_U64(actual_addr);
                            // IN_DPU_ASSERT(heights[k] > L2_HEIGHT, "bsc!
                            // ht\n");
                        } else {
                            // IN_DPU_ASSERT((heights[k] == 1) && (!leave),
                            //               "bsc! ht2\n");
                        }
                    }
                }
            }
        }
        keysdb = wram_keys.la.nxt;
        addrsdb = wram_addrs.la.nxt;
        caddrsdb = wram_caddrs.la.nxt;
    }
}

static inline int nested_search(int len, int64_t* keys, int64_t* repkeys,
                                pptr* addrs, uint64_t* heights,
                                offset_pptr* paths, int siz) {
    int pathlen = 0;
    bool complete = false;

    while (!complete) {
        complete = true;
        int l, r;
        for (l = 0; l < len; l = r) {
            for (r = l + 1; r < len; r++) {
                if (PPTR_TO_I64(addrs[r]) != PPTR_TO_I64(addrs[l])) {
                    break;
                }
            }
            pptr addr = addrs[l];
            if (addr.id == DPU_ID && in_bbuffer((mBptr)addr.addr)) {
                complete = false;
                mBptr nn = (mBptr)addr.addr;
                if (heights != NULL) {
                    b_search(nn, r - l, keys + l, repkeys + l, addrs + l,
                             heights + l);
                    for (int j = l; j < r; j++) {
                        if (heights[j] > L2_HEIGHT) {  // RECORDING ADDRESSES
                            pptr ad = I64_TO_PPTR(heights[j]);
                            paths[pathlen++] = (offset_pptr){
                                .addr = ad.addr, .id = ad.id, .offset = j};
                        }
                        IN_DPU_ASSERT(
                            pathlen <= siz &&
                                (heights[j] > L2_HEIGHT || heights[j] == 1),
                            "ns! inv\n");
                        // IN_DPU_ASSERT_EXEC(
                        //     pathlen <= siz &&
                        //         (heights[j] > L2_HEIGHT || heights[j] == 1),
                        //     {
                        //         printf("ns! p=%d s=%d h=%lu\n", pathlen, siz,
                        //                heights[j]);
                        //         for (int x = 0; x < pathlen; x++) {
                        //             printf("p[%d]=%llx\n", x,
                        //                    PPTR_TO_I64(paths[x]));
                        //         }
                        //     });
                    }
                } else {
                    b_search(nn, r - l, keys + l, repkeys + l, addrs + l, NULL);
                }
            }
        }
    }
    return pathlen;
}

static inline void cache_newnode(mBptr nn, pptr actual_addr, pptr caddr) {
    Bnode bn;
    m_read_single(nn, &bn, sizeof(Bnode));
    int nnlen = (int)bn.len;
    mdbptr addrsdb = bn.addrs, caddrsdb = bn.caddrs;

    struct data_block wram_addrs, wram_caddrs;
    int64_t* nnaddrs = wram_addrs.data;
    int64_t* nncaddrs = wram_caddrs.data;

    for (int i = 0; i < nnlen; i += DB_SIZE) {
        int curlen = MIN(DB_SIZE, nnlen - i);
        // IN_DPU_ASSERT(in_dbbuffer(addrsdb) && in_dbbuffer(caddrsdb),
        //               "cn! inv2\n");
        m_read_single(addrsdb, &wram_addrs, sizeof(data_block));
        m_read_single(caddrsdb, &wram_caddrs, sizeof(data_block));
        for (int j = 0; j < curlen; j++) {
            if (nnaddrs[j] == PPTR_TO_I64(actual_addr)) {
                nncaddrs[j] = PPTR_TO_I64(caddr);
                m_write_single(&wram_caddrs, caddrsdb, sizeof(data_block));
                return;
            }
        }
        addrsdb = wram_addrs.la.nxt;
        caddrsdb = wram_caddrs.la.nxt;
    }
    // IN_DPU_ASSERT(false, "cn! eof\n");
}

static inline mBptr cache_find(mBptr nn, int64_t key, int ht) {
    int nnht = (int)nn->height;
    IN_DPU_ASSERT((nnht == 2 && ht == 1), "ci! inv\n");
    int64_t repkey;
    pptr repaddr;
    while (nnht > ht) {
        b_search(nn, 1, &key, &repkey, &repaddr, NULL);
        IN_DPU_ASSERT(repaddr.id == DPU_ID, "ci! id\n");
        nn = (mBptr)repaddr.addr;
        nnht = (int)nn->height;
    }
    return nn;
}

#if (defined DPU_INSERT) || (defined DPU_BUILD)
static inline void cache_insert(mBptr nn, int64_t key, pptr t_addr, int ht) {
    nn = cache_find(nn, key, ht);
    b_insert(nn, 1, &key, (int64_t*)&t_addr);
}
#endif

static inline void cache_multi_insert(mBptr nn, int len, int64_t* keys, int64_t* t_addrs, int ht) {
    nn = cache_find(nn, keys[0], ht);
    b_insert(nn, len, keys, t_addrs);
}

#ifdef DPU_DELETE
static inline void cache_remove(mBptr nn, int64_t key, int ht) {
    nn = cache_find(nn, key, ht);
    b_remove(nn, 1, &key);
}
#endif

int b_filter_cache_mram(mBptr nn, int64_t l, int64_t r, mpint64_t keys, mpint64_t addrs, mpint64_t caddrs) {
    Bnode bn;
    m_read_single(nn, &bn, sizeof(Bnode));
    int nnlen = (int)bn.len;
    mdbptr keysdb = bn.keys, addrsdb = bn.addrs, caddrsdb = bn.caddrs;

    struct data_block wram_keys, wram_addrs, wram_caddrs;
    int64_t* nnkeys = wram_keys.data;
    int64_t* nnaddrs = wram_addrs.data;
    int64_t* nncaddrs = wram_caddrs.data;


    int len = 0;
    for (int i = 0; i < nnlen; i += DB_SIZE) {
        int curlen = MIN(DB_SIZE, nnlen - i);
        m_read_single(keysdb, &wram_keys, sizeof(data_block));
        m_read_single(addrsdb, &wram_addrs, sizeof(data_block));
        m_read_single(caddrsdb, &wram_caddrs, sizeof(data_block));

        for (int j = 0; j < curlen; j ++) {
            int64_t nnkey = nnkeys[j];
            // IN_DPU_ASSERT(valid_pptr(I64_TO_PPTR(nnaddrs[j]), NR_DPUS),
            //               "bf! inv addr!\n");
            if (nnkey >= l && nnkey <= r) {
                keys[len] = nnkey;
                addrs[len] = nnaddrs[j];
                caddrs[len] = nncaddrs[j];
                len++;
            } else {
            }
        }

        keysdb = wram_keys.la.nxt;
        addrsdb = wram_addrs.la.nxt;
        caddrsdb = wram_caddrs.la.nxt;
    }

    return len;
}

static inline int b_filter_cache(mBptr nn, int64_t l, int64_t r, int64_t* keys,
                                 int64_t* addrs,
                                 int64_t* caddrs) {  // l <= x <= r
    // IN_DPU_ASSERT(b_length_check(nn), "bf! len\n");

    Bnode bn;
    m_read_single(nn, &bn, sizeof(Bnode));
    int nnlen = (int)bn.len;
    mdbptr keysdb = bn.keys, addrsdb = bn.addrs, caddrsdb = bn.caddrs;

    struct data_block wram_keys, wram_addrs, wram_caddrs;
    int64_t* nnkeys = wram_keys.data;
    int64_t* nnaddrs = wram_addrs.data;
    int64_t* nncaddrs = wram_caddrs.data;

    int len = 0;
    for (int i = 0; i < nnlen; i += DB_SIZE) {
        int curlen = MIN(DB_SIZE, nnlen - i);
        // IN_DPU_ASSERT(in_dbbuffer(keysdb) && in_dbbuffer(addrsdb) &&
        //                   in_dbbuffer(caddrsdb),
        //               "bf! inv2\n");

        // IN_DPU_ASSERT_EXEC(
        //     in_dbbuffer(keysdb) && in_dbbuffer(addrsdb) &&
        //         in_dbbuffer(caddrsdb),
        //     {
        //         printf("i=%d nnlen=%d keys=%x addrs=%x caddrs=%x\n", i,
        //         nnlen,
        //                (uint32_t)keysdb, (uint32_t)addrsdb,
        //                (uint32_t)caddrsdb);
        //         return false;
        //     });
        m_read_single(keysdb, &wram_keys, sizeof(data_block));
        m_read_single(addrsdb, &wram_addrs, sizeof(data_block));
        m_read_single(caddrsdb, &wram_caddrs, sizeof(data_block));
        for (int j = 0; j < curlen; j++) {
            int64_t nnkey = nnkeys[j];
            // IN_DPU_ASSERT(valid_pptr(I64_TO_PPTR(nnaddrs[j]), NR_DPUS),
            //               "bf! inv addr!\n");
            if (nnkey >= l && nnkey <= r) {
                keys[len] = nnkey;
                addrs[len] = nnaddrs[j];
                caddrs[len] = nncaddrs[j];
                len++;
            }
        }
        keysdb = wram_keys.la.nxt;
        addrsdb = wram_addrs.la.nxt;
        caddrsdb = wram_caddrs.la.nxt;
    }
    return len;
}

static inline void cache_truncate(mBptr nn, int64_t key, int ht,
                                  varlen_buffer* keysbuf,
                                  varlen_buffer* addrsbuf,
                                  varlen_buffer* caddrsbuf) {
    nn = cache_find(nn, key, ht);

    pptr nptr = PPTR(DPU_ID, nn);

    Bnode bn;
    m_read_single(nn, &bn, sizeof(Bnode));
    int nnlen = (int)bn.len;
    mdbptr keysdb = bn.keys, addrsdb = bn.addrs, caddrsdb = bn.caddrs;

    varlen_buffer_set_capacity(keysbuf, nnlen);
    varlen_buffer_set_capacity(addrsbuf, nnlen);
    varlen_buffer_set_capacity(caddrsbuf, nnlen);

    int newlen = b_filter_cache(nn, INT64_MIN, key - 1, keysbuf->ptr,
                                addrsbuf->ptr, caddrsbuf->ptr);

    keysbuf->len = addrsbuf->len = caddrsbuf->len = newlen;
    data_block_from_buffer(keysdb, keysbuf);
    data_block_from_buffer(addrsdb, addrsbuf);
    data_block_from_buffer(caddrsdb, caddrsbuf);
    nn->len = newlen;
}

#ifdef DPU_SCAN
static inline int b_scan(int64_t bb, int64_t ee, mBptr nn, mpint64_t keys, mpint64_t addrs) {
    int len = 0;
    int64_t lkeys = INT64_MIN;
    Bnode bn;
    m_read_single(nn, &bn, sizeof(Bnode));
    int nnlen = (int)bn.len;
    mdbptr keysdb = bn.keys, addrsdb = bn.addrs;
    struct data_block wram_keys, wram_addrs;
    int64_t* nnkeys = wram_keys.data;
    int64_t* nnaddrs = wram_addrs.data;
    
    for (int i = 0; i < nnlen; i += DB_SIZE) {
        int curlen = MIN(DB_SIZE, nnlen - i);
        m_read_single(keysdb, &wram_keys, sizeof(data_block));
        m_read_single(addrsdb, &wram_addrs, sizeof(data_block));

        for (int j = 0; j < curlen; j ++) {
            int64_t nnkey = nnkeys[j];
            if (nnkey >= bb && nnkey <= ee) {
                keys[len] = nnkey;
                addrs[len] = nnaddrs[j];
                len++;
            } else if(nnkey < bb && nnkey >= lkeys){
                keys[nnlen - 1] = nnkey;
                addrs[nnlen - 1] = nnaddrs[j];
                lkeys = nnkey;
            }
        }

        keysdb = wram_keys.la.nxt;
        addrsdb = wram_addrs.la.nxt;
    }

    if(lkeys != INT64_MIN) {
        lkeys = addrs[nnlen - 1];
        addrs[len] = addrs[0];
        addrs[0] = lkeys;
        len++;
    }
    return len;
}
#endif
