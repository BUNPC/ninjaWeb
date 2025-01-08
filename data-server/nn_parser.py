# import numpy as np
import nn_shared_memory
import time
import array

def parserLoop():
    sm = nn_shared_memory.NNSharedMemory(main=False)
    
    while not sm.status_shm.buf[sm.STATUS_SHM_IDX['shutdown']]:
        if sm.getStatus('raw_rbuf_wr_idx') != sm.getStatus('raw_rbuf_rd_idx'):
            # get raw data from n states
            si = sm.status_shm.buf[sm.STATUS_SHM_IDX['raw_rbuf_rd_idx']]%sm.RAW_RBUF_SLOTS * sm.RAW_RBUF_SLOT_SIZE
            se = si + sm.RAW_RBUF_SLOT_SIZE
            buf = array.array('B', sm.raw_rbuf.buf[si:se])
            sm.status_shm.buf[sm.STATUS_SHM_IDX['raw_rbuf_rd_idx']] = (sm.status_shm.buf[sm.STATUS_SHM_IDX['raw_rbuf_rd_idx']] + 1) % sm.RAW_RBUF_SLOTS

            # example parse of aux 
            aux_val = float(sum([buf[5+ib]*256**ib for ib in range(3)]) * 3.3/(237)/4095)
            si = sm.status_shm.buf[sm.STATUS_SHM_IDX['disp_rbuf_wr_idx']]
            sm.disp_rbuf[si] = aux_val
            sm.status_shm.buf[sm.STATUS_SHM_IDX['disp_rbuf_wr_idx']] = (sm.status_shm.buf[sm.STATUS_SHM_IDX['disp_rbuf_wr_idx']] + 1) % sm.DISP_RBUF_SIZE
        
        else:
            time.sleep(0.001)
        


    # clean up when shutting down
    sm.close()
    return



# 