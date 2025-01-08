from multiprocessing import shared_memory

class NNSharedMemory:
    # Status shared memory
    STATUS_SHM_NAME = "nn_status_shm"
    STATUS_SHM_FIELDS_INIT = [('run', False), \
                              ('disp_rbuf_rd_idx', 0), \
                              ('disp_rbuf_wr_idx', 0), \
                              ('raw_rbuf_rd_idx', 0), \
                              ('raw_rbuf_wr_idx', 0), \
                              ('shutdown', False) ]
    STATUS_SHM_IDX = dict([ (key[0], idx) for idx, key in enumerate(STATUS_SHM_FIELDS_INIT) ])
    STATUS_SHM_SIZE = len(STATUS_SHM_FIELDS_INIT)

    # Display ring buffer
    DISP_RBUF_NAME = "nn_disp_rbuf"
    DISP_RBUF_SIZE = 2**8
    DISP_RBUF_INIT = [float('nan')] * DISP_RBUF_SIZE

    # Raw data ring buffer
    # one state packet per slot, this avoids packets wrapping across the end of the buffer
    RAW_RBUF_NAME = "nn_raw_rbuf"
    RAW_RBUF_SLOTS = 16
    RAW_RBUF_SLOT_SIZE = 1024
    RAW_RBUF_SIZE = RAW_RBUF_SLOTS * RAW_RBUF_SLOT_SIZE


    def __init__(self, main = False):
        self.main = main

        if main:
            self.status_shm = shared_memory.SharedMemory(name=self.STATUS_SHM_NAME, create=True, size=self.STATUS_SHM_SIZE)
            self.status_shm.buf[:] = bytearray([x[1] for x in self.STATUS_SHM_FIELDS_INIT]) # initialize status shared memory
            self.disp_rbuf = shared_memory.ShareableList(name=self.DISP_RBUF_NAME, sequence=self.DISP_RBUF_INIT)
            self.raw_rbuf = shared_memory.SharedMemory(name=self.RAW_RBUF_NAME, create=True, size=self.RAW_RBUF_SIZE)
        else:
            self.status_shm = shared_memory.SharedMemory(name=self.STATUS_SHM_NAME, create=False)
            self.disp_rbuf = shared_memory.ShareableList(name=self.DISP_RBUF_NAME)
            self.raw_rbuf = shared_memory.SharedMemory(name=self.RAW_RBUF_NAME, create=False)

    def getStatus(self, key='run'):
        return self.status_shm.buf[self.STATUS_SHM_IDX[key]]

    def close(self):
        self.status_shm.close()
        self.disp_rbuf.shm.close()
        self.raw_rbuf.close()

        if self.main:
            self.status_shm.unlink()
            self.disp_rbuf.shm.unlink()
            self.raw_rbuf.unlink()

        
