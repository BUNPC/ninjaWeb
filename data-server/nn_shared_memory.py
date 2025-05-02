import sys
from multiprocessing import shared_memory
import parameter_parser
import struct
import numpy as np

class NNSharedMemory:
    # Status shared memory
    STATUS_SHM_NAME = "nn_status_shm"
    STATUS_SHM_FIELDS_INIT = [('run', False), \
                              ('disp_rbuf_rd_idx', 0), \
                              ('disp_rbuf_wr_idx', 0), \
                              ('raw_rbuf_rd_idx', 0), \
                              ('raw_rbuf_wr_idx', 0), \
                              ('power_calib', False), \
                              ('sig_level_tuning', False),\
                              ('update_statemap_file', False), \
                              ('shutdown', False) ]
    STATUS_SHM_IDX = dict([ (key[0], idx) for idx, key in enumerate(STATUS_SHM_FIELDS_INIT) ])
    STATUS_SHM_SIZE = len(STATUS_SHM_FIELDS_INIT)

    # Display ring buffer
    DISP_RBUF_NAME = "nn_disp_rbuf"
    DISP_RBUF_SIZE = 2**8
    DISP_RBUF_INIT = [float('nan')] * DISP_RBUF_SIZE

    DISP_RBUF_TIME_NAME = "nn_disp_rbuf_time"
    DISP_RBUF_TIME_SIZE = 2**8
    DISP_RBUF_TIME_INIT = [float('nan')] * DISP_RBUF_TIME_SIZE

    # Raw data ring buffer
    # one state packet per slot, this avoids packets wrapping across the end of the buffer
    RAW_RBUF_NAME = "nn_raw_rbuf"
    RAW_RBUF_SLOTS = 50
    RAW_RBUF_SLOT_SIZE = 1024
    RAW_RBUF_SIZE = RAW_RBUF_SLOTS * RAW_RBUF_SLOT_SIZE

    # store probe, srcram and mapped indices
    SYS_STATUS_NAME = "nn_sys_status_shm"
    SYS_STATUS_SIZE = 6

    # srcram
    srcram_shape = (7, 1024, 32)
    dtype = np.uint16
    srcram_nbytes = np.prod(srcram_shape) * np.dtype(dtype).itemsize

    # measurement list signal values for calibration signal adjustment display
    # ml_sig_values_shape =


    def __init__(self, main = False, parser=False, ml_length=0, n_srcs=0, n_dets=0):
        self.main = main

        if main:
            self.status_shm = shared_memory.SharedMemory(name=self.STATUS_SHM_NAME, create=True, size=self.STATUS_SHM_SIZE)
            self.status_shm.buf[:] = bytearray([x[1] for x in self.STATUS_SHM_FIELDS_INIT]) # initialize status shared memory
            self.disp_rbuf = shared_memory.ShareableList(name=self.DISP_RBUF_NAME, sequence=self.DISP_RBUF_INIT)
            self.disp_rbuf_time = shared_memory.ShareableList(name=self.DISP_RBUF_TIME_NAME, sequence=self.DISP_RBUF_TIME_INIT)
            self.raw_rbuf = shared_memory.SharedMemory(name=self.RAW_RBUF_NAME, create=True, size=self.RAW_RBUF_SIZE)
            self.SYS_STATUS = shared_memory.SharedMemory(name=self.SYS_STATUS_NAME, create=True, size=self.SYS_STATUS_SIZE)
            self.plot_ml_idx = shared_memory.SharedMemory(name='plot_ml_idx', create=True, size=2)
            self.plot_ml_idx.buf[:2] = struct.pack('H', 0)
            self.plot_wavelength = shared_memory.SharedMemory(name='plot_wavelength', create=True, size=2)
            self.plot_wavelength.buf[:2] = struct.pack('H', 1)
            self.power_calib_level = shared_memory.SharedMemory(name='power_calib_level', create=True, size=2)
            self.srcram = shared_memory.SharedMemory(name='srcram', create=True, size=self.srcram_nbytes)
            # acq_params = parameter_parser.ParameterParser()

            # # get memory size for acq_parameters
            # acq_params_size = (sys.getsizeof(acq_params.srcram) + sys.getsizeof(self.mapped_indices) +
            #                    sys.getsizeof(acq_params.meas_list))
            # self.acq_params = shared_memory.SharedMemory(name='acq_params', create=True, size=acq_params_size)
            # self.acq_params = acq_params

        else:
            self.status_shm = shared_memory.SharedMemory(name=self.STATUS_SHM_NAME, create=False)
            self.disp_rbuf = shared_memory.ShareableList(name=self.DISP_RBUF_NAME)
            self.disp_rbuf_time = shared_memory.ShareableList(name=self.DISP_RBUF_TIME_NAME)
            self.raw_rbuf = shared_memory.SharedMemory(name=self.RAW_RBUF_NAME, create=False)
            self.SYS_STATUS = shared_memory.SharedMemory(name=self.SYS_STATUS_NAME, create=False)
            self.plot_ml_idx = shared_memory.SharedMemory(name='plot_ml_idx', create=False)
            self.plot_wavelength = shared_memory.SharedMemory(name='plot_wavelength', create=False)
            self.power_calib_level = shared_memory.SharedMemory(name='power_calib_level', create=False)
            self.srcram = shared_memory.SharedMemory(name='srcram', create=False)
            if parser:
                ml_sig_dtype = np.int16
                ml_sig_values_size = ml_length*np.dtype(ml_sig_dtype).itemsize
                n_poor_srcs_size = n_srcs*np.dtype(ml_sig_dtype).itemsize
                n_poor_dets_size = n_dets * np.dtype(ml_sig_dtype).itemsize
                self.ml_sig_values = shared_memory.SharedMemory(name='ml_sig_values', create=True, size=ml_sig_values_size)
                self.n_poor_srcs = shared_memory.SharedMemory(name='n_poor_srcs', create=True,
                                                                size=n_poor_srcs_size)
                self.n_poor_dets = shared_memory.SharedMemory(name='n_poor_dets', create=True,
                                                                size=n_poor_dets_size)
            else:
                self.ml_sig_values = shared_memory.SharedMemory(name='ml_sig_values', create=False)
                self.n_poor_srcs = shared_memory.SharedMemory(name='n_poor_srcs', create=False)
                self.n_poor_dets = shared_memory.SharedMemory(name='n_poor_dets', create=False)
            # self.acq_params = parameter_parser.ParameterParser()

    def getStatus(self, key='run'):
        return self.status_shm.buf[self.STATUS_SHM_IDX[key]]

    def close(self):
        self.status_shm.close()
        self.disp_rbuf.shm.close()
        self.disp_rbuf_time.shm.close()
        self.raw_rbuf.close()
        self.SYS_STATUS.close()
        self.plot_ml_idx.close()
        self.plot_wavelength.close()
        self.power_calib_level.close()
        self.srcram.close()
        self.ml_sig_values.close()
        self.n_poor_srcs.close()
        self.n_poor_dets.close()

        if self.main:
            self.status_shm.unlink()
            self.disp_rbuf.shm.unlink()
            self.disp_rbuf_time.shm.unlink()
            self.raw_rbuf.unlink()
            self.SYS_STATUS.unlink()
            self.plot_ml_idx.unlink()
            self.plot_wavelength.unlink()
            self.power_calib_level.unlink()
            self.srcram.unlink()
            self.ml_sig_values.unlink()
            self.n_poor_srcs.unlink()
            self.n_poor_dets.unlink()
        
