# NinjaNIRS 2024 - Raspberry Pi Zero SPI byte stream recorder
#
# bzim@bu.edu
# Initial version 2023-3-29
# Modified to listen to digital in and start a new file 2024-10-16
import json

import NN24SystemClass
import nn_shared_memory
import time
import numpy as np
import nn_utils
import struct
import scipy.io
import parameter_parser

# ====  SPI recording loop  ====
def recorderLoop(acq_params):

    print("\n-----  NinjaNIRS 2024 byte stream recorder  -----\n")
    print("Press Ctrl-C to exit.\n")
    # acq_params = parameter_parser.ParameterParser()
    nn = NN24SystemClass.NNSystem(acq_params.srcram)
    nn.flush()
    nn_gpios = nn_utils.NN_GPIOS()
    sm = nn_shared_memory.NNSharedMemory(main=False)
    loop_stats = nn_utils.DataLoopStats()

    N_STATES_PER_TRANSACT = 1
    N_BYTES_PER_TRANSACT = nn.N_BYTES_TO_READ_PER_STATE * N_STATES_PER_TRANSACT
    if N_BYTES_PER_TRANSACT > sm.RAW_RBUF_SLOT_SIZE:
        print(f"Error: too large N_BYTES_PER_TRANSACT = f{N_BYTES_PER_TRANSACT}")
        # raise

    last_report_time = time.perf_counter()
    bytes_available = 0

    sys_running = False
    run_event_last = 0
    run_event_curr = 0
    nn.flush()
    sm.SYS_STATUS.buf[:4] = struct.pack("i", np.int32(nn.n_detb_active))
    sm.SYS_STATUS.buf[4] = int(nn.aux_active)
    sm.SYS_STATUS.buf[5] = int(nn.acc_active)
    previous_time = time.time()
    while not sm.status_shm.buf[sm.STATUS_SHM_IDX['shutdown']]:
        run_event_curr = sm.status_shm.buf[sm.STATUS_SHM_IDX['run']]

        if bytes_available < N_BYTES_PER_TRANSACT:
            buf = nn.spi.readbytes(2)
            bytes_available = buf[0] + buf[1]*256

        # only store data and keep statistics if the system is running
        if sys_running:
            loop_stats.n_cyc += 1
            loop_stats.max_bytes_available = max(bytes_available, loop_stats.max_bytes_available)


            if bytes_available >= N_BYTES_PER_TRANSACT:
                buf = nn.spi.readbytes(2 + N_BYTES_PER_TRANSACT)
                bytes_available = buf[0] + buf[1]*256 - N_BYTES_PER_TRANSACT
                ishift = 0
                # check that we have the correct start
                while (buf[2+ishift]!=254 or buf[5+ishift]!=250) and (ishift+3)<N_BYTES_PER_TRANSACT:
                    ishift += 1
                loop_stats.ishift_max = max(ishift, loop_stats.ishift_max)
                if ishift > 0: # there was an error
                    buf = nn.spi.readbytes(2 + ishift) # flush bytes to realign
                    bytes_available = buf[0] + buf[1]*256 - ishift
                else:
                    fbin.write(bytearray(buf[2:]))
                    si = sm.status_shm.buf[sm.STATUS_SHM_IDX['raw_rbuf_wr_idx']]%sm.RAW_RBUF_SLOTS * sm.RAW_RBUF_SLOT_SIZE
                    se = si + N_BYTES_PER_TRANSACT
                    sm.raw_rbuf.buf[si:se] = bytearray(buf[2:])
                    sm.status_shm.buf[sm.STATUS_SHM_IDX['raw_rbuf_wr_idx']] = (sm.status_shm.buf[sm.STATUS_SHM_IDX['raw_rbuf_wr_idx']] + 1) % sm.RAW_RBUF_SLOTS
                    # current_time = time.time()
                    # elapsed_time = time.time() - previous_time
                    # previous_time = current_time
                    # print(f"Elapsed time after idx update: {elapsed_time:.4f} seconds")
                    # if not data_server.data_queue.full():
                    #     data_server.data_queue.put(aux_val)
                    # else:
                    #     loop_stats.n_queue_skip += 1
                    loop_stats.n_bytes_rxd += N_BYTES_PER_TRANSACT
            else: # waiting for more data
                loop_stats.n_cyc_wait += 1
                time.sleep(0.0001)

            if time.perf_counter() - last_report_time > 2:
                loop_stats.n_bytes_rxd_total += loop_stats.n_bytes_rxd
                # data_queue.put(aux_val)
                print(str(loop_stats))
                flog.write(str(loop_stats) + "\n")
                last_report_time = time.perf_counter()
                loop_stats.reset_loop()

        else: # system not running
            if time.perf_counter() - last_report_time > 4:
                print(f"Idle {loop_stats.n_idle_reports:5d}")
                last_report_time = time.perf_counter()
                loop_stats.n_idle_reports += 1
            if bytes_available > 0:
                # there should be no bytes when sys is not running
                nn.spi.readbytes(2 + bytes_available) # flush
            calib_level = struct.unpack('H', sm.power_calib_level.buf[:2])[0]
            if calib_level == 8:
                shared_arr_loaded = np.ndarray((7, 1024, 32), dtype=np.uint16, buffer=sm.srcram.buf)
                acq_params.srcram = np.copy(shared_arr_loaded)
                nn.updateSrcRAM(acq_params.srcram)
                sm.power_calib_level.buf[:2] = struct.pack('H', 0)
                n_states = len(np.where(acq_params.srcram[0, :, 31] == 0)[0])
                print('n_states after power calibration', n_states)
                print('srcram updated')
            time.sleep(0.02)

        # system just started running
        if run_event_last==0 and run_event_curr==1:
            # Open binary and logging files
            if not sm.status_shm.buf[sm.STATUS_SHM_IDX['power_calib']]:
                fname_bin, fname_log, filename_json = nn_utils.getFileNames()
                print(f"Opening {fname_bin}")
                fbin = open(fname_bin, "wb")
                flog = open(fname_log, "w")
                json_sidecar = get_json_struct()
                json_sidecar = update_json_sidecar(json_sidecar, acq_params, nn)
                # with open(filename_json, 'w') as f:
                #     json.dump(json_sidecar, f, indent=4)
                scipy.io.savemat(filename_json, json_sidecar)
                sm.status_shm.buf[sm.STATUS_SHM_IDX['update_statemap_file']] = True
            else:
                calib_level = struct.unpack('H', sm.power_calib_level.buf[:2])[0]
                if calib_level == 0:
                    fname_bin, fname_log, filename_json = nn_utils.getFileNames(is_calib=True, calib_level=calib_level)
                else:
                    fname_bin = fname_bin.replace(f'LEDPowerCalibration_{calib_level-1:02d}',
                                                  f'LEDPowerCalibration_{calib_level:02d}')
                    fname_log = fname_log.replace(f'LEDPowerCalibration_{calib_level-1:02d}',
                                                  f'LEDPowerCalibration_{calib_level:02d}')
                print(f"Opening {fname_bin}")
                fbin = open(fname_bin, "wb")
                flog = open(fname_log, "w")
                # update srcram
                shared_arr_loaded = np.ndarray((7, 1024, 32), dtype=np.uint16, buffer=sm.srcram.buf)
                acq_params.srcram = np.copy(shared_arr_loaded)
                print('power calib - before updating source ram')
                nn.updateSrcRAM(acq_params.srcram, True)
                print('srcram updated')
                nn.flush() # test to see if that fixes the issue
            nn.startAcq()
            nn_gpios.led_set('green', True)
            last_report_time = time.perf_counter()
            sys_running = True

        # system just stopped running
        if run_event_last==1 and run_event_curr==0:
            print('System just stopped running-2')
            nn.stopAcq()
            fbin.close()
            flog.close()

            #nn_gpios.buzz.start(50)
            nn_gpios.led_set('green', False)
            loop_stats.reset_all()

            sys_running = False

            time.sleep(0.1)
            #nn_gpios.buzz.stop()

        run_event_last = run_event_curr

    # clean up when shutting down
    if 'fbin' in locals():
        fbin.close()
        print("Closing {}".format(fname_bin))
    if 'flog' in locals():
        flog.close()
    sm.close()
    nn.spi.close()
    nn_gpios.cleanup()
    print("\n{:,d} bytes received total.".format(loop_stats.n_bytes_rxd_total))
    print("Exiting.\n")

def get_json_struct():
    devInfo = {
        "flagSpatialMultiplex": np.float64(1),
        "state_fs": np.float64(800),
        "acc_active": 0,
        "aux_active": 0,
        "N_DETECTOR_BOARDS": 0,
        "stat": {"n_smp": 237.0, "accfs":0.0, "gyrofs":0.0},
        "dataLEDPowerCalibration": [],
        "optPowerLevel": [],
        "srcPowerLowHigh": [],
        "dSig": [],
        "srcModuleGroups": []
    }
    json_sidecar = {
        "devInfo": devInfo,
        "nSD": [],
        "stateIndices": [],
        "stateMap": []
    }
    return json_sidecar

def update_json_sidecar(json_sidecar, acq_params, nn):
    probe_data = convert_to_float64(acq_params.probe['SD'])
    json_sidecar['nSD'] = probe_data
    json_sidecar['stateMap'] = acq_params.srcram.astype(np.float64)
    json_sidecar['devInfo']['acc_active'] = np.float64(nn.acc_active)
    if nn.acc_active:
        json_sidecar['devInfo']['stat']['accfs'] = np.float64(nn.accfs)
        json_sidecar['devInfo']['stat']['gyrofs'] = np.float64(nn.gyrofs)
    json_sidecar['devInfo']['aux_active'] = np.float64(nn.aux_active)
    json_sidecar['devInfo']['N_DETECTOR_BOARDS'] = np.float64(nn.n_detb_active)
    json_sidecar['devInfo']['stat']['n_smp'] = np.float64(nn.n_smp)
    return json_sidecar

def convert_to_float64(data):
    """
    Recursively traverses a data structure (dict, list, tuple, ndarray)
    and converts any numerical NumPy arrays found within it to float64 dtype.
    Handles nested structures, structured arrays, and object arrays common
    in loaded .mat files.

    Args:
        data: The data structure (potentially nested) to traverse.

    Returns:
        The modified data structure with numerical NumPy arrays converted.
    """
    # Case 0: Handle None explicitly if it appears (e.g., from empty MATLAB cells)
    if data is None:
        return None
    # Case 1: Dictionary
    elif isinstance(data, dict):
        new_dict = {}
        for key, value in data.items():
            # Skip internal keys added by loadmat
            if key.startswith('__') and key.endswith('__'):
                new_dict[key] = value # Keep header/version info
            else:
                new_dict[key] = convert_to_float64(value)
        return new_dict
    # Case 2: List or Tuple
    elif isinstance(data, (list, tuple)):
        # Apply conversion to each item in the sequence
        new_list = [convert_to_float64(item) for item in data]
        # Return the same type (list or tuple)
        return type(data)(new_list)
    # Case 3: NumPy Array
    elif isinstance(data, np.ndarray):
        # Case 3a: Structured Array (dtype has named fields)
        if data.dtype.names:
            # Create a copy with the same structured dtype to modify safely
            converted_array = np.copy(data)
            for name in data.dtype.names:
                try:
                    # Recursively convert the data associated with this field name
                    # Accessing data[name] gives an array of all values for that field
                    original_field_data = data[name]
                    converted_field_data = convert_to_float64(original_field_data)
                    # Assign the potentially converted data back to the field in the copy
                    converted_array[name] = converted_field_data
                except Exception as e:
                    # Catch potential errors during conversion or assignment
                    # Keep original data in the copy for this field if conversion fails
                    converted_array[name] = data[name] # Or handle differently if needed
            return converted_array

        # Case 3b: Simple Numerical Array (and not structured)
        elif np.issubdtype(data.dtype, np.number):
            # Convert to float64 if it's numerical and not already float64
            if data.dtype != np.float64:
                return data.astype(np.float64)
            else:
                # Already float64, return as is
                return data

        # Case 3c: Array of Objects (dtype is 'O') - often contains nested data
        elif data.dtype == np.object_:
             converter_func = np.frompyfunc(convert_to_float64, 1, 1)
             return converter_func(data)

        # Case 3d: Other non-numerical array types (e.g., strings, bools)
        else:
            # No conversion needed for non-numerical, non-structured, non-object arrays
            return data
    # Case 4: Other data types (scalars, strings, etc.)
    else:
        # Return scalars, strings, etc., as is
        return data

