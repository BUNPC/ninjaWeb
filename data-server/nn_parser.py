# import numpy as np
from tokenize import endpats

import nn_shared_memory
import array
import numpy as np
import struct
import parameter_parser
import math_utils as mu
import matplotlib.cm as cm
import time
import os
import datetime
import glob
from scipy.io import loadmat, savemat

def parserLoop(queue):

    acq_params = parameter_parser.ParameterParser()
    ml_length = len(np.where(acq_params.meas_list[:, 3] == 1)[0])
    n_srcs = acq_params.probe['SD']['nSrcs'][0][0][0][0]
    n_dets = acq_params.probe['SD']['nDets'][0][0][0][0]
    sm = nn_shared_memory.NNSharedMemory(main=False, parser=True, ml_length=ml_length, n_srcs=n_srcs, n_dets=n_dets)
    queue.put(acq_params)
    raw_bytes_packet = np.array([], dtype=np.uint8)
    have_packet_length = False
    tt = 0
    previous_time = time.time()

    # initialise variables
    opt_power_level = []
    dark_sig = []
    src_module_groups = []
    src_power_low_high = []
    data_power = []
    while not sm.status_shm.buf[sm.STATUS_SHM_IDX['shutdown']]:
        if not sm.getStatus('power_calib'):
            # current_time = time.time()
            # elapsed_time = time.time() - previous_time
            # previous_time = current_time
            # print(f"Elapsed time before packet length: {elapsed_time:.4f} seconds")
            if sm.getStatus('raw_rbuf_wr_idx') != sm.getStatus('raw_rbuf_rd_idx'):

                if not have_packet_length:
                    packet_length = get_packet_length(sm)
                    have_packet_length = True
                # get raw data from n states
                # current_time = time.time()
                # elapsed_time = time.time() - previous_time
                # previous_time = current_time
                # print(f"Elapsed time: {elapsed_time:.4f} seconds")
                si = sm.status_shm.buf[sm.STATUS_SHM_IDX['raw_rbuf_rd_idx']]%sm.RAW_RBUF_SLOTS * sm.RAW_RBUF_SLOT_SIZE
                # se = si + sm.RAW_RBUF_SLOT_SIZE
                se = si + packet_length
                buf = array.array('B', sm.raw_rbuf.buf[si:se])
                sm.status_shm.buf[sm.STATUS_SHM_IDX['raw_rbuf_rd_idx']] = (sm.status_shm.buf[sm.STATUS_SHM_IDX['raw_rbuf_rd_idx']] + 1) % sm.RAW_RBUF_SLOTS
                raw_bytes = np.frombuffer(buf, dtype=np.uint8)
                # print('raw_bytes[1]=', raw_bytes[1])
                if raw_bytes[1] == 0 and len(raw_bytes_packet) != 0:
                    # parse the data
                    # current_time = time.time()
                    # elapsed_time = time.time()-previous_time
                    # previous_time = current_time
                    # print(f"Elapsed time: {elapsed_time:.4f} seconds")
                    n_states = len(np.where(acq_params.srcram[0, :, 31] == 0)[0])+1
                    dt = n_states / 800
                    tt = tt+dt
                    n_detb_active = struct.unpack("i", sm.SYS_STATUS.buf[:4])[0]
                    data_ml, data_dark, data_organized_by_state = bytes_to_timeseries(raw_bytes_packet, acq_params.srcram, acq_params.mapped_indices, acq_params.meas_list, n_detb_active, bool(sm.SYS_STATUS.buf[5]), bool(sm.SYS_STATUS.buf[4]), packet_length)

                    current_time = time.time()
                    elapsed_time = time.time()-previous_time
                    previous_time = current_time
                    print(f"Elapsed time after parsing data: {elapsed_time:.4f} seconds")
                    # example parse of aux
                    aux_val = float(sum([buf[5+ib]*256**ib for ib in range(3)]) * 3.3/(237)/4095)
                    si = sm.status_shm.buf[sm.STATUS_SHM_IDX['disp_rbuf_wr_idx']]
                    ml_idx = struct.unpack('H', sm.plot_ml_idx.buf[:2])[0]
                    sm.disp_rbuf[si] = float(data_ml[:,ml_idx][0])
                    # print(float(data_ml[:,ml_idx][0]))
                    if np.isnan(sm.disp_rbuf_time[si-1]):
                        sm.disp_rbuf_time[si] = 0
                    else:
                        sm.disp_rbuf_time[si] = sm.disp_rbuf_time[si-1]+dt
                    sm.status_shm.buf[sm.STATUS_SHM_IDX['disp_rbuf_wr_idx']] = (sm.status_shm.buf[sm.STATUS_SHM_IDX['disp_rbuf_wr_idx']] + 1) % sm.DISP_RBUF_SIZE
                    raw_bytes_packet = raw_bytes[0:packet_length]
                else:
                    raw_bytes_packet = np.append(raw_bytes_packet, raw_bytes[:packet_length])
                    # current_time = time.time()
                    # elapsed_time = time.time() - previous_time
                    # previous_time = current_time
                    # print(f"Elapsed time in else: {elapsed_time:.4f} seconds")
            else:
                time.sleep(0.0001)
                if not sm.status_shm.buf[sm.STATUS_SHM_IDX['run']]:
                    if sm.status_shm.buf[sm.STATUS_SHM_IDX['update_statemap_file']]:
                        statemap_folder = os.path.join('..', 'meas', datetime.datetime.now().strftime('%y-%m-%d'))
                        pattern = os.path.join(statemap_folder, '*_stateMap.mat')
                        state_map_files = glob.glob(pattern)

                        if state_map_files:
                            # Find the latest file
                            print('updating state map...........................')
                            latest_file = max(state_map_files, key=os.path.getmtime)
                            mat_data = loadmat(latest_file, struct_as_record=False, squeeze_me=True)
                            devInfo = mat_data['devInfo']
                            setattr(devInfo, 'dataLEDPowerCalibration', data_power)
                            setattr(devInfo, 'optPowerLevel', opt_power_level)
                            setattr(devInfo, 'srcPowerLowHigh', src_power_low_high)
                            setattr(devInfo, 'dSig', dark_sig)
                            setattr(devInfo, 'srcModuleGroups', src_module_groups)
                            mat_data['stateIndices'] = acq_params.mapped_indices+np.array([1,1,2])
                            mat_data['devInfo'] = devInfo
                            savemat(latest_file, mat_data)
                            sm.status_shm.buf[sm.STATUS_SHM_IDX['update_statemap_file']] = False

        else:
            pow_range = 7
            shared_arr = np.ndarray((7, 1024, 32), dtype=np.uint16, buffer=sm.srcram.buf)
            shared_arr_ml_sig = np.ndarray(ml_length, dtype=np.int16, buffer=sm.ml_sig_values.buf)
            shared_arr_n_poor_srcs = np.ndarray(n_srcs, dtype=np.int16, buffer=sm.n_poor_srcs.buf)
            shared_arr_n_poor_dets = np.ndarray(n_dets, dtype=np.int16, buffer=sm.n_poor_dets.buf)

            if not have_packet_length:
                packet_length = get_packet_length(sm)
                have_packet_length = True
            for iPower  in range(pow_range+1):
                print('running for level ', iPower)
                sm.power_calib_level.buf[:2] = struct.pack('H', iPower)
                acq_params.srcram = acq_params.create_srcram(acq_params.meas_list, iPower)
                acq_params.mapped_indices = acq_params.map_to_measurementlist(acq_params.srcram, acq_params.meas_list)
                np.copyto(shared_arr, acq_params.srcram)
                n_states = len(np.where(acq_params.srcram[0, :, 31] == 0)[0])+1
                print('running for level before start running', iPower)
                sm.status_shm.buf[sm.STATUS_SHM_IDX['run']] = True
                raw_bytes_temp = get_n_frames(sm, acq_params, packet_length, n_states, 1)
                raw_bytes_packet = get_n_frames(sm, acq_params, packet_length, n_states, 4)
                print('raw_bytes_packet', raw_bytes_packet[0], raw_bytes_packet[1])
                sm.status_shm.buf[sm.STATUS_SHM_IDX['run']] = False
                n_detb_active = struct.unpack("i", sm.SYS_STATUS.buf[:4])[0]
                data_ml, data_dark_temp, data_organized_by_state = bytes_to_timeseries(raw_bytes_packet, acq_params.srcram,
                                                                                  acq_params.mapped_indices,
                                                                                  acq_params.meas_list, n_detb_active,
                                                                                  bool(sm.SYS_STATUS.buf[5]),
                                                                                  bool(sm.SYS_STATUS.buf[4]),
                                                                                  packet_length)
                data_ml = np.transpose(data_ml)
                if iPower > 0:
                    data_power[:, iPower - 1] = np.mean(data_ml, axis=1, where=~np.isnan(data_ml))
                    data_states_power[:, :, iPower - 1] = np.mean(data_organized_by_state, axis=0, where=~np.isnan(data_organized_by_state))
                else:
                    data_dark = np.nanmean(data_dark_temp, axis=0)
                    data_sates_dark = data_organized_by_state
                    data_states_power = np.zeros((data_organized_by_state.shape[1], data_organized_by_state.shape[2], pow_range))
                    data_power = np.zeros((data_dark_temp.shape[1], pow_range))

            print('Data power...')
            print(data_power.shape)
            print(data_power)
            print('plotting results now................')
            thresholds = [-45.7625, -2.2407]
            threshHigh = 10 ** (thresholds[1] / 20)
            threshLow = 10 ** (thresholds[0] / 20)
            n_srcs = acq_params.probe['SD']['nSrcs'][0][0][0][0]
            n_dets = acq_params.probe['SD']['nDets'][0][0][0][0]
            print('n_srcs', n_srcs)
            print('n_dets', n_dets)
            srcram, opt_power_level, dark_sig, src_module_groups, src_power_low_high = power_calibration_dual_levels(acq_params.meas_list, n_srcs, data_power, thresholds, 1)

            time.sleep(5)
            acq_params.srcram = srcram
            np.copyto(shared_arr, acq_params.srcram)
            acq_params.mapped_indices = acq_params.map_to_measurementlist(acq_params.srcram,acq_params.meas_list, src_power_low_high)
            # if power_calib_level == 8, srcram will be uploaded
            sm.power_calib_level.buf[:2] = struct.pack('H', 8)

            lst1 = np.where(acq_params.meas_list[:, 3] == 1)[0]
            lst2 = np.where(acq_params.meas_list[:, 3] == 2)[0]

            sm.status_shm.buf[sm.STATUS_SHM_IDX['power_calib']] = False
            sm.status_shm.buf[sm.STATUS_SHM_IDX['sig_level_tuning']] = True
            print('running signal level adjustment')
            sm.status_shm.buf[sm.STATUS_SHM_IDX['run']] = True
            ml_sig_values = np.ones(len(lst1), dtype=np.int16)
            jet_colormap = cm.get_cmap('jet', 80)
            cm_lookup_table = (jet_colormap(np.linspace(0, 1, 80))[:, :3] * 255).astype(np.uint8)
            n_states = len(np.where(acq_params.srcram[0, :, 31] == 0)[0])+1
            print('n_states', n_states)
            time.sleep(0.5)
            while True:
                if sm.getStatus('raw_rbuf_wr_idx') != sm.getStatus('raw_rbuf_rd_idx'):
                    print('inside signal level adjustment')
                if not sm.status_shm.buf[sm.STATUS_SHM_IDX['run']]:
                    break
                si = sm.status_shm.buf[sm.STATUS_SHM_IDX['raw_rbuf_rd_idx']] % sm.RAW_RBUF_SLOTS * sm.RAW_RBUF_SLOT_SIZE
                se = si + sm.RAW_RBUF_SLOT_SIZE
                sm.status_shm.buf[sm.STATUS_SHM_IDX['raw_rbuf_rd_idx']] = (sm.status_shm.buf[sm.STATUS_SHM_IDX[
                    'raw_rbuf_rd_idx']] + 1) % sm.RAW_RBUF_SLOTS
                raw_bytes_packet_temp = get_n_frames(sm, acq_params, packet_length, n_states, 1)
                raw_bytes_packet = get_n_frames(sm, acq_params, packet_length,n_states, 2)
                data_ml, data_dark_temp, data_organized_by_state = bytes_to_timeseries(raw_bytes_packet, acq_params.srcram,
                                                                                       acq_params.mapped_indices,
                                                                                       acq_params.meas_list, n_detb_active,
                                                                                       bool(sm.SYS_STATUS.buf[5]),
                                                                                       bool(sm.SYS_STATUS.buf[4]),
                                                                                       packet_length)

                data_sig = np.nanmean(data_ml, axis=0)

                # Plot saturated and poor SNR channels
                lsth = np.where((data_sig[lst1] > threshHigh) | (data_sig[lst2] > threshHigh))[0]
                lstl = np.where((data_sig[lst1] < threshLow) | (data_sig[lst2] < threshLow))[0]

                lsth1 = np.where(data_sig[lst1] > threshHigh)[0]
                lstl1 = np.where(data_sig[lst1] < threshLow)[0]

                lsth2 = np.where(data_sig[lst2] > threshHigh)[0]
                lstl2 = np.where(data_sig[lst2] < threshLow)[0]

                ml_idx_sat = lst1[lsth]
                ml_sig_values = np.ones(len(lst1), dtype=np.int16)
                ml_sig_values[ml_idx_sat] = 2
                n_poor_srcs = np.zeros(n_srcs, dtype=np.int16)
                n_poor_dets = np.zeros(n_dets, dtype=np.int16)

                lst_tmp1 = lst1[lstl]
                lst_tmp2 = lst2[lstl]

                for i_ml in range(len(lstl)):
                    n_poor_srcs[int(acq_params.meas_list[lst_tmp1[i_ml],0])-1] += 1
                    n_poor_dets[int(acq_params.meas_list[lst_tmp1[i_ml],1])-1] += 1
                    i_col = int(np.ceil((20 * np.log10(max(min(min(data_sig[lst_tmp1[i_ml]], data_sig[lst_tmp2[i_ml]]), 1e-2), 1e-4)) + 81) * cm_lookup_table.shape[0] / 41))
                    ml_sig_values[lstl[i_ml]] = -i_col
                # print('ml_sig_values = ', ml_sig_values)

                np.copyto(shared_arr_ml_sig, ml_sig_values)
                np.copyto(shared_arr_n_poor_srcs, n_poor_srcs)
                np.copyto(shared_arr_n_poor_dets, n_poor_dets)
                sm.status_shm.buf[sm.STATUS_SHM_IDX['disp_rbuf_wr_idx']] = (sm.status_shm.buf[sm.STATUS_SHM_IDX[
                    'disp_rbuf_wr_idx']] + 1) % sm.DISP_RBUF_SIZE
                time.sleep(0.0001)

    # clean up when shutting down
    sm.close()
    return

def get_packet_length(sm):
    header_indicator = np.array([254])  # byte indicating the header
    detector_header_indicator = np.array([253, 252])  # bytes indicating the detector header
    state_number_length = 2
    sample_counter_length = 1
    n_det_per_board = 8
    n_bytes_per_det = 3
    n_det_boards = struct.unpack("i", sm.SYS_STATUS.buf[:4])[0]
    n_detectors = n_det_per_board * n_det_boards

    if bool(sm.SYS_STATUS.buf[5]):
        acc_bytes = 18
    else:
        acc_bytes = 0

    if bool(sm.SYS_STATUS.buf[4]):
        aux_bytes = 8
    else:
        aux_bytes = 0

    # Estimate package length
    det_bytes_per_board = n_det_per_board * n_bytes_per_det + len(
        detector_header_indicator) + sample_counter_length + 1
    payload_size = n_det_boards * det_bytes_per_board
    offset = len(header_indicator) + state_number_length + aux_bytes  # offset for first payload byte
    packet_length = offset + payload_size + acc_bytes
    return packet_length

def get_n_frames(sm, acq_params, packet_length, n_states, n):
    # raw_bytes_packet = np.array([], dtype=np.uint8)
    raw_bytes_packet = np.zeros(packet_length * n_states, dtype=np.uint8)
    print('raw_bytes_packet_shape',raw_bytes_packet.shape)
    frame_counter = 0
    print('beginning get n_frames')
    previous_time = time.time()
    while True:
        if sm.getStatus('raw_rbuf_wr_idx') != sm.getStatus('raw_rbuf_rd_idx'):
            # get raw data from n states
            si = sm.status_shm.buf[sm.STATUS_SHM_IDX['raw_rbuf_rd_idx']] % sm.RAW_RBUF_SLOTS * sm.RAW_RBUF_SLOT_SIZE
            se = si + sm.RAW_RBUF_SLOT_SIZE
            buf = array.array('B', sm.raw_rbuf.buf[si:se])
            sm.status_shm.buf[sm.STATUS_SHM_IDX['raw_rbuf_rd_idx']] = (sm.status_shm.buf[sm.STATUS_SHM_IDX[
                'raw_rbuf_rd_idx']] + 1) % sm.RAW_RBUF_SLOTS
            raw_bytes = np.frombuffer(buf, dtype=np.uint8)

            if raw_bytes[1] == n_states-1 and np.any(raw_bytes_packet != 0): #len(raw_bytes_packet) != 0:
                raw_bytes_packet[raw_bytes[1]*packet_length:(raw_bytes[1]+1)*packet_length] = raw_bytes[:packet_length]
                frame_counter += 1
                if frame_counter >= n:
                    break
            else:
                # raw_bytes_packet = np.append(raw_bytes_packet, raw_bytes[:packet_length])
                print('current byte index and length', raw_bytes[1]*packet_length, (raw_bytes[1]+1)*packet_length)
                raw_bytes_packet[raw_bytes[1]*packet_length:(raw_bytes[1]+1)*packet_length] = raw_bytes[:packet_length]
                previous_time, elapsed_time = get_lapsed_time(previous_time)
                print(f"Elapsed time in get n frames: {elapsed_time:.4f} seconds")
                print('raw bytes[1] and n_states:', raw_bytes[1], n_states)
        else:
            time.sleep(0.0001)
        if not sm.status_shm.buf[sm.STATUS_SHM_IDX['run']]:
            break

    print('end get n_frames')
    return raw_bytes_packet

def power_calibration_dual_levels(ml, nSrcs, dataLEDPowerCalibration, thresholds, flagSpatialMultiplex):
    rhoSDS = ml[:, 4]

    # rhoSDS = np.zeros((ml.shape[0],))
    # for iML in range(ml.shape[0]):
    #     iS = int(ml[iML, 0]) - 1
    #     iD = int(ml[iML, 1]) - 1
    #     rhoSDS[iML] = np.sqrt(np.sum((SD.SrcPos3D[iS, :] - SD.DetPos3D[iD, :]) ** 2))
    # if flagSpatialMultiplex and nSrcs == 56:
    #     srcModuleGroups = [[1, 3, 5], [2, 4, 6], [7]]
    #
    # else:
    #     srcModuleGroups = [[1], [2], [3], [4], [5], [6], [7]]
    #     srcModuleGroups = srcModuleGroups[0:int(np.ceil((nSrcs - 0.1) / 8))]

    srcModuleGroups = [[1], [2], [3], [4], [5], [6], [7]]
    srcModuleGroups_length = len(srcModuleGroups)

    lstSMS = [[{} for _ in range(2)] for _ in range(8)]

    for iSg in range(srcModuleGroups_length):
        for iWav in range(2):
            for iSrc in range(8):
                lstSMS[iSrc][iWav][iSg] = []
                for iMod in srcModuleGroups[iSg]:
                    lst = np.where(
                        (ml[:, 0] == ((iMod - 1) * 8 + iSrc + 1)) & (ml[:, 3] == iWav + 1) & (rhoSDS >= 0) & (
                                rhoSDS <= 45))[0]
                    lstSMS[iSrc][iWav][iSg] = np.append(lstSMS[iSrc][iWav][iSg], lst)

    threshHigh = 10 ** (thresholds[1] / 20)
    threshLow = 10 ** (thresholds[0] / 20)

    optPowerLevel = np.zeros((8, 2, 2, srcModuleGroups_length))

    for iSg in range(srcModuleGroups_length):
        numDetGood12 = np.zeros((8, 7, 7, 2))
        numDetGood2 = np.zeros((8, 7, 7, 2))
        for iWav in range(2):
            for iSrc in range(8):
                for iPow1 in range(6):
                    for iPow2 in range(iPow1 + 1, 7):
                        lst = lstSMS[iSrc][iWav][iSg]
                        lst = lst.astype(int)
                        lstGood12 = np.where((dataLEDPowerCalibration[lst, iPow1] > threshLow) &
                                             (dataLEDPowerCalibration[lst, iPow1] < threshHigh) |
                                             (dataLEDPowerCalibration[lst, iPow2] > threshLow) &
                                             (dataLEDPowerCalibration[lst, iPow2] < threshHigh))[0]
                        numDetGood12[iSrc, iPow1, iPow2, iWav] = len(lstGood12)
                        lstGood2 = np.where((dataLEDPowerCalibration[lst, iPow2] > threshLow) &
                                            (dataLEDPowerCalibration[lst, iPow2] < threshHigh))[0]
                        numDetGood2[iSrc, iPow1, iPow2, iWav] = len(lstGood2)

                ic, ir = np.where(numDetGood12[iSrc, :, :, iWav].T == np.max(numDetGood12[iSrc, :, :, iWav]))
                ic2, ir2 = np.where((numDetGood12[iSrc, :, :, iWav].T == np.max(numDetGood12[iSrc, :, :, iWav])) &
                                    (numDetGood2[iSrc, :, :, iWav].T > 0))

                if True:
                    if True:  # not ir2.any():
                        optPowerLevel[iSrc, 0, iWav, iSg] = ir[0]  # [-1]
                        optPowerLevel[iSrc, 1, iWav, iSg] = ic[0]  # [-1]
                    else:
                        # require 1 or more channels to use the high power
                        optPowerLevel[iSrc, 0, iWav, iSg] = ir2[-1]
                        optPowerLevel[iSrc, 1, iWav, iSg] = ic2[-1]
                else:
                    optPowerLevel[iSrc, 0, iWav, iSg] = 7
                    optPowerLevel[iSrc, 1, iWav, iSg] = 7

    # Initialize dSig and srcPowerLowHigh
    dSig = np.zeros(len(ml))
    nDets = max(ml[:, 1])
    srcPowerLowHigh = np.zeros((8 * 7, int(nDets), 2))
    maxPower = np.round(np.logspace(3, np.log10(2 ** 16 - 1), 7)).astype(int)

    for iML in range(len(ml)):
        iS = int((ml[iML, 0] - 1) % 8)
        iD = int(ml[iML, 1] - 1)
        iW = int(ml[iML, 3] - 1)

        iSrcModule = np.ceil(ml[iML, 0] / 8)
        # iSg = next((ii for ii, src in enumerate(srcModuleGroups) if (iSrcModule + 1) in src), -1)

        iSg = 0
        ii = 0

        while ii < len(srcModuleGroups):
            ii += 1
            if sum([1 for item in srcModuleGroups[ii - 1] if iSrcModule == item]) > 0:
                iSg = ii - 1
                break

        iPL1 = int(optPowerLevel[iS, 0, iW, iSg])
        iPL2 = int(optPowerLevel[iS, 1, iW, iSg])

        if (threshLow < dataLEDPowerCalibration[iML, iPL2] < threshHigh):
            srcPowerLowHigh[int(ml[iML, 0]) - 1, iD, iW] = 2
            dSig[iML] = dataLEDPowerCalibration[iML, iPL2]
        else:
            srcPowerLowHigh[int(ml[iML, 0]) - 1, iD, iW] = 1
            dSig[iML] = dataLEDPowerCalibration[iML, iPL1]

    # Create the statemap
    srcram = np.zeros((7, 1024, 32), dtype=np.uint8)
    srcram[:, :, 20] = 1
    srcram[:, :,
    30] = 1  # using this bit as a hackfor identifying source 0 when we look at 5 bits in mapToMeasurementList()

    lstS = np.unique(ml[:, 0])
    nSrcModules = 7
    iState = 0

    srcModuleGroups = [[1, 2, 3, 4, 5, 6, 7]]
    srcModuleGroups_length = len(srcModuleGroups)
    for iSg in range(srcModuleGroups_length):
        for iS in range(1, 9):
            lstSMG = srcModuleGroups[iSg]

            for iSrcMod in range(len(lstSMG)):
                srcram[lstSMG[iSrcMod] - 1, iState, 0:16] = mu.bitget(
                    maxPower[int(optPowerLevel[iS - 1, 0, 0, lstSMG[iSrcMod] - 1])], range(0, 16))  # Set the power
                srcram[lstSMG[iSrcMod] - 1, iState, 16:20] = mu.bitget(((iS - 1) * 2), range(0,
                                                                                             4))  # Select the source for wavelength 1
                srcram[lstSMG[iSrcMod] - 1, iState, 20] = 0
                srcram[lstSMG[iSrcMod] - 1, iState, 30] = 0

            iState += 1
            for iSrcMod in range(len(lstSMG)):
                srcram[lstSMG[iSrcMod] - 1, iState, 0:16] = mu.bitget(
                    maxPower[int(optPowerLevel[iS - 1, 0, 1, lstSMG[iSrcMod] - 1])], range(0, 16))  # Set the power
                srcram[lstSMG[iSrcMod] - 1, iState, 16:20] = mu.bitget(((iS - 1) * 2 + 1), range(0, 4))
                srcram[lstSMG[iSrcMod] - 1, iState, 20] = 0
                srcram[lstSMG[iSrcMod] - 1, iState, 30] = 0

            iState += 1

    iState += 1

    if flagSpatialMultiplex and nSrcs == 56:
        srcModuleGroups = [[1, 3, 5], [2, 4, 6], [7]]
    else:
        srcModuleGroups = [[1], [2], [3], [4], [5], [6], [7]]
        srcModuleGroups = srcModuleGroups[0:int(np.ceil((nSrcs - 0.1) / 8))]

    srcModuleGroups_length = len(srcModuleGroups)

    for iSg in range(srcModuleGroups_length):
        for iS in range(1, 9):  # Equivalent to 1:8 in MATLAB
            lstSMG = srcModuleGroups[iSg]

            for iSrcMod in range(len(lstSMG)):
                srcram[lstSMG[iSrcMod] - 1, iState, 0:16] = mu.bitget(
                    maxPower[int(optPowerLevel[iS - 1, 1, 0, lstSMG[iSrcMod] - 1])], range(0, 16))  # Set the power
                srcram[lstSMG[iSrcMod] - 1, iState, 16:20] = mu.bitget(((iS - 1) * 2), range(0,
                                                                                             4))  # Select the source for wavelength 1
                srcram[lstSMG[iSrcMod] - 1, iState, 20] = 0
                srcram[lstSMG[iSrcMod] - 1, iState, 30] = 0

            iState += 1
            for iSrcMod in range(len(lstSMG)):
                srcram[lstSMG[iSrcMod] - 1, iState, 0:16] = mu.bitget(
                    maxPower[int(optPowerLevel[iS - 1, 1, 1, lstSMG[iSrcMod] - 1])], range(0, 16))  # Set the power
                srcram[lstSMG[iSrcMod] - 1, iState, 16:20] = mu.bitget(((iS - 1) * 2 + 1), range(0, 4))
                srcram[lstSMG[iSrcMod] - 1, iState, 20] = 0
                srcram[lstSMG[iSrcMod] - 1, iState, 30] = 0

            iState += 2
    srcram[:, iState - 1:, 31] = 1  # mark sequence end

    print('src ram states after dual power calibration:.....', iState)

    return srcram, optPowerLevel, dSig, srcModuleGroups, srcPowerLowHigh



def bytes_to_timeseries(raw_bytes, srcram, mapped_indices, meas_list, n_det_boards,acc_active, aux_active, packet_length):
    """

    """
    # raw_bytes = np.frombuffer(raw_bytes, dtype=np.uint8)

    # header_indicator = np.array([254])  # byte indicating the header
    # detector_header_indicator = np.array([253, 252])  # bytes indicating the detector header
    # state_number_length = 2
    # sample_counter_length = 1
    n_det_per_board = 8
    n_bytes_per_det = 3
    # # n_det_boards = struct.unpack("i", sm.SYS_STATUS.buf[:4])[0]
    # n_detectors = n_det_per_board * n_det_boards
    #
    # if bool(acc_active):
    #     acc_bytes = 18
    # else:
    #     acc_bytes = 0
    #
    # if bool(aux_active):
    #     aux_bytes = 8
    # else:
    #     aux_bytes = 0
    #
    # # Estimate package length
    # det_bytes_per_board = n_det_per_board * n_bytes_per_det + len(detector_header_indicator) + sample_counter_length + 1
    # payload_size = n_det_boards * det_bytes_per_board
    # offset = len(header_indicator) + state_number_length + aux_bytes  # offset for first payload byte
    # packet_length = offset + payload_size + acc_bytes
    num_packets = len(raw_bytes) // packet_length
    reshaped_data = raw_bytes[:num_packets * packet_length].reshape(num_packets, packet_length).T

    det_indices = [14 + i * 28 + j * 3 for i in range(n_det_boards) for j in range(n_det_per_board)]

    det_data = reshaped_data[det_indices, :]
    det_data = det_data.astype(np.float64)
    for ii in range(1, n_bytes_per_det):
        det_indices = [(14 + ii) + i * 28 + j * 3 for i in range(n_det_boards) for j in range(n_det_per_board)]
        det_data = det_data + reshaped_data[det_indices, :].astype(np.float64) * pow(256, ii)

    det_data = (det_data > pow(2, 23 - 1)) * pow(2, 24) - det_data

    n_states = len(np.where(srcram[0,:,31]==0)[0])+1
    n_frames = int(np.ceil(det_data.shape[1] / n_states))
    data_organized_by_state = np.full((n_states * n_frames, n_det_per_board * n_det_boards), np.nan)
    # indices = list(range(int(raw_bytes[1]), int(raw_bytes[1]) + det_data.shape[1]))
    indices = list(range(0,  det_data.shape[1])) ## please verify this
    data_organized_by_state[indices, :] = det_data.T
    data_organized_by_state = np.reshape(data_organized_by_state, (n_states, n_frames, n_det_per_board * n_det_boards), order='F')
    data_organized_by_state = np.transpose(data_organized_by_state, (1, 2, 0))
    data_organized_by_state = np.roll(data_organized_by_state, shift=-1, axis=2)
    data_organized_by_state = data_organized_by_state / (237 * (pow(2, 15) - 1))
    data_ml = np.array([[np.nan] * meas_list.shape[0]] * data_organized_by_state.shape[0])
    data_dark = np.array([[np.nan] * meas_list.shape[0]] * data_organized_by_state.shape[0])
    for ki in range(0, meas_list.shape[0]):
        data_ml[:, ki] = data_organized_by_state[:, mapped_indices[ki, 1], mapped_indices[ki, 0]]
        data_dark[:, ki] = data_organized_by_state[:, mapped_indices[ki, 1], mapped_indices[ki, 2]]

    data_ml[data_ml < 0] = 1e-6

    return data_ml, data_dark, data_organized_by_state

def get_lapsed_time(previous_time):
    current_time = time.time()
    elapsed_time = time.time() - previous_time
    return current_time, elapsed_time








