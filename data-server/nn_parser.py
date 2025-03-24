# import numpy as np
from tokenize import endpats

import nn_shared_memory
import time
import array
import numpy as np
import struct
import parameter_parser
import math_utils as mu

def parserLoop(queue):
    sm = nn_shared_memory.NNSharedMemory(main=False)
    acq_params = parameter_parser.ParameterParser()
    queue.put(acq_params)
    raw_bytes_packet = np.array([], dtype=np.uint8)
    have_packet_length = False


    while not sm.status_shm.buf[sm.STATUS_SHM_IDX['shutdown']]:
        if not sm.getStatus('power_calib'):
            if sm.getStatus('raw_rbuf_wr_idx') != sm.getStatus('raw_rbuf_rd_idx'):
                if not have_packet_length:
                    packet_length = get_packet_length(sm)
                    have_packet_length = True
                # get raw data from n states
                si = sm.status_shm.buf[sm.STATUS_SHM_IDX['raw_rbuf_rd_idx']]%sm.RAW_RBUF_SLOTS * sm.RAW_RBUF_SLOT_SIZE
                se = si + sm.RAW_RBUF_SLOT_SIZE
                buf = array.array('B', sm.raw_rbuf.buf[si:se])
                sm.status_shm.buf[sm.STATUS_SHM_IDX['raw_rbuf_rd_idx']] = (sm.status_shm.buf[sm.STATUS_SHM_IDX['raw_rbuf_rd_idx']] + 1) % sm.RAW_RBUF_SLOTS
                raw_bytes = np.frombuffer(buf, dtype=np.uint8)
                if raw_bytes[1] == 0 and len(raw_bytes_packet) != 0:
                    # parse the data
                    n_states = len(np.where(acq_params.srcram[0, :, 31] == 0)[0])
                    dt = n_states / 800
                    n_detb_active = struct.unpack("i", sm.SYS_STATUS.buf[:4])[0]
                    data_ml, data_dark, data_organized_by_state = bytes_to_timeseries(raw_bytes_packet, acq_params.srcram, acq_params.mapped_indices, acq_params.meas_list, n_detb_active, bool(sm.SYS_STATUS.buf[5]), bool(sm.SYS_STATUS.buf[4]), packet_length)

                    # example parse of aux
                    aux_val = float(sum([buf[5+ib]*256**ib for ib in range(3)]) * 3.3/(237)/4095)
                    si = sm.status_shm.buf[sm.STATUS_SHM_IDX['disp_rbuf_wr_idx']]
                    ml_idx = struct.unpack('H', sm.plot_ml_idx.buf[:2])[0]
                    sm.disp_rbuf[si] = float(data_ml[:,ml_idx][0])
                    if np.isnan(sm.disp_rbuf_time[si-1]):
                        sm.disp_rbuf_time[si] = 0
                    else:
                        sm.disp_rbuf_time[si] = sm.disp_rbuf_time[si-1]+dt
                    sm.status_shm.buf[sm.STATUS_SHM_IDX['disp_rbuf_wr_idx']] = (sm.status_shm.buf[sm.STATUS_SHM_IDX['disp_rbuf_wr_idx']] + 1) % sm.DISP_RBUF_SIZE
                    raw_bytes_packet = raw_bytes[0:packet_length]
                else:
                    raw_bytes_packet = np.append(raw_bytes_packet, raw_bytes[:packet_length])
            else:
                time.sleep(0.001)
        else:
            pow_range = 7
            shared_arr = np.ndarray((7, 1024, 32), dtype=np.uint16, buffer=sm.srcram.buf)
            if not have_packet_length:
                packet_length = get_packet_length(sm)
                have_packet_length = True
            for iPower  in range(pow_range+1):
                print('running for level ', iPower)
                sm.power_calib_level.buf[:2] = struct.pack('H', iPower)
                acq_params.srcram = acq_params.create_srcram(acq_params.meas_list, iPower)
                np.copyto(shared_arr, acq_params.srcram)
                sm.status_shm.buf[sm.STATUS_SHM_IDX['run']] = True
                raw_bytes_packet = get_n_frames(sm, acq_params, packet_length, 4)
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

            print('plotting results now................')
            thresholds = [-45.7625, -1.9]
            srcram, opt_power_level, dark_sig, src_module_groups, src_power_low_high = power_calibration_dual_levels(acq_params.meas_list, acq_params.probe['SD']['nSrcs'][0][0][0][0], data_power, thresholds, 0)

            acq_params.srcram = srcram
            np.copyto(shared_arr, acq_params.srcram)
            acq_params.mapped_indices = acq_params.map_to_measurementlist(acq_params.srcram,acq_params.meas_list, src_power_low_high)
            # if power_calib_level == 8, srcram will be uploaded
            sm.power_calib_level.buf[:2] = struct.pack('H', 8)

            lst1 = np.where(acq_params.meas_list[:, 3] == 1)[0]
            lst2 = np.where(acq_params.meas_list[:, 3] == 2)[0]

            sm.status_shm.buf[sm.STATUS_SHM_IDX['power_calib']] = False
            print('running signal level adjustment')
            sm.status_shm.buf[sm.STATUS_SHM_IDX['run']] = True
            while True:
                if not sm.status_shm.buf[sm.STATUS_SHM_IDX['run']]:
                    break
                raw_bytes_packet = get_n_frames(sm, acq_params, packet_length, 2)
                data_ml, data_dark_temp, data_organized_by_state = bytes_to_timeseries(raw_bytes_packet, acq_params.srcram,
                                                                                       acq_params.mapped_indices,
                                                                                       acq_params.meas_list, n_detb_active,
                                                                                       bool(sm.SYS_STATUS.buf[5]),
                                                                                       bool(sm.SYS_STATUS.buf[4]),
                                                                                       packet_length)

                data_sig = np.nanmean(data_ml, axis=0)

                # Plot saturated and poor SNR channels
                lsth = np.where((data_sig[lst1 - 1] > thresholds[1]) | (data_sig[lst2 - 1] > thresholds[1]))[0]
                lstl = np.where((data_sig[lst1 - 1] < thresholds[0]) | (data_sig[lst2 - 1] < thresholds[0]))[0]

                lsth1 = np.where(data_sig[lst1 - 1] > thresholds[1])[0]
                lstl1 = np.where(data_sig[lst1 - 1] < thresholds[0])[0]

                lsth2 = np.where(data_sig[lst2 - 1] > thresholds[1])[0]
                lstl2 = np.where(data_sig[lst2 - 1] < thresholds[0])[0]

                print('higher than threshold = ', len(lsth))
                print('lower than threshold = ', len(lsth))

                time.sleep(0.002)

        time.sleep(0.02)

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

def get_n_frames(sm, acq_params, packet_length, n):
    raw_bytes_packet = np.array([], dtype=np.uint8)
    frame_counter = 0
    while True:
        if sm.getStatus('raw_rbuf_wr_idx') != sm.getStatus('raw_rbuf_rd_idx'):
            # get raw data from n states
            si = sm.status_shm.buf[sm.STATUS_SHM_IDX['raw_rbuf_rd_idx']] % sm.RAW_RBUF_SLOTS * sm.RAW_RBUF_SLOT_SIZE
            se = si + sm.RAW_RBUF_SLOT_SIZE
            buf = array.array('B', sm.raw_rbuf.buf[si:se])
            sm.status_shm.buf[sm.STATUS_SHM_IDX['raw_rbuf_rd_idx']] = (sm.status_shm.buf[sm.STATUS_SHM_IDX[
                'raw_rbuf_rd_idx']] + 1) % sm.RAW_RBUF_SLOTS
            raw_bytes = np.frombuffer(buf, dtype=np.uint8)
            if raw_bytes[1] == 0 and len(raw_bytes_packet) != 0:
                frame_counter += 1
                if frame_counter >= n:
                    break
            else:
                raw_bytes_packet = np.append(raw_bytes_packet, raw_bytes[:packet_length])
        else:
            time.sleep(0.001)
        if not sm.status_shm.buf[sm.STATUS_SHM_IDX['run']]:
            break

    return raw_bytes_packet

def power_calibration_dual_levels(ml, nSrcs, dataLEDPowerCalibration, thresholds, flagSpatialMultiplex):
    rhoSDS = ml[:, 4]
    # rhoSDS = np.zeros((ml.shape[0],))
    # for iML in range(ml.shape[0]):
    #     iS = int(ml[iML, 0]) - 1
    #     iD = int(ml[iML, 1]) - 1
    #     rhoSDS[iML] = np.sqrt(np.sum((SD.SrcPos3D[iS, :] - SD.DetPos3D[iD, :]) ** 2))
    if flagSpatialMultiplex and nSrcs == 56:
        srcModuleGroups = [[1, 3, 5], [2, 4, 6], [7]]

    else:
        srcModuleGroups = [[1], [2], [3], [4], [5], [6], [7]]
        srcModuleGroups = srcModuleGroups[0:int(np.ceil((nSrcs - 0.1) / 8))]

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

                ir, ic = np.where(numDetGood12[iSrc, :, :, iWav] == np.max(numDetGood12[iSrc, :, :, iWav]))
                ir2, ic2 = np.where((numDetGood12[iSrc, :, :, iWav] == np.max(numDetGood12[iSrc, :, :, iWav])) &
                                    (numDetGood2[iSrc, :, :, iWav] > 0))

                if True:
                    if not ir2.any():
                        optPowerLevel[iSrc, 0, iWav, iSg] = ir[-1]
                        optPowerLevel[iSrc, 1, iWav, iSg] = ic[-1]
                    else:
                        # require 1 or more channels to use the high power
                        optPowerLevel[iSrc, 0, iWav, iSg] = ir2[-1]
                        optPowerLevel[iSrc, 1, iWav, iSg] = ic2[-1]
                else:
                    optPowerLevel[iSrc, 0, iWav, iSg] = 7
                    optPowerLevel[iSrc, 1, iWav, iSg] = 7

    # Initialize dSig and srcPowerLowHigh
    dSig = np.zeros(len(ml))
    srcPowerLowHigh = np.zeros((8 * 7, len(ml), 2))
    maxPower = np.round(np.logspace(2, np.log10(2 ** 16 - 1), 7)).astype(int)

    for iML in range(len(ml)):
        iS = int((ml[iML, 0] - 1) % 8)
        iD = int(ml[iML, 1] - 1)
        iW = int(ml[iML, 3] - 1)

        iSrcModule = (ml[iML, 0] - 1) // 8
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

    for iSg in range(srcModuleGroups_length):
        for iS in range(1, 9):  # Equivalent to 1:8 in MATLAB
            lstSMG = srcModuleGroups[iSg]

            for iSrcMod in range(len(lstSMG)):
                srcram[lstSMG[iSrcMod] - 1, iState, 0:16] = mu.bitget(
                    maxPower[int(optPowerLevel[iS - 1, 0, 0, iSg])], range(0, 16))  # Set the power
                srcram[lstSMG[iSrcMod] - 1, iState, 16:20] = mu.bitget(((iS - 1) * 2), range(0,
                                                                                                  4))  # Select the source for wavelength 1
                srcram[lstSMG[iSrcMod] - 1, iState, 20] = 0
                srcram[lstSMG[iSrcMod] - 1, iState, 30] = 0

            iState += 1
            for iSrcMod in range(len(lstSMG)):
                srcram[lstSMG[iSrcMod] - 1, iState, 0:16] = mu.bitget(
                    maxPower[int(optPowerLevel[iS - 1, 0, 1, iSg])], range(0, 16))  # Set the power
                srcram[lstSMG[iSrcMod] - 1, iState, 16:20] = mu.bitget(((iS - 1) * 2 + 1), range(0, 4))
                srcram[lstSMG[iSrcMod] - 1, iState, 20] = 0
                srcram[lstSMG[iSrcMod] - 1, iState, 30] = 0

            iState += 2

    iState += 1

    for iSg in range(srcModuleGroups_length):
        for iS in range(1, 9):  # Equivalent to 1:8 in MATLAB
            lstSMG = srcModuleGroups[iSg]

            for iSrcMod in range(len(lstSMG)):
                srcram[lstSMG[iSrcMod] - 1, iState, 0:16] = mu.bitget(
                    maxPower[int(optPowerLevel[iS - 1, 1, 0, iSg])], range(0, 16))  # Set the power
                srcram[lstSMG[iSrcMod] - 1, iState, 16:20] = mu.bitget(((iS - 1) * 2), range(0,
                                                                                                  4))  # Select the source for wavelength 1
                srcram[lstSMG[iSrcMod] - 1, iState, 20] = 0
                srcram[lstSMG[iSrcMod] - 1, iState, 30] = 0

            iState += 1
            for iSrcMod in range(len(lstSMG)):
                srcram[lstSMG[iSrcMod] - 1, iState, 0:16] = mu.bitget(
                    maxPower[int(optPowerLevel[iS - 1, 1, 1, iSg])], range(0, 16))  # Set the power
                srcram[lstSMG[iSrcMod] - 1, iState, 16:20] = mu.bitget(((iS - 1) * 2 + 1), range(0, 4))
                srcram[lstSMG[iSrcMod] - 1, iState, 20] = 0
                srcram[lstSMG[iSrcMod] - 1, iState, 30] = 0

            iState += 2
    srcram[:, iState - 1:, 31] = 1  # mark sequence end

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

    n_states = len(np.where(srcram[0,:,31]==0)[0])
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








