# ---  NinjaNIRS  ---
# System class for NinjaNIRS.
#
# Initial version: 2023-6-2
# Bernhard Zimmermann - bzim@bu.edu
# Boston University Neurophotonics Center
#

import time
import configparser
from time import sleep

import numpy as np
from math import ceil, floor
import spidev
import pickle

# ----------------------------------------------------------------------
def bitfield(num, nbits=8):
    return [num >> i & 1 for i in range(nbits)]


# ----------------------------------------------------------------------
def genSingleSourceRAMB(isrc, comm_to_brd=False):
    # Generate RAM B contents to collect data only from a single data source.
    # comm_to_brd: If True, set RAM B in such a way to allow pass through communication
    # to plug in cards (src or det).

    ramb = np.zeros([1024, 32], dtype=np.uint8)

    # set source selector bits (UART Rx Mux)
    ramb[:, 0:6] = bitfield(isrc, 6)
    # set flow enable bit (DetB Rx En)
    ramb[1:, 6] = 1

    if comm_to_brd:
        # set flow enable bit also for initial state
        ramb[0, 6] = 1
    else:
        # write end cycle bits  (so that det card loads data into UART)
        ramb[10:500, 9] = 1
    return ramb


##################################################################
class NNSystem():

    # ----------------------------------------------------------------------
    def __init__(self, srcram=None):

        self.initConfig()

        self.initSPI()

        self.initStatReg()
        
        self.powerOn()

        self.initActiveBrds()

        # create RAM B and upload
        self.createRAMB()
        self.uploadToRAM('b')

        # create SRCRAM and RAM A and upload them
        self.updateSrcRAM(srcram)

        self.flush()


    # ----------------------------------------------------------------------
    def initConfig(self):
        # System constants
        self.N_DET_PER_BOARD = 8
        self.N_BYTES_PER_DET = 3
        self.N_BYTES_TO_READ_PER_DETB = self.N_DET_PER_BOARD * self.N_BYTES_PER_DET + 4
        self.N_BYTES_TO_READ_PER_ACCELEROMETER = 3+7*2+1
        self.N_DET_SLOTS = 24
        self.N_SRC_SLOTS = 7
        self.N_BYTES_TO_READ_PER_SRCB_RAM_LOC = 7
        self.BRD_RELAY_OFFSET = 48

        self.n_states_a = 0
        self.n_status_states = 0

        self.rama = np.zeros([1024, 32], dtype=np.uint8)
        self.ramb = np.zeros([1024, 32], dtype=np.uint8)
        self.srcram = np.zeros([7, 1024, 32], dtype=np.uint8)

        configurations = configparser.ConfigParser()
        configFilename = 'NN24config.cfg'
        configurations.read(configFilename)     # TODO: Use 'try' or 'with'?

        self.config = configurations['DEFAULT']
        self.n_states_b = 0
        self.n_smp = 0
        self.acc_active = 0
        self.detb_active = [0]*self.N_DET_SLOTS
        self.imub_active = [0]*self.N_DET_SLOTS
        self.srcb_active = [0]*self.N_SRC_SLOTS
        self.n_detb_active = 0
        self.n_imub_active = 0
        self.n_srcb_active = 0
        self.aux_active = self.config.getboolean('aux_active')


    # ----------------------------------------------------------------------
    def initSPI(self):
        # Initialize SPI bus
        self.spi = spidev.SpiDev()
        self.spi.open(self.config.getint('spi_bus'), self.config.getint('spi_dev')) #(bus, device)
        self.spi.max_speed_hz = self.config.getint('spi_max_speed_hz')
        self.spi.mode = self.config.getint('spi_mode')
        self.rxtx_buf = np.zeros(8196, dtype=np.uint8)


    # ----------------------------------------------------------------------
    def createRAMB(self):
        self.ramb = np.zeros([1024, 32], dtype=np.uint8)

        # number of used RAM B states
        self.n_states_b = self.config.getint('n_states_b')
        # duration of end cycle pulse period
        t_end_cyc = self.config.getfloat('t_end_cyc')
        # holdoff between different selected detector board uart transmissions
        t_bsel_holdoff = self.config.getfloat('t_bsel_holdoff')
        # minimum duration of each board select period
        # (32bytes*10bits/6000000baud = 53.4e-6 s)
        t_bsel_min = self.config.getfloat('t_bsel_min')
        # time to hold off sampling after state switch to let analog signal settle
        # at 350 us the signal should be approx 101% of final value
        t_smp_holdoff_start = self.config.getfloat('t_smp_holdoff_start')
        # time to hold off sampling before state switch (at end of cycle)
        t_smp_holdoff_end = self.config.getfloat('t_smp_holdoff_end')
        # duration of the source step signal at the end of a cycle
        # should be less than t_smp_holdoff_end
        t_src_step = self.config.getfloat('t_src_step')
        # minimum period for each sample 
        # (depends on ADC, ADC internal oversampling, transfer to MCU, computation in MCU)
        t_smp_min = self.config.getfloat('t_smp_min')
        # target number of samples to be averaged for each A state
        # values >256 risk overflow of the result (ADC: 16bit, result: 24bit)
        n_smp_target = self.config.getint('n_smp_target')

        # duration of each RAM B state
        t_state_b = self.clk_div*8/96e6
        # duration of each RAM A state
        self.t_state_a = t_state_b * self.n_states_b
        # target ADC sampling period
        t_smp_target = max((self.t_state_a -t_smp_holdoff_start - t_smp_holdoff_end)/n_smp_target, t_smp_min)
        # number of B states for each ADC sample
        ni_smp = max(ceil(t_smp_target/t_state_b), 2)
        # number of B states the trig signal is '1' for each ADC sample
        ni_smp_on = floor(ni_smp/2)

        # write end cycle bits
        self.ramb[1:(2+ceil(t_end_cyc/t_state_b)), 9] = 1
    
        # write ADC sampling bits
        for ii in range(ni_smp_on):
            start_idx = round(t_smp_holdoff_start/t_state_b) + ii - 1
            end_idx = self.n_states_b - round(t_smp_holdoff_end/t_state_b)  + ii
            # TODO: check that end_idx + ni_smp_off < n_state_b
            self.ramb[start_idx:end_idx:ni_smp, 10] = 1
        
        # number of samples collected during each A state
        # (first sample will be discarded by detector board)
        self.n_smp = sum(self.ramb[start_idx:end_idx:ni_smp, 10]) -1
        if self.n_smp > 255: # 256*16bit number could overflow 24 bit result
            print("Warning: nsamples large, overflow possible.")

        # write source step bits at the end of the cycle that will switch to the
        # next LED
        end_idx = self.n_states_b
        start_idx = end_idx - floor(t_src_step/t_state_b) -1
        self.ramb[start_idx:end_idx, 11] = 1

        # Data transmission management
        # number of B states allocated to each UART data source (det boards + IMU)
        ni_bsel = floor((self.n_states_b - round(t_end_cyc/t_state_b) - 2 -
                        (self.n_detb_active+self.n_imub_active+1+1)*ceil(t_bsel_holdoff/t_state_b) )
                         / (self.n_detb_active+self.n_imub_active+1))
        if ni_bsel*t_state_b < t_bsel_min:
            print("Warning: t_bsel too short.")

        start_idx = round(t_end_cyc/t_state_b) + ceil(t_bsel_holdoff/t_state_b)
        self.ramb[start_idx-2, 8] = 1                       # transmit program counter A
        if self.aux_active:
            self.ramb[start_idx-1, 7] = 1                   # transmit aux data
        for (ii, x) in enumerate(self.detb_active + self.imub_active + [self.acc_active]):
            if x:                                           # detector is active
                # set source selector bits (UART Rx Mux)
                for jj in range(ni_bsel):
                    self.ramb[start_idx+jj, 0:6] = bitfield(ii, 6)
                # set flow enable bit (DetB Rx En)
                for jj in range(ni_bsel-1):
                    self.ramb[start_idx+jj+1, 6] = 1
                start_idx += ni_bsel + ceil(t_bsel_holdoff/t_state_b)
        # mark sequence end
        self.ramb[(self.n_states_b-1):, 17] = 1


    # ----------------------------------------------------------------------
    def powerOn(self):
        # turn power on to subsystems sequentially
        # repeated update calls will create natural delay
        self.vn22clk_en = True
        self.v5p1rpi_off = False
        self.rst_pico = False
        self.updateStatReg()
        self.v5p1b23_en = True
        self.updateStatReg()
        self.v5p1b01_en = True
        self.updateStatReg()
        self.vn3p4_en = True
        self.updateStatReg()
        self.v9p0_en = True
        self.updateStatReg()
        self.vn22_en = True
        self.updateStatReg()
        self.v5p1src_en = True
        self.updateStatReg()

        # releasing the reset on the detector cards will in turn power on the detector optodes
        # turning on the power one backplane at a time to reduce the spike in current draw
        for ii in range(4):
            self.rst_detb[ii] = 0
            self.updateStatReg()
            time.sleep(0.25)

        self.rst_srcb = False
        self.updateStatReg()
        time.sleep(0.1)

        # this section does not need a delay
        self.rst_pca = False
        self.run = False
        self.updateStatReg()


    # ----------------------------------------------------------------------
    def initStatReg(self):
        # Create default status register values
        self.rpi0_rec_en = False
        self.v_src_boost = False # increase voltage for high power LED
        self.vn22clk_en = False
        self.vn3p4_en = False
        self.vn22_en = False
        self.v9p0_en = False
        self.v5p1src_en = False
        self.v5p1rpi_off = False
        self.v5p1b23_en = False
        self.v5p1b01_en = False

        self.clk_div = self.config.getint('clk_div')

        # Reset program counter A
        self.rst_pca = True
        # Reset all RP2040 microcontrollers
        self.rst_detb = [1] *4
        # reset rp2040 on all source boards (held in reset while true)
        self.rst_srcb = True
        # reset rp2040 on main control board (held in reset while true)
        self.rst_pico = True
        # reset external RAM used for data buffer
        self.rst_ram = True
        # reset transmit FIFO in external PSRAM - SPI firmware only?
        self.rst_tx_fifo = False
        # clk_div will not advance if false -> also a and b not advancing
        self.run = False
        self.updateStatReg()

        # turn off reset to allow communication FPGA->FTDI->PC
        self.rst_ram = False
        self.updateStatReg()
    

    # ----------------------------------------------------------------------
    def initActiveBrds(self):
        # detecting active boards takes some time, so first try to load from file
        act_sd_boards_fname = 'act_sd_boards.pkl'
        try:
            f_act_sd = open(act_sd_boards_fname, 'rb')
            self.detb_active, self.imub_active, self.srcb_active, self.acc_active = pickle.load(f_act_sd)
            self.updateBytesPerState()
            print(f"Loaded {act_sd_boards_fname}")
            print(f"Assuming {self.n_srcb_active} source, {self.n_imub_active} IMU, and {self.n_detb_active} detector cards. Control board IMU active: {self.acc_active}")
        except:
            self.updateActiveBrds()
            with open(act_sd_boards_fname, 'wb') as f_act_sd:
                pickle.dump([self.detb_active, self.imub_active, self.srcb_active, self.acc_active], f_act_sd)
        # Accelerometer / IMU constants
        # these are defined in the IMU MCU firmware
        # TODO: read this from the MCU directly (e.g. implement in status packet)
        if self.acc_active or self.n_imub_active>0:
            self.accfs = self.config.getfloat('accfs')
            self.gyrofs = self.config.getfloat('gyrofs')


    # ----------------------------------------------------------------------
    def updateActiveBrds(self):
        # This method will detect which adapter cards are plugged in 
        # and are active by sequentially trying to read from them.
        # It will also try to detect whether the IMU MCU is active.

        N_BYTES_TO_READ_PER_SRCB_RAM_LOC = 7

        if not self.v5p1b01_en or not self.v5p1b23_en:
            print("Warning: Detectors not powered, cannot determine active detectors accurately\n")

        self.detb_active = [0] * self.N_DET_SLOTS
        self.imub_active = [0] * self.N_DET_SLOTS
        self.srcb_active = [0] * self.N_SRC_SLOTS
        self.acc_active = False
    
        # bring instrument into a known state
        self.rst_pca = False
        self.rst_detb = [0] *4
        self.rst_srcb = False
        self.rst_pico = False
        clk_div_old = self.clk_div
        self.clk_div = 20
        self.run = False
        self.updateStatReg(True)
        time.sleep(0.01)
        self.flush()
        time.sleep(0.05)
        self.flush()
        self.updateStatReg(False)

        rama_old = self.rama.copy()
        ramb_old = self.ramb.copy()

        self.rama = np.zeros([1024, 32], dtype=np.uint8)
        self.rama[:, 8] = 1 # set stop bit
        self.uploadToRAM('a')

        # Check for active/plugged in detector or IMU boards
        for isrc in range(self.N_DET_SLOTS):
            self.flush()
            self.ramb = genSingleSourceRAMB(isrc)
            self.uploadToRAM('b', True)
            self.run = True
            self.updateStatReg(True)
            n_bytes_rxd = self.readBytes(self.N_BYTES_TO_READ_PER_DETB, 0.07)
            self.run = False
            self.updateStatReg(True)
            #if isrc == 6:
            #    print(n_bytes_rxd)
            #    print(self.rxtx_buf[0:10])
            if n_bytes_rxd==self.N_BYTES_TO_READ_PER_DETB:
                if self.rxtx_buf[0]==253 and self.rxtx_buf[1]==252:    # check for detector packet header
                    self.detb_active[isrc] = True
                elif self.rxtx_buf[0]==253 and self.rxtx_buf[1]==251:    # check for IMU packet header
                    self.imub_active[isrc] = True
            self.flush()

        self.n_detb_active = sum(self.detb_active)
        self.n_imub_active = sum(self.imub_active)
        print("{:d} detector adapter cards detected.".format(self.n_detb_active))
        print("{:d} IMU adapter cards detected.".format(self.n_imub_active))

        # check IMU / accelerometer status
        self.ramb = genSingleSourceRAMB(self.N_DET_SLOTS)
        self.uploadToRAM('b', False)
        self.run = True
        self.updateStatReg(True)
        n_bytes_rxd = self.readBytes(self.N_BYTES_TO_READ_PER_ACCELEROMETER, 0.1)
        self.run = False
        self.updateStatReg(True)
        if n_bytes_rxd==self.N_BYTES_TO_READ_PER_ACCELEROMETER:
            if self.rxtx_buf[0]==249 and self.rxtx_buf[1]==248:    #check packet header
                # buf(4:5) would be the temperature reading, which is unlikely = 0
                if self.rxtx_buf[3]!=0 or self.rxtx_buf[4]!=0:
                    self.acc_active = True
        print(f"Control board IMU active: {self.acc_active}")
        self.flush()

        # Check for active/plugged in source boards
        for isrc in range(self.N_SRC_SLOTS):
            self.flush()
            self.ramb = genSingleSourceRAMB(isrc+32, True)
            self.uploadToRAM('b', True)
            cmdbuf = np.zeros(1+2+4, dtype=np.uint8)
            cmdbuf[0] = 255
            cmdbuf[2] = self.BRD_RELAY_OFFSET
            self.writeBytes(cmdbuf)
            n_bytes_rxd = self.readBytes(N_BYTES_TO_READ_PER_SRCB_RAM_LOC, 0.07)
            if n_bytes_rxd==N_BYTES_TO_READ_PER_SRCB_RAM_LOC:
                # check packet header
                if self.rxtx_buf[0]==255 and self.rxtx_buf[1]==0 and self.rxtx_buf[2]==self.BRD_RELAY_OFFSET:
                    self.srcb_active[isrc] = True
            self.flush()
        self.n_srcb_active = sum(self.srcb_active)
        print("{:d} source adapter cards detected.".format(self.n_srcb_active))

        # restore old config
        self.clk_div = clk_div_old
        self.updateStatReg(False)
        self.rama = rama_old
        self.uploadToRAM('a', False)
        self.ramb = ramb_old
        self.uploadToRAM('b', False)

        self.updateBytesPerState()


    # ----------------------------------------------------------------------
    def updateBytesPerState(self):
        self.n_detb_active = sum(self.detb_active)
        self.n_imub_active = sum(self.imub_active)
        self.n_srcb_active = sum(self.srcb_active)
        # calculate bytes per state
        #N_BYTES_TO_READ_PER_DETB_STATUS = 18
        N_BYTES_TO_READ_PER_HEADER = 3
        #N_BYTES_TO_READ_PER_ACCELEROMETER_STATUS = 20
        N_AUX_ADC = 2
        N_BYTES_PER_AUX = 3
        N_BYTES_TO_READ_PER_AUX = 2+N_BYTES_PER_AUX*N_AUX_ADC

        self.N_BYTES_TO_READ_PER_STATE = N_BYTES_TO_READ_PER_HEADER + self.N_BYTES_TO_READ_PER_DETB*(self.n_detb_active+self.n_imub_active) \
            + self.acc_active*self.N_BYTES_TO_READ_PER_ACCELEROMETER + self.aux_active*N_BYTES_TO_READ_PER_AUX

    
    # ----------------------------------------------------------------------
    def updateStatReg(self, skipreadback=True):
        sreg = [0] * 32

        sreg[27] = self.rpi0_rec_en
        sreg[26] = self.v_src_boost
        sreg[25] = self.vn22clk_en
        sreg[24] = self.vn3p4_en
        sreg[23] = self.vn22_en
        sreg[22] = self.v9p0_en
        sreg[21] = self.v5p1src_en
        sreg[20] = self.v5p1rpi_off
        sreg[19] = self.v5p1b23_en
        sreg[18] = self.v5p1b01_en
        sreg[10:18] = bitfield(self.clk_div-1)           #[(self.clk_div-1) >> ii & 1 for ii in range(8)]
        sreg[9] = self.rst_tx_fifo
        sreg[8] = self.rst_ram
        sreg[7] = self.rst_pca
        sreg[3:7] = self.rst_detb
        sreg[2] = self.rst_srcb
        sreg[1] = self.rst_pico
        sreg[0] = self.run

        addr_offset = 64

        valbytes = [0] * 4
        cmdbytes = [0] * 2

        for iv in range(4):
            valbytes[iv] = sum([sreg[ii+8*iv] << ii for ii in range(8)])

        cmdbytes[0] = 1
        cmdbytes[1] = 128 + addr_offset

        self.writeBytes([255] + cmdbytes + valbytes)

        if not skipreadback:
            self.flush()
            cmdbytes[1] = addr_offset
            self.writeBytes([255] + cmdbytes + [0]*4)
            nrb = self.readBytes(7)
            if nrb != 7:
                print('Error: StatReg readback not enough bytes received\n')
            if self.rxtx_buf[0] != 240:
                print('Error: StatReg readback header error\n')
            if not all(self.rxtx_buf[1:3] == cmdbytes):
                print("Error: StatReg readback command error\n")
            if not all(self.rxtx_buf[3:7] == valbytes):
                print("Error: StatReg readback data error\n")


    # ----------------------------------------------------------------------
    # def uploadToRAM(self, ram_select, skipreadback=False, brd_sel=0):
    #     # Uploads selected ram variable to FPGA
    #
    #     if ram_select == 'a':
    #         addr_offset = 16
    #         data_relay = False
    #         ram_rb_header = 240
    #         d = self.rama                       # create alias
    #     elif ram_select == 'b':
    #         addr_offset = 32
    #         data_relay = False
    #         ram_rb_header = 240
    #         d = self.ramb
    #     elif ram_select == 'src':
    #         addr_offset = self.BRD_RELAY_OFFSET
    #         data_relay = True
    #         ram_rb_header = 255
    #         ramb_old = self.ramb.copy()
    #         self.ramb = genSingleSourceRAMB(brd_sel+32, True)
    #         self.uploadToRAM('b', True)
    #         d = self.srcram[brd_sel, :, :]
    #     else:
    #         print("ram_select invalid\n")
    #         return
    #
    #     self.flush()
    #     # flush FSM in FPGA
    #     self.writeBytes([0]*8)
    #
    #     buf = np.zeros(1024*(1+2+4), dtype=np.uint8)
    #
    #     for irow in range(1024):
    #         offset = irow*(1+2+4)
    #         buf[offset] = 255                                       # command header
    #         buf[offset+1] = irow % 256                              # address low byte
    #         buf[offset+2] = 128 + addr_offset + (irow >> 8)         # address high byte
    #         # convert the 32 individual bits of each row to 4 bytes
    #         buf[offset+3:offset+3+4] = np.packbits(d[irow,:], bitorder='little')
    #
    #     # split writes into chunks of less than SPI bufsiz
    #     # (/sys/module/spidev/parameters/bufsiz ~= 4096 on this machine)
    #     # split needs to be at command boundary
    #     for icnk in range(8):
    #         self.writeBytes(buf[icnk*128*7:(icnk+1)*128*7])
    #         if data_relay: # slow down transmission since UART to boards is not so fast
    #             time.sleep(0.03)
    #
    #     if not skipreadback:
    #         err_cnt = 0
    #         time.sleep(0.1)
    #         self.flush()
    #
    #         rbcmd_buf = np.zeros(1024*(1+2+4), dtype=np.uint8)
    #         for irow in range(1024):
    #             offset = irow*(1+2+4)
    #             rbcmd_buf[offset] = 255                                 # command header
    #             rbcmd_buf[offset+1] = irow % 256                        # address low byte
    #             rbcmd_buf[offset+2] = addr_offset + (irow >> 8)         # address high byte
    #             rbcmd_buf[(offset+3):(offset+7)] = 0                    # (value bytes)
    #         for icnk in range(8):
    #             self.writeBytes(rbcmd_buf[icnk*128*7:(icnk+1)*128*7])
    #             if data_relay: # slow down transmission since UART to boards is not so fast
    #                 time.sleep(0.05)
    #
    #         nb = self.readBytes(1024*7)
    #         if nb != (1024*7):
    #             print('Error: RAM readback not enough bytes received\n')
    #         for irow in range(1024):
    #             offset = irow*(1+2+4)
    #             if self.rxtx_buf[offset] != ram_rb_header or \
    #                 self.rxtx_buf[offset+1] != buf[offset+1] or\
    #                 self.rxtx_buf[offset+2] != buf[offset+2]-128 or\
    #                 not all(self.rxtx_buf[offset+3:offset+7] == buf[offset+3:offset+7]):
    #                 err_cnt += 1
    #
    #         if err_cnt > 0:
    #             print('Error: {:d} RAM {:s} readback errors\n'.format(err_cnt, ram_select))
    #
    #     if data_relay:
    #         # restore RAM B
    #         self.ramb = ramb_old
    #         self.uploadToRAM('b', False)


    # ----------------------------------------------------------------------
    def startAcq(self):
        self.run = True
        self.updateStatReg()


    # ----------------------------------------------------------------------
    def stopAcq(self):
        self.run = False
        self.updateStatReg()
        self.flush()
        time.sleep(max(0.01, self.n_states_a*self.t_state_a)) # This should be the max length of one frame
        self.flush()


    # ----------------------------------------------------------------------
    def updateSrcRAM(self, srcram, skipreadback=False):
        if srcram is None:
            srcram = np.zeros([7, 1024, 32], dtype=np.uint8)
            srcram[:, :, [20, 31]] = 1  # disable LEDs and set all stop bits
        self.srcram = srcram
        # Generate matching RAM A
        iStatesOn = np.where(srcram[0, :, 31] == 0)[0]  # TODO: what happens if src card 0 not present
        self.rama = np.zeros([1024, 32], dtype=np.uint8)
        self.n_states_a = len(iStatesOn) + 1
        self.n_status_states = 0
        self.rama[self.n_states_a-1:, 8] = 1
        # set up RAM B for relaying data to src cards
        ramb_old = self.ramb.copy()
        # upload SRCRAM
        for isrcb in range(self.N_SRC_SLOTS):
            if self.srcb_active[isrcb]:
                self.ramb = genSingleSourceRAMB(isrcb + 32, True)
                self.uploadToRAM('b', True, 0, 1)
                self.uploadToRAM('src_raw', skipreadback, isrcb, self.n_states_a)
        # restore RAM B
        self.ramb = ramb_old
        self.uploadToRAM('b', False, 0, 1)
        # upload matching RAM A
        self.uploadToRAM('a', skipreadback, 0, self.n_states_a)
        self.frame_rate = 1 / (self.n_states_a * self.t_state_a)

    # ----------------------------------------------------------------------
    def uploadToRAM(self, ram_select, skipreadback=False, brd_sel=0, nrows=1024):
        # Uploads selected ram variable to FPGA

        if ram_select == 'a':
            addr_offset = 16
            data_relay = False
            ram_rb_header = 240
            d = self.rama  # create alias
        elif ram_select == 'b':
            addr_offset = 32
            data_relay = False
            ram_rb_header = 240
            d = self.ramb
        elif ram_select == 'src':
            addr_offset = self.BRD_RELAY_OFFSET
            data_relay = True
            ram_rb_header = 255
            ramb_old = self.ramb.copy()
            self.ramb = genSingleSourceRAMB(brd_sel + 32, True)
            self.uploadToRAM('b', True, 0, 1)
            d = self.srcram[brd_sel, :, :]
        elif ram_select == 'src_raw':  # for fast upload of all srcrams, will not change RAM B, needs to be done externally
            addr_offset = self.BRD_RELAY_OFFSET
            data_relay = True
            ram_rb_header = 255
            d = self.srcram[brd_sel, :, :]
        else:
            print("ram_select invalid\n")
            return

        self.flush()
        # flush FSM in FPGA
        self.writeBytes([0] * 8)

        buf = np.zeros(nrows * (1 + 2 + 4), dtype=np.uint8)

        for irow in range(nrows):
            offset = irow * (1 + 2 + 4)
            buf[offset] = 255  # command header
            buf[offset + 1] = irow % 256  # address low byte
            buf[offset + 2] = 128 + addr_offset + (irow >> 8)  # address high byte
            # convert the 32 individual bits of each row to 4 bytes
            buf[offset + 3:offset + 3 + 4] = np.packbits(d[irow, :], bitorder='little')

        # split writes into chunks of less than SPI bufsiz
        # (/sys/module/spidev/parameters/bufsiz ~= 4096 on this machine)
        # split needs to be at command boundary
        for icnk in range(ceil(nrows / 128)):
            self.writeBytes(buf[icnk * 128 * 7:min((icnk + 1) * 128 * 7, len(buf))])
            if data_relay:  # slow down transmission since UART to boards is not so fast
                time.sleep(0.0015)

        if not skipreadback:
            err_cnt = 0
            self.flush()

            rbcmd_buf = np.zeros(nrows * (1 + 2 + 4), dtype=np.uint8)
            for irow in range(nrows):
                offset = irow * (1 + 2 + 4)
                rbcmd_buf[offset] = 255  # command header
                rbcmd_buf[offset + 1] = irow % 256  # address low byte
                rbcmd_buf[offset + 2] = addr_offset + (irow >> 8)  # address high byte
                rbcmd_buf[(offset + 3):(offset + 7)] = 0  # (value bytes)
            for icnk in range(ceil(nrows / 128)):
                self.writeBytes(rbcmd_buf[icnk * 128 * 7:min((icnk + 1) * 128 * 7, len(buf))])
                if data_relay:  # slow down transmission since UART to boards is not so fast
                    time.sleep(0.0015)

            nb = self.readBytes(nrows * 7)
            if nb != (nrows * 7):
                print('Error: RAM readback not enough bytes received\n')
            for irow in range(nrows):
                offset = irow * (1 + 2 + 4)
                if self.rxtx_buf[offset] != ram_rb_header or \
                        self.rxtx_buf[offset + 1] != buf[offset + 1] or \
                        self.rxtx_buf[offset + 2] != buf[offset + 2] - 128 or \
                        not all(self.rxtx_buf[offset + 3:offset + 7] == buf[offset + 3:offset + 7]):
                    err_cnt += 1

            if err_cnt > 0:
                print('Error: {:d} RAM {:s} readback errors\n'.format(err_cnt, ram_select))

        if data_relay and ram_select != 'src_raw':
            # restore RAM B
            self.ramb = ramb_old
            self.uploadToRAM('b', False, 0, 1)
    # def updateSrcRAM(self, srcram, skipreadback=False):
    #     if srcram is None:
    #         srcram = np.zeros([7, 1024, 32], dtype=np.uint8)
    #         srcram[:,:,[20, 31]] = 1 # disable LEDs and set all stop bits
    #     self.srcram = srcram
    #     iSatesOn = np.where(srcram[0,:,31]==0)[0] # TODO: what happens if src card 0 not present
    #     self.rama = np.zeros([1024, 32], dtype=np.uint8)
    #     self.n_states_a = len(iSatesOn)
    #     self.n_status_states = 0
    #     self.rama[self.n_states_a:, 8] = 1
    #     for isrcb in range(self.N_SRC_SLOTS):
    #         if self.srcb_active[isrcb]:
    #             self.uploadToRAM('src', skipreadback, isrcb)
    #     self.uploadToRAM('a', skipreadback)


    # ----------------------------------------------------------------------
    def flush(self):
        # Simple flush of data in SPI buffer
        self.rst_tx_fifo = True
        self.run = False
        self.updateStatReg(True)
        time.sleep(0.001)
        self.rst_tx_fifo = False
        self.updateStatReg(True)

        buf = self.spi.readbytes(2)
        bytes_available = buf[0] + 256*buf[1]
        
        while bytes_available>0:
            buf = self.spi.readbytes(bytes_available+2)
            bytes_available = buf[0] + 256*buf[1] - bytes_available
    
        return bytes_available


    # ----------------------------------------------------------------------
    def flushFull(self):
        # This function will flush serial data from all microcontrollers and the FPGA
        # It will also reset the controllers on the detector cards.

        # make sure system is not running
        self.run = False
        self.updateStatReg(True)
        time.sleep(0.05)

        # Reset program counters 
        self.flush()
        self.rst_pca = True
        self.rst_detb = [1] *4
        self.rst_ram = True             # reset external ram
        self.updateStatReg()

        time.sleep(0.05)
        self.rst_pca = False
        self.rst_ram = False
        for ii in range(4):
            self.rst_detb[ii] = 0
            self.updateStatReg()
            time.sleep(0.5)
        time.sleep(0.1)

        # Redundant: also run a bit without sampling
        # (flush data out of tx buffers in all microcontrollers)
        ramb_old = self.ramb.copy()
        self.ramb[:, [8, 9]] = 0                # delete Acq Trg and End Cyc bits
        self.uploadToRAM('b', False)
        self.run = True
        self.updateStatReg()
        time.sleep(0.05)
        self.run = False
        self.updateStatReg()
        time.sleep(0.1)
        self.flush()
        time.sleep(0.1)
        self.flush()

        # Restore RAM B
        self.ramb = ramb_old
        self.uploadToRAM('b', False)


    # ----------------------------------------------------------------------
    def writeBytes(self, vals):
        self.spi.writebytes2(vals)
        # TODO: - implement writing to PySerial
        # TODO: - implement writing to network socket


    # ----------------------------------------------------------------------
    def readBytes(self, n_bytes_req, spi_timeout=0.0):
        n_bytes_rxd = 0
        if spi_timeout==0:
            end_time = time.perf_counter() + self.config.getfloat('spi_timeout')
        else:
            end_time = time.perf_counter() + spi_timeout

        bytes_available = 0
        while time.perf_counter() < end_time and n_bytes_rxd < n_bytes_req:
            buf = self.spi.readbytes(bytes_available+2)
            bytes_available = max(buf[0] + 256*buf[1] - bytes_available, 0)

            if len(buf)>2:  # received data
                self.rxtx_buf[n_bytes_rxd:(n_bytes_rxd+len(buf)-2)] = buf[2:] 
                n_bytes_rxd += len(buf)-2
            bytes_available = min(bytes_available, n_bytes_req - n_bytes_rxd)
        return n_bytes_rxd


