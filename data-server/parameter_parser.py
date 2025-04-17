import os
import configparser
from sys import getsizeof

from scipy import io
import numpy as np
# from scipy.special import parameters
import math_utils as mu


class ParameterParser:
    def __init__(self):
        # load Config.cfg
        file_dir = os.path.dirname(os.path.realpath(__file__))
        config_file_path = os.path.join(file_dir, '..', 'probes', 'Config.cfg')
        config = configparser.ConfigParser()
        config.read(config_file_path)

        # load SD file
        probe_filename = config['Probe']['filename']
        probe_file_path = os.path.join(file_dir, '..', 'probes', probe_filename)
        probe = io.loadmat(probe_file_path)
        self.probe = probe

        # create srcrm and mapped_indices
        meas_list = probe['SD']['MeasList'][0][0]
        self.srcram = self.create_srcram(meas_list, 3)
        # meas_list = self.meas_list
        rhoSDS = np.zeros((meas_list.shape[0],))
        SrcPos3D = probe['SD']['SrcPos3D'][0][0]
        DetPos3D = probe['SD']['DetPos3D'][0][0]
        for iML in range(meas_list.shape[0]):
            iS = int(meas_list[iML, 0]) - 1
            iD = int(meas_list[iML, 1]) - 1
            rhoSDS[iML] = np.sqrt(np.sum((SrcPos3D[iS, :] - DetPos3D[iD, :]) ** 2))

        meas_list = np.column_stack((meas_list, rhoSDS))
        self.meas_list = meas_list
        self.mapped_indices = self.map_to_measurementlist(self.srcram, self.meas_list)

    def create_srcram(self, meas_list, iPower):
        """
        This will create state matrices for sources and rama
        MeasList: measurement list from probe
        iPower: power level for sources
        """

        # Initialize srcram array
        srcram = np.zeros((7, 1024, 32), dtype=np.uint16)
        srcram[:, :, 20] = 1
        srcram[:, :, 30] = 1  # using this bit as a hack for identifying source 0 when we look at 5 bits in mapToMeasurementList()

        lstS = np.unique(meas_list[:, 0])

        # loop over unique sources
        # fill in odd states and leave even states as dark states
        # alternate between two wavelengths for each state

        # Initialize maxPower array
        maxPower = np.round(np.logspace(3, np.log10(2**16 - 1), 7)).astype(int)
        maxPower = np.insert(maxPower, 0, 0)

        iState = 0

        for iS in range(len(lstS)):
            iSrcMod = int(np.ceil((lstS[iS] - 0.1) / 8))-1
            iSrc = int((lstS[iS] - 1) % 8)


            srcram[iSrcMod, iState, :16] = mu.bitget(maxPower[iPower], range(0,16))
            srcram[iSrcMod, iState, 16:20] = mu.bitget(iSrc*2, range(0,4))
            srcram[iSrcMod, iState, 20] = 0
            srcram[iSrcMod, iState, 30] = 0
            if iPower == 0:
                srcram[iSrcMod, iState, 20] = 1

            iState += 2

            srcram[iSrcMod, iState, :16] = mu.bitget(maxPower[iPower], range(0,16))
            srcram[iSrcMod, iState, 16:20] = mu.bitget(iSrc*2+1, range(0,4))
            srcram[iSrcMod, iState, 20] = 0
            srcram[iSrcMod, iState, 30] = 0
            if iPower == 0:
                srcram[iSrcMod, iState, 20] = 1

            iState += 2  # +2 to include a dark state

        # Mark sequence end
        srcram[:, (iState - 1):, 31] = 1

        return srcram

    def map_to_measurementlist(self, srcram, meas_list, srcpower_low_high=None):
        """
        Function to map the state map (detector and sources) to standard measurement list
        the standard measurement list is a list of four element arrays, each one
        indicates, the source, the detector and the wavelength associated which
        each channel. The function will identiy which which channel corresponds
        to which state and will return the indices that move the elements of the
        B matrix to their correct channel position
        RIGHT NOW MAPPED INDICES IS NOT WHAT IT IS SUPPOSED TO BE, I AM NOT SURE
        HOW TO DO THE INDEXING AND THE ARRAYS BEING 3D COMPLICATES THINGS
        """

        SSfirstDidx = int(np.max(meas_list[:, 1]) + 1)
        if meas_list.shape[1] == 5:
            rhoSDS = meas_list[:, 4]
            lstDindices = meas_list[np.where(rhoSDS < 10)[0], 1]  # these are the SS detectors
            if lstDindices.size > 0:
                SSfirstDidx = int(np.min(
                    lstDindices))  # this is the first SS detector.  The next 7 are also assumed to be SS detectors measured by this first same Det Inde

        meas_list = meas_list.astype(int)
        if srcpower_low_high is None:
            srcpower_low_high = np.uint16([[[2] * 2] * max(meas_list[:, 1])] * max(meas_list[:, 0]))

        foo = np.where(srcram[0, :, 31] == 1)[0]
        N_STATES = foo[0] + 1

        # Initialize arrays
        estados = np.full(meas_list.shape[0], np.nan)
        estados2 = np.full(meas_list.shape[0], np.nan)
        estados_det = np.full(meas_list.shape[0], np.nan)

        for ki in range(meas_list.shape[0]):
            meas = meas_list[ki, :]

            srcID0 = meas[0]
            srcID = srcID0
            srcModule = int(np.ceil((srcID0 - 0.1) / 8)) - 1

            sourceIDs = np.squeeze(srcram[srcModule, 0:N_STATES, [16, 17, 18, 19, 30]])
            sourceIDs = sourceIDs.T
            detID = meas[1]
            lambdaID = meas[3]

            detID0 = detID
            if detID >= SSfirstDidx:  # This is an SS detector
                detID = SSfirstDidx

            # try:
            # if some sources are never turned on but are required by the
            # measurement list, those states will be marked as nan
            foo = (srcID - 1) % 8 * 2 + (lambdaID - 1)
            bits = mu.bitget(foo, range(0,5))
            Lia = np.all(sourceIDs == bits, axis=1)
            # Lia = mu.ismember(sourceIDs, mu.bitget(foo, list(range(0, 5))), 'rows')
            lst = np.where(Lia)[0]

            if len(lst) == 1:
                estados[ki] = lst[0]
            else:
                estados[ki] = lst[int(srcpower_low_high[srcID - 1, detID0 - 1, lambdaID - 1]) - 1]

            iDark = np.where(srcram[srcModule, int(estados[ki]) + 1:N_STATES, 20] == 1)[0]
            if iDark.size > 0:
                estados2[ki] = iDark[0] + estados[ki]

            estados_det[ki] = detID

        mapped_indices = np.column_stack((estados, estados_det-1, estados2))
        return mapped_indices.astype(int)

if __name__ == "__main__":
    parameters = ParameterParser()





