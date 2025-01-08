# NinjaNIRS 2024 - Raspberry Pi Zero SPI byte stream recorder
#
# bzim@bu.edu
# Initial version 2023-3-29
# Modified to listen to digital in and start a new file 2024-10-16

import nn_shared_memory
import nn_data_recorder
import nn_parser
from multiprocessing import Process

if __name__ == "__main__":
    try:
        sm = nn_shared_memory.NNSharedMemory(main=True)
        parser = Process(target=nn_parser.parserLoop)
        parser.start()
        nn_data_recorder.recorderLoop()

    except KeyboardInterrupt: # Close the program by pressing Ctrl-C
        sm.status_shm[sm.STATUS_SHM_IDX['shutdown']] = True
        parser.join()
        sm.close()

  

