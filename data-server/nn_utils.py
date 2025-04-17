import RPi.GPIO as GPIO
import os
import datetime


# The raspberry pi will not have a correct system time if not connected to the internet,
# thus creating filenames with timestamps does not make sense. Instead use an increasing
# counter as the filename root.
def getFileNames(is_calib = False, calib_level = 0):
    data_dir = os.path.join('..', 'meas', datetime.datetime.now().strftime('%y-%m-%d'))
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    if not is_calib:
        filename = 'ninjaWeb_'+datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        filename_bin = os.path.join(data_dir,filename+'.bin')
        filename_log = os.path.join(data_dir, filename + '.log')
        filename_json = os.path.join(data_dir, filename+'_stateMap.mat')
    else:
        calib_folder = os.path.join(data_dir, 'LEDPowerCalibration')
        if not os.path.exists(calib_folder):
            os.makedirs(calib_folder)
        filename = f'LEDPowerCalibration_{calib_level:02d}_' + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        filename_bin = os.path.join(calib_folder, filename + '.bin')
        filename_log = os.path.join(calib_folder, filename + '.log')
        filename_json = os.path.join(data_dir, filename + '_stateMap.mat')

    # counter_file = SAVE_FOLD + "file_counter.txt"
    # if os.path.exists(counter_file):
    #     with open(counter_file, 'r') as f:
    #         counter = int(f.read().strip())
    # else:
    #     counter = 0
    #
    # fname_root = f"{counter:04}_NN24"
    #
    # # make sure .bin and .log files don't exist
    # while os.path.exists(SAVE_FOLD + fname_root + ".bin") or os.path.exists(SAVE_FOLD + fname_root + ".log"):
    #     counter += 1
    #     fname_root = f"{counter:04}_NN24"
    #
    # # Store counter of next filename to use
    # with open(counter_file, 'w') as f:
    #     f.write(str(counter+1))
    #
    # fname_bin = SAVE_FOLD + fname_root + ".bin"
    # fname_log = SAVE_FOLD + fname_root + ".log"

    return filename_bin, filename_log, filename_json



class NN_GPIOS():
    LED_RED_PIN = 11
    LED_GREEN_PIN = 7
    BUZZ_PIN = 33   

    def __init__(self):
        # Setup GPIOs
        # RUN_PIN = 40
        GPIO.setmode(GPIO.BOARD)
        # GPIO.setup(RUN_PIN, GPIO.IN)
        GPIO.setup(self.LED_GREEN_PIN, GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(self.LED_RED_PIN, GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(self.BUZZ_PIN, GPIO.OUT, initial=GPIO.LOW)
        #self.buzz = GPIO.PWM(self.BUZZ_PIN, 600) # 600 Hz

    def led_set(self, color='red', on=False):
        if 'red' in color:
            pin = self.LED_RED_PIN
        else:
            pin = self.LED_GREEN_PIN
        if on:
            GPIO.output(pin, GPIO.HIGH)
        else:
            GPIO.output(pin, GPIO.LOW)

    def cleanup(self):
        GPIO.cleanup()
    


class DataLoopStats():
    def __init__(self):
        self.n_bytes_rxd = 0
        self.n_bytes_rxd_total = 0
        self.n_cyc = 0
        self.n_cyc_wait = 0
        self.ishift_max = 0
        self.max_bytes_available = 0
        self.n_queue_skip = 0
        self.n_idle_reports = 0

    def __str__(self):
        out_str = f"Bytes rcvd tot: {self.n_bytes_rxd_total:11}  Last: {self.n_bytes_rxd:8}"\
                f" | Cycs (wait/tot): {self.n_cyc_wait:7} /{self.n_cyc:7}"\
                f" | iShiftMax: {self.ishift_max} | Max bytes avail: {self.max_bytes_available:4} | N queue skips: {self.n_queue_skip}"
        return out_str
    
    def reset_loop(self):
        self.n_bytes_rxd = 0
        #self.n_bytes_rxd_total = 0
        self.n_cyc = 0
        self.n_cyc_wait = 0
        self.ishift_max = 0
        self.max_bytes_available = 0
        self.n_queue_skip = 0

    def reset_all(self):
        self.reset_loop()
        self.n_bytes_rxd_total = 0
        self.n_idle_reports = 0

    

