
---

## Table of Contents

* [Prepare Your SD Card with Raspberry Pi5 OS](#prepare-your-sd-card-with-raspberry-pi-os)
* [Setup Wi-Fi Router / Mobile Hotspot](#setup-wi-fi-router--mobile-hotspot)
* [Copy ninjaWeb Code to Raspberry Pi](#copy-ninjaweb-code-to-raspberry-pi)
* [SSH into Raspberry Pi](#ssh-into-raspberry-pi)
* [Enable SPI kernal module](#Enable-SPI-kernal-module)
* [Setup `data-server` and `bokeh-server` Environments](#Setup-data-server-and-bokeh-server-Environments)
* [Run ninjaWeb](#Run-NinjaWeb)

---

## Prepare Your SD Card with Raspberry Pi OS

Here's a step-by-step guide to preparing your SD card for Raspberry Pi OS:

### 1. Insert SD Card

Insert the SD card into your computer’s SD card reader.

### 2. Download Raspberry Pi Imager

Open your web browser and go to the [Raspberry Pi Software](https://www.raspberrypi.com/software/) download page.

### 3. Download and Install Raspberry Pi Imager

Press the **Download for Windows** button. Once the download completes, run the `.exe` file to install Raspberry Pi Imager on your system.

### 4. Launch Raspberry Pi Imager

After installation, Raspberry Pi Imager will automatically open.

### 5. Select Device

In the Imager interface, under **Device**, select **Raspberry Pi 5**.

### 6. Choose Operating System

Under **Operating System**, choose:

* **Raspberry Pi OS (other)** $\rightarrow$ **Raspberry Pi OS Lite (64-bit)**.

### 7. Select Storage

Under **Storage**, choose the SD card that you inserted earlier.

### 8. Proceed to Settings

Press **Next** to continue.

### 9. Edit Settings

Click on **Edit Settings** to configure the Raspberry Pi’s network, hostname, and other settings. Enter the following:

**General:**

* **Set Hostname:** `ninja-pi.local`

* **Set Username:** `NinjaNIRS`

* **Set Password:** `nn2022pidev`

**LAN Settings:**

* **SSID (Wi-Fi network name):** `ninjaGUIpy`

* **Password:** `ninjaGUIpy2023`

* **Wireless LAN Country:** `US`

**Set Time Zone:**

* Configure the correct time zone (for example, `America/New_York` if you're in the Eastern Time Zone).

### 10. Services

Enable **SSH** and select **Use password authentication**.

### 11. Options

Uncheck **Enable telemetry**.

Once all settings are configured, press **Save**.

### 12. Write to SD Card

After saving, you'll be asked to confirm; press **Yes** to begin writing the Raspberry Pi OS to the SD card.

### 13. Completion

Wait for the process to finish. Once done, safely eject the SD card and insert it into your Raspberry Pi 5.

Your Raspberry Pi 5 is now set up with the **ninjaWeb** configuration. Proceed to power it up and connect it to the internet via LAN or Wi-Fi, and you should be good to go!

---

## Setup Wi-Fi Router / Mobile Hotspot

To ensure seamless connectivity, configure your Wi-Fi router or mobile hotspot with the following details:

* **Wi-Fi Network Name:**
    ```
    ninjaGUIpy
    ```
* **Password:**
    ```
    ninjaGUIpy2023
    ```

Insert the **SD card** into the `ninjaNIRS` Raspberry Pi. Connect your computer's Wi-Fi to the same router.

---

## Copy ninjaWeb Code to Raspberry Pi

You'll need to transfer the `ninjaWeb` code from GitHub to your Raspberry Pi. Software like **WinSCP** is ideal for this.

Use the following connection details:

* **Hostname:**
    ```
    ninja-pi.local
    ```
* **Username:**
    ```
    NinjaNIRS
    ```
* **Password:**
    ```
    nn2022pidev
    ```

---

## SSH into Raspberry Pi

To access your Raspberry Pi's command line, use an SSH client like **PuTTY**.

Here are the SSH details:

* **Hostname:**
    ```
    ninja-pi.local
    ```
* **Username:**
    ```
    NinjaNIRS
    ```
* **Password:**
    ```
    nn2022pidev
    ```

---

## Enable SPI kernal module

Run ```sudo raspi-config``` in the terminal, select interface options and Enable automatic loading of SPI kernal module.

---

## Setup `data-server` and `bokeh-server` Environments

Now run these commands to set up the `data-server` and `bokeh-server` environments:

```bash
cd ~/ninjaWeb/data-server/
python3 -m venv .
source ./bin/activate
pip install -r requirements.txt
deactivate

cd ~/ninjaWeb/bokeh-server/
python3 -m venv .
source ./bin/activate
pip install -r requirements.txt
deactivate
```

---

## Run NinjaWeb

### 1. Power On
- Turn on the **ninjaNIRS** device.

### 2. Network Configuration
- Ensure that **both the ninjaNIRS system and the laptop running ninjaWeb** are connected to the **same Wi-Fi network**.

### 3. Launch Servers
Open **two putty terminal windows** on the computer and ssh into  raspberry pi

### Terminal 1: Start Data Server
```bash
cd ~/ninjaWeb/data-server/
source ./bin/activate
python3 nn_data_server_main.py
```

### Terminal 2: Start Bokeh Server
```bash
cd ~/ninjaWeb/bokeh-server/
source ./bin/activate
bokeh serve web-app --allow-websocket-origin=ninja-pi.local:5006
```
These two processes can be automatically launched when the Raspberry Pi boots up.

### 4. Access Web Interface
- Open a web browser and navigate to:  
  **http://ninja-pi.local:5006/web-app**

### 5. Set System Time
- Click the **“Date and Time”** button to sync the system clock on ninjaNIRS.  
  > 🔹 *The button will fade to gray once the time is set successfully.*  
  > 🔹 *If the router is connected to the internet, the time may already be set.*

### 6. Enable Power Calibration
- Click the **"Enable Power Calibration"** button to activate calibration controls.

## 7. Perform Calibration
- Complete the **power calibration** process and press **Return** button

## 8. Start Data Collection
- After calibration, click the **“Run”** button to begin data collection.

