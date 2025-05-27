# Setup Raspberry Pi 5 for ninjaWeb on Windows

This guide outlines the steps to set up your **Raspberry Pi 5** for the **ninjaWeb** application using a Windows computer.

---

## Table of Contents

* [Prepare Your SD Card with Raspberry Pi OS](#prepare-your-sd-card-with-raspberry-pi-os)
* [Setup Wi-Fi Router / Mobile Hotspot](#setup-wi-fi-router--mobile-hotspot)
* [Copy ninjaWeb Code to Raspberry Pi](#copy-ninjaweb-code-to-raspberry-pi)
* [SSH into Raspberry Pi](#ssh-into-raspberry-pi)
* [Setup `data-server` Environment](#setup-data-server-environment)
* [Setup `bokeh-server` Environment](#setup-bokeh-server-environment)

---

## Prepare Your SD Card with Raspberry Pi OS

First, insert the **SD card** into your computer’s **SD card reader**.

Next, open your web browser and go to the [Raspberry Pi Software](https://www.raspberrypi.com/software/) download page. Click the **Download for Windows** button, and once the download is complete, run the `.exe` file to install **Raspberry Pi Imager** on your system. The Imager will automatically open after installation.

In the Imager interface, under **Device**, select **Raspberry Pi 5**. Under **Operating System**, choose:

For **Storage**, select the **SD card** you inserted earlier, then click **Next** to continue.

Now, click on **Edit Settings** to configure your Raspberry Pi.

Under **General**, set the following:
* **Set Hostname:**
    ```
    ninja-pi.local
    ```
* **Set Username:**
    ```
    NinjaNIRS
    ```
* **Set Password:**
    ```
    nn2022pidev
    ```

For **LAN Settings**:
* **SSID (Wi-Fi network name):**
    ```
    ninjaGUIpy
    ```
* **Password:**
    ```
    ninjaGUIpy2023
    ```
* **Wireless LAN Country:**
    ```
    US
    ```

Make sure to configure the correct **Time Zone** (e.g., `America/New_York` if you're in the Eastern Time Zone).

Under **Services**, enable **SSH** and select **Use password authentication**.

In **Options**, uncheck **Enable telemetry**.

Once all settings are configured, press **Save**. You'll be asked to confirm; press **Yes** to begin writing the Raspberry Pi OS to the SD card.

Wait for the process to finish. Once done, safely eject the **SD card** and insert it into your **Raspberry Pi 5**.

Your Raspberry Pi 5 is now set up with the **ninjaWeb** configuration. Proceed to power it up and connect it to the internet via LAN or Wi-Fi.

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

## Setup `data-server` Environment

Once you've SSH'd into the Raspberry Pi, run these commands to set up the `data-server` environment:

```bash
cd ~/ninjaWeb/data-server/
python3 -m venv .
source ./bin/activate
# Install required packages (e.g., pip install -r requirements.txt if a requirements file exists)
deactivate

cd ~/ninjaWeb/bokeh-server/
python3 -m venv .
source ./bin/activate
# Install required packages (e.g., pip install -r requirements.txt if a requirements file exists)
deactivate
