# Setup Raspberry Pi 5 for ninjaWeb on Windows

## 1. Insert SD Card
Insert the SD card into your computer’s SD card reader.

## 2. Download Raspberry Pi Imager
Open your web browser and go to the [Raspberry Pi Software](https://www.raspberrypi.com/software/).

## 3. Download and Install Raspberry Pi Imager
Click the **Download for Windows** button. Once the download completes, run the `.exe` file to install Raspberry Pi Imager on your system.

## 4. Launch Raspberry Pi Imager
After installation, Raspberry Pi Imager will automatically open.

## 5. Select Device
In the Imager interface, under **Device**, select **Raspberry Pi 5**.

## 6. Choose Operating System
Under **Operating System**, choose:
```
Raspberry Pi OS (other) → Raspberry Pi OS Lite (64-bit)
```

## 7. Select Storage
Under **Storage**, choose the SD card that you inserted earlier.

## 8. Proceed to Settings
Click **Next** to continue.

## 9. Edit Settings
Click on **Edit Settings** to configure the Raspberry Pi’s network, hostname, and other settings.

### General:
- **Set Hostname:** `ninja-pi.local`
- **Set Username:** `NinjaNIRS`
- **Set Password:** `nn2022pidev`

### LAN Settings:
- **SSID (Wi-Fi network name):** `ninjaGUIpy`
- **Password:** `ninjaGUIpy2023`
- **Wireless LAN Country:** `US`

### Set Time Zone:
Configure the correct time zone (e.g., `America/New_York` if you're in the Eastern Time Zone).

## 10. Services:
- Enable **SSH** → Use password authentication

## 11. Options:
- Uncheck **Enable telemetry**

Once all settings are configured, press **Save**.

## 12. Write to SD Card
After saving, press **Yes** to confirm and begin writing the Raspberry Pi OS to the SD card.

## 13. Completion
Wait for the process to finish. Once done, safely eject the SD card and insert it into your Raspberry Pi 5.

Your Raspberry Pi 5 is now set up with the **ninjaWeb** configuration. Proceed to power it up and connect it to the internet via LAN or Wi-Fi.

---

## Setup Wi-Fi Router / Mobile Hotspot
- **Wi-Fi Network Name:** `ninjaGUIpy`  
- **Password:** `ninjaGUIpy2023`

Insert the SD card into the `ninjaNIRS` Raspberry Pi.  
Connect your computer's Wi-Fi to the same router.

---

## Copy ninjaWeb Code to Raspberry Pi

Use software like **WinSCP** to transfer the downloaded `ninjaWeb` code from GitHub to the SD card on the Raspberry Pi.

**Connection Details:**
- **Hostname:** `ninja-pi.local`
- **Username:** `NinjaNIRS`
- **Password:** `nn2022pidev`

---

## SSH into Raspberry Pi

Use **PuTTY** to SSH into the Raspberry Pi.

**SSH Details:**
- **Hostname:** `ninja-pi.local`
- **Username:** `NinjaNIRS`
- **Password:** `nn2022pidev`

---

## Setup `data-server` Environment

```bash
cd ~/ninjaWeb/data-server/
python3 -m venv .
source ./bin/activate
# Install required packages
deactivate
```

---

## Setup `bokeh-server` Environment

```bash
cd ~/ninjaWeb/bokeh-server/
python3 -m venv .
source ./bin/activate
# Install required packages
deactivate
```
