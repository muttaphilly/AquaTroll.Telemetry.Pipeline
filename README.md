# AquaTroll Logger Data Pipeline
<img src="images/gorgeMonitoring.jpg" alt="Gorge monitoring site" style="width:100%; max-height:400px; object-fit:cover; border-radius: 8px;">

## Project Overview

The industry standard for collecting surface pool depth readings largely relies on pressure-based logger data, which requires regular calibration to remain accurate. In areas with challenging terrain and anthropological sensitivities, infrequent site access leads to unreliable data and delayed identification of issues. This situation makes it impossible to reliably deliver the best possible environmental outcomes.

Off-the-shelf telemetry systems were either too bulky, too power-hungry, or lacked the connectivity required to operate in steep gorges of the project environment. To overcome this, [Maxy Engineering](https://maxyengineering.com.au/) developed a highly portable, power-independent logging system capable of transmitting data via 4G/5G or the Iridium satellite network—eliminating the need for regular site visits.

This project links Maxy’s hardware with a software system that automates key processes: fetching raw data, performing daily pressure calibrations using external weather data, and distributing validated results. The calibrated data enables timely review of anomalies and corrective actions. Results are emailed to the site environmental team and formatted for upload into the company’s environmental data storage system. The full pipeline runs autonomously on a Raspberry Pi, providing reliable, continuous monitoring with no manual data handling.

## Hardware Components

<table>
  <tr>
    <td width="50%" valign="top" align="center">
      <strong>Maxy Remote Logging Unit</strong><br><br>
      <img src="images/logger.jpg" alt="Remote Logger" width="100%"> 
    </td>
    <td width="50%" valign="top" align="center">
      <strong>Headless Pi & AquaTroll200</strong><br><br>
      <img src="images/raspberry_pi.jpg" alt="Headless Raspberry Pi Setup" width="90%">
      <br><br>
      <img src="images/aquaTroll200.jpg" alt="AquaTroll 200" width="90%">
    </td>
  </tr>
</table>

> *To discuss the most suitable setup for your location, contact Maxy Engineering at:* <br>
> ✉️ **office@MAXYEngineering.com.au** <br>
> 📞 **0478 221 776**

## How it Works

The main script, `runPipeline.py`, orchestrates the following steps:

1.  **Scraping Logger Data (`loggerScraper.py`):** Uses Playwright to automate a web browser and download raw level and barometric pressure CSV files into the `data_downloads/` directory. Merges these two files for each location.
2.  **Scraping Weather Data (`weatherStation.py`):** Uses a GET Request and BeautifulSoup to scrape weather website for daily barometric pressure readings.
3.  **Data Validation & Calibration (`dataValidation.py`):**
    *   Reads CSVs from `data_downloads/`.
    *   Cleans and validates the data (Converts types. Also handles erroneous and missing values).
    *   Retrieves external barometric data (by calling `weatherStation.py`).
    *   Merges external weather data with the logger data based on date.
    *   Calculates an 'adjusted depth' by comparing barometer data from the logger's internal sensor and the  external weather station. Uses In-Situ's formula specific to the AquaTroll sensors.
    *   Consolidates processed data into final output CSV files (`validatedDepthData.csv` and `SWLVLGenericTemplate_greaterPBOPools.csv`) saved in `transformed_data/`.
4.  **Emailing Results (`autoEmail.py`):** Sends the generated CSV files as attachments to configured email recipients.

Configuration for site details, email settings, and target URLs is managed through the `.env` file.

## Setup Instructions

1.  **Clone Repository:**
    ```bash
    git clone https://github.com/muttaphilly/AquaTroll.Telemetry.Pipeline.git
    cd AquaTroll.Telemetry.Pipeline
    ```

2.  **Create Environment File:**
    Create `.env` file in project folder. Populate it with the necessary credentials and configuration.

3.  **Set up Python Virtual Environment:**
    The Pi requires a virtual environment to manage Python dependencies.
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

4.  **Install Python Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

5.  **Install Node.js Dependencies:**
    Ensure you have Node.js and npm installed.
    ```bash
    sudo apt update
    sudp apt install nodejs npm
    npm install
    ```

6.  **Install Playwright Browsers:**
    Playwright needs browser binaries specific to your system.
    ```bash
    python -m playwright
    ```
    *Note for OS:* This was built and tested on macOS and Raspberry Pi 5 64-bit OS. For deployment, it's headless on 64-bit OS Lite. Use Windows at your own peril— corporate restrictons, inconsistent dependencies or weird shell behavior will break stuff.

## Running the Pipeline (manually)

Use this in the terminal:

```bash
python runPipeline.py
```

## Automating Pipeline with Cron (Linux/Raspberry Pi)

To setup a scheduled run:

1.  Open the crontab editor for the current user:
    ```bash
    crontab -e
    ```

2.  Add the following line at the bottom of the file to schedule the script. This example runs at 17:00 (5 PM) on the 28th of every month:
    ```cron
    # Run AquaTroll Pipeline monthly
    0 17 28 * * /path/to/your/project/AquaTroll.Telemetry.Pipeline/venv/bin/python /path/to/your/project/AquaTroll.Telemetry.Pipeline/runPipeline.py >> /path/to/your/project/AquaTroll.Telemetry.Pipeline/cron.log 2>&1
    ```
    *   **Important:** Replace `/path/to/your/project/` with the actual absolute path to where you cloned the `AquaTroll.Telemetry.Pipeline` directory (e.g., `/home/pi/`).
    *   This command explicitly uses the Python interpreter inside your virtual environment (`venv/bin/python`).
    *   Output and errors from the script will be appended (`>>`) to `cron.log` in the project directory.

3.  Save and close the editor.
    *   For `nano`: Press `Ctrl+O`, Enter, then `Ctrl+X`.
    *   You should see a message like `crontab: installing new crontab`.

4.  **Verify the cron job was added:** List the active cron jobs for the user:
    ```bash
    crontab -l
    ```
    You should see the line you just added listed in the output. This confirms the schedule is active.