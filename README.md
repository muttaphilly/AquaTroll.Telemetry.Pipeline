# AquaTroll Logger Data Pipeline

AquaTroll loggers are widely used for telemetry in hydrology. However, the existing hardware options isn't suitable for use in remote or hard-to-access surface water pools. With support from [Maxy Engineering](https://maxyengineering.com.au/), highly portable, power-independent hardware was developed—capable of transmitting data over 4G/5G or the Iridium satellite network.

This project automates the collection of that data. It performs daily pressure calibrations using external weather data and emails these results to the site environmental team and the company’s environmental data system. Designed for automated operation, the pipeline runs as a cron job on a Raspberry Pi.

## Hardware Components

| Telstra Logger | Iridium Logger | Raspberry Pi Setup |
|----------------|----------------|---------------------|
<!-- 
| ![Telstra Logger](images/telstra_logger.jpg) | ![Iridium Logger](images/iridium_logger.jpg) | ![Raspberry Pi](images/raspberry_pi.jpg) |
-->

## How it Works

The main script, `runPipeline.py`, orchestrates the following steps:
- Fetches AquaTroll logger data
- Calibrates pressure readings using external weather data
- Emails results to relevant stakeholders


1.  **Scraping Logger Data (`loggerScraper.py`):** Uses Playwright to automate a web browser and download raw level and barometric pressure CSV files into the `data_downloads/` directory. Merges these two files for each location.
2.  **Scraping Weather Data (`weatherStation.py`):** Uses a GET Request and BeautifulSoup to scrape weather website for daily barometric pressure readings.
3.  **Data Validation & Calibration (`dataValidation.py`):**
    *   Reads CSVs from `data_downloads/`.
    *   Cleans and validates the data (Coverts types. Also handles erroneous and missing values).
    *   Retrieves external barometric data (by calling `weatherStation.py`).
    *   Merges external weather data with the logger data based on date.
    *   Calculates an 'adjusted depth' by comparing barometer data from the logger's internal sensor and the  external weather station. Uses In-Situ's formula specific to the AquaTroll sensors.
    *   Consolidates processed data into final output CSV files (`validatedDepthData.csv` and `greaterPBOPools.csv`) saved in `transformed_data/`.
4.  **Emailing Results (`autoEmail.py`):** Sends the generated CSV files as attachments to configured email recipients using credentials from the `.env` file.

Configuration for site details, email settings, and target URLs is managed through the `.env` file.

## Setup Instructions

1.  **Clone Repository:**
    ```bash
    git clone https://github.com/muttaphilly/AquaTroll.Telemetry.Pipeline.git
    cd AquaTroll.Telemetry.Pipeline
    ```

2.  **Create Environment File:**
    Create `.env` file in project folder. Populate it with the necessary credentials and configuration.

    For help with configuration or troubleshooting, you can contact me through [GitHub's contact feature](https://github.com/muttaphilly).

3.  **Set up Python Virtual Environment:**
    The Pi requires a virtual environment to manage Python dependencies.
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Linux/macOS
    # venv\Scripts\activate  # On Windows
    ```

4.  **Install Python Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

5.  **Install Node.js Dependencies:**
    Ensure you have Node.js and npm installed.
    ```bash
    npm install
    ```

6.  **Install Playwright Browsers:**
    Playwright needs browser binaries specific to your system.
    ```bash
    python -m playwright install

    ```
    *Note for Raspberry Pi:* This was built and tested on a Raspberry Pi 5 running 64-bit OS. For deployment, it's headless on a Raspberry Pi Zero W using Pi OS Lite to keep power usage and costs low. If issues come up, check compatibility for your specific Pi model and install any missing system dependencies manually.

## Running the Pipeline (ad-hoc)

Execute the main script from the project root directory:

```bash
python runPipeline.py
```

The script will:
*   Scrape data using Playwright, saving files to `data_downloads/`.
*   Validate the data, saving results and logs to `transformed_data/`.
*   Send email reports based on the configuration in `.env`.

## Automating Pipeline with Cron (Linux/Raspberry Pi)

To run automatically, set up a cron job in the terminal.

1.  Open the crontab editor:
    ```bash
    crontab -e
    ```
2.  Add line below to schedule script. For optimising the barometric calibrations, recomending setting for 17:00 on the 28th of every month:
    ```cron
    0 17 28 * * /home/muttaphilly/Desktop/AquaTroll.Telemetry.Pipeline/venv/bin/python /home/muttaphilly/Desktop/AquaTroll.Telemetry.Pipeline/runPipeline.py >> /home/muttaphilly/Desktop/AquaTroll.Telemetry.Pipeline/cron.log 2>&1
    ```
* Make sure to replace the paths if your project is in a different location.

* This uses the Python interpreter inside your virtual environment.

* Output and errors are logged to cron.log in the project root.

3.  Save and close the editor. Cron will automatically pick up the schedule.

```💡 Tip:
To test if your cron job is working, you can run the command manually in your terminal.
Use tail -f cron.log to live-monitor output.
```