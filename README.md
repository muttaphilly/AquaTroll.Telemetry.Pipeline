# AquaTroll Logger Data Pipeline

This project scrapes data from loggers, perfomrs daily calibrations then emails results. It is designed to be run automatically, using a cron job on a Raspberry Pi.

## How it Works

The project automates the process of fetching AquaTroll logger data, calibrating it using external weather data, and emailing the results. The main script, `runPipeline.py`, orchestrates the following steps:

1.  **Scraping Logger Data (`loggerScraper.py`):** Uses Playwright to automate a web browser, log into the logger website for each configured site, navigate to the data export section, download raw level and barometric pressure CSV files into the `data_downloads/` directory, and merges these two files per site.
2.  **Scraping Weather Data (`weatherStation.py`):** Uses Requests and BeautifulSoup to scrape the Bureau of Meteorology (BoM) website for daily barometric pressure readings.
3.  **Data Validation & Calibration (`dataValidation.py`):**
    *   Reads the merged raw data CSVs from `data_downloads/`.
    *   Cleans and validates the data (handles missing values, converts types).
    *   Retrieves the BoM barometric data (by calling `weatherStation.py`).
    *   Merges the BoM data with the logger data based on date.
    *   Calculates an 'adjusted depth' by comparing the logger's internal barometer reading with the external BoM reading, applying a formula specific to the AquaTroll sensors.
    *   Consolidates the processed data from all sites into final output CSV files (`validatedDepthData.csv` and `greaterPBOPools.csv`) saved in `transformed_data/`.
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

    For help with configuration or troubleshooting, contact me through [GitHub's contact feature](https://github.com/muttaphilly).

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
    *Note for Raspberry Pi:* Ensure you are using a compatible Raspberry Pi OS (like the 64-bit version) and that Playwright supports the ARM architecture. Installation might take longer. Check the official Playwright documentation for ARM/Raspberry Pi support if you encounter issues.

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
