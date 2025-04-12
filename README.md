# AquaTroll Project Data Pipeline

This project scrapes data from the Unidata Neon logger portal, validates it, and emails reports. It is designed to be run automatically, using a cron job on a Raspberry Pi.

## Setup Instructions

1.  **Clone Repository:**
    ```bash
    git clone https://github.com/muttaphilly/AquaTroll.Telemetry.Pipeline.git
+   cd AquaTroll.Telemetry.Pipeline
    ```
    *(Replace `<your-repository-url>` with the actual URL after uploading to GitHub)*

2.  **Create Environment File:**
    Copy the `.env` file into the project root folder. Populate it with the necessary credentials and configuration. 
    ```

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
    # or npx playwright install
    ```
    *Note for Raspberry Pi:* Ensure you are using a compatible Raspberry Pi OS (like the 64-bit version) and that Playwright supports the ARM architecture. Installation might take longer. Check the official Playwright documentation for ARM/Raspberry Pi support if you encounter issues.

## Running the Pipeline

Execute the main script from the project root directory:

```bash
python runPipeline.py
```

The script will:
*   Scrape data using Playwright, saving files to `data_downloads/`.
*   Validate the data, saving results and logs to `transformed_data/`.
*   Send email reports based on the configuration in `.env`.

## Automating with Cron (Linux/Raspberry Pi)

To run the pipeline automatically, set up a cron job.

1.  Open the crontab editor:
    ```bash
    crontab -e
    ```
2.  Add a line to schedule the script. For optimal barometric calibrations, I have set to run 17:00 on the 28th of every month:
    ```cron
    0 17 28 * * /path/to/your/project/venv/bin/python /path/to/your/project/runPipeline.py >> /path/to/your/project/cron.log 2>&1
    ```
    *   **Replace `/path/to/your/project/`** with the absolute path to the `AquaTroll-Project` directory.
    *   Ensure you use the Python executable from within your virtual environment (`venv/bin/python`).
    *   `>> /path/to/your/project/cron.log 2>&1` redirects standard output and errors to a log file. Adjust the log path as needed.

3.  Save and close the editor. Cron will automatically pick up the schedule.
