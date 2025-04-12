# AquaTroll Project Data Pipeline

This project scrapes data from the Unidata Neon logger portal, validates it, and emails reports. It is designed to be run automatically, using a cron job on a Raspberry Pi.

## Setup Instructions

1.  **Clone Repository:**
    ```bash
    git clone https://github.com/muttaphilly/AquaTroll.Telemetry.Pipeline.git
+   cd AquaTroll.Telemetry.Pipeline
    ```

2.  **Create Environment File:**
    Copy the `.env` file into the project root folder. Populate it with the necessary credentials and configuration. 

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
```