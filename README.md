# AquaTroll Logger Data Pipeline
<img src="images/gorgeMonitoring.jpg" alt="Gorge monitoring site" style="width:100%; max-height:400px; object-fit:cover; border-radius: 8px;">

## Project Overview
The industry standard for collecting surface pool depth readings largely relies on pressure-based logger data, which requires regular calibration to remain accurate. In areas with challenging terrain and anthropological sensitivities, infrequent site access leads to unreliable data and delayed identification of issues. This situation makes it impossible to reliably deliver the best possible environmental outcomes.

Off-the-shelf telemetry systems were either too bulky, too power-hungry, or lacked the connectivity required to operate in steep gorges of the project environment. To overcome this, [Maxy Engineering](https://maxyengineering.com.au/) developed a highly portable, power-independent logging system capable of transmitting data via 4G/5G or the Iridium satellite network which eliminates the need for regular site visits.

This project links Maxy's hardware with a software system that automates key processes: fetching raw data, performing daily pressure calibrations using external weather data, and distributing validated results. The calibrated data enables timely review of anomalies and corrective actions. Results are emailed to the site environmental team and formatted for upload into the company's environmental data storage system. The full pipeline runs autonomously on a Raspberry Pi, providing reliable, continuous monitoring with no manual data handling.

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

1. **AquaTroll scraper (`httpLoggerScraper.py`)** Handles the full login process, navigates through the portal menus to each site, and downloads the latest data files. A short configurable pause between each site download keeps requests to the portal at a reasonable rate.
2. **Scraping Weather Data (`weatherStation.py`):** Uses a GET request and BeautifulSoup to scrape the weather website for daily barometric pressure readings.
3. **Data Validation & Calibration (`dataValidation.py`):**
   * Reads CSVs from `data_downloads/`.
   * Cleans and validates the data (converts types, handles erroneous and missing values).
   * Retrieves external barometric data (by calling `weatherStation.py`).
   * Merges external weather data with the logger data based on date.
   * Calculates an 'adjusted depth' by comparing barometer data from the logger's internal sensor and the external weather station. Uses In-Situ's formula specific to the AquaTroll sensors.
   * Monitors telemetry unit battery voltage against low voltage thresholds.
   * Consolidates processed data into final output files (`validatedDepthData.csv`, `SWLVLGenericTemplate_greaterPBOPools.csv` & `abTestsReport.pdf`) saved in `transformed_data/`.
4. **Emailing Results (`autoEmail.py`):** Sends the generated CSV files as attachments to configured email recipients.

Configuration for site details, email settings, and target URLs is managed through the `.env` file.

## Threshold Calculations
Depth adjustments are applied using a four-tier system based on the difference between the logger's internal barometric sensor and the external weather station:

| Pressure Difference | Behaviour |
|---|---|
| ≤ 5 hPa | No adjustment — within sensor noise tolerance |
| 5–20 hPa | Adjustment applied using AquaTroll formula |
| 20–50 hPa | No adjustment — excessive drift flagged in comments |
| > 50 hPa | No adjustment — possible sensor failure |

An additional minimum depth threshold of **0.3 metres** prevents corrections in very shallow or dry pools.

Users should verify if there are any local variations between observed logger values and weather station data post deployment. If significant discrepancies are noted (e.g., individual instruments, elevation differences or microclimates), consider modifying these thresholds in `dataValidation.py` and `testsAB.py` to better suit your specific environment. The goal is to align the deployed, field-calibrated AquaTroll with the observed weather station hPa value.

## Quality Assurance and Testing

Each time the pipeline runs, it automatically generates a PDF report (`abTestsReport.pdf`) saved to `transformed_data/` and attached to the validation email. The report is designed to give the environmental team a clear, at-a-glance view of whether the field equipment is online and the software is running correctly — no technical knowledge required.

Each check returns a simple **PASS**, **WARNING**, or **FAIL**. If everything is green, no action is needed. If a WARNING or FAIL appears, it is intended as a prompt to investigate rather than an immediate cause for concern — a low battery flag, for example, may simply mean a site is approaching its normal replacement window.

The report is stamped with the machine name, username, and timestamp for audit trail purposes.

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd AquaTroll.Telemetry.Pipeline
```

2. Create and activate virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Linux/Mac
```

3. Install system-level dependencies for PDF report generation:

   WeasyPrint (used to generate the PDF report) requires libraries that must be installed at the OS level before running `pip install`.

   **macOS (Homebrew):**
   ```bash
   brew install pango libffi jpeg openjpeg
   ```

   **Raspberry Pi / Debian Linux:**
   ```bash
   sudo apt install -y libpango-1.0-0 libpangoft2-1.0-0 libharfbuzz0b libffi-dev libjpeg-dev libopenjp2-7
   ```

4. Install Python dependencies:
```bash
pip install -r requirements.txt
```

5. Configure environment:
```bash
cp .env.example .env
# Edit .env with your credentials and site configuration
```

6. Test the pipeline:
```bash
python runPipeline.py
```

## Configuration
### Required Environment Variables
```
LOGIN_URL="https://your-portal-url.com/login"
LOGIN_USERNAME="your_username"
LOGIN_PASSWORD="your_password"
PAUSE_SECONDS=3  # Delay between site scrapes (seconds) — prevents rapid sequential portal requests
WEATHER_URL="http://www.independentWeatherStation//your-station-id.shtml"
RECIPIENT_VALIDATION="team@company.com,supervisor@company.com"
RECIPIENT_DATABASE="database-upload@company.com"
VALIDATION_EMAIL_SUBJECT="Logger Validation Report"
DATABASE_EMAIL_SUBJECT="{month} Depth Data"
DATABASE_EMAIL_BODY="For Upload To Database: {filename}"
SITES_CONFIG='{
  "Site1": {
    "display_name": "Site 1",
    "nav_option": "12345",
    "depth_conversion_type": "default",
    "pressure_unit": "hpa",
    "enabled": true,
    "level_channel_id": 10000,
    "baro_channel_id": 10001,
    "battery_channel_id": 10002
  }
}'
```

### Configuration Notes

**Site ID (the key)**
Must match the CSV filename without extension. CSV files will be named `{site_id}.csv` and `baro{site_id}.csv`. Used as the "Sample Point" name in output files. Example: `"Site1"` creates `Site1.csv` and `baroSite1.csv`.

**Depth Conversion Types**

| Value | Behaviour |
|---|---|
| `"default"` | Converts feet to metres (× 0.3048) — standard AquaTroll output |
| `"divide_by_100"` | Converts centimetres to metres (÷ 100) — for cm-based sensors |
| `"metres"` | No conversion applied — data already in metres |

**Pressure Units**

| Value | Behaviour |
|---|---|
| `"hpa"` | Hectopascals — default for most barometric pressure sensors |
| `"psi"` | Pounds per square inch — for some AquaTroll configurations |

**Enabled Flag**
`true`: site will be scraped, validated, and included in reports. `false`: site will be skipped (use for sites not yet deployed or temporarily offline).

**SITES_CONFIG Fields**

| Field | Description |
|---|---|
| `display_name` | Give it a site name |
| `nav_option` | Set the upper navigation site ID from logger portal |
| `depth_conversion_type` | Depth unit conversion to apply |
| `pressure_unit` | Pressure unit of the logger's barometric sensor |
| `enabled` | Adds the site to the pipeline run |
| `level_channel_id` | Channel ID for depth data download |
| `baro_channel_id` | Channel ID for barometric pressure data download |
| `battery_channel_id` | Channel ID for battery voltage data download |

**Adding New Sites**

To add a new monitoring site:
1. Get the `nav_option` from your logger portal.
2. Determine the pressure unit your sensor uses (`hpa` or `psi`).
3. Determine the depth conversion needed (`default` for feet, `divide_by_100` for cm, `metres` if already in metres).
4. Obtain the `level_channel_id`, `baro_channel_id`, and `battery_channel_id` from the portal.
5. Add the entry to `SITES_CONFIG` in your `.env` file.

## Automating Pipeline with Cron (Linux)

To set up a scheduled run:

1. Open the crontab editor for the current user:
```bash
crontab -e
```

2. Add the following line at the bottom of the file to schedule the script. This example runs at 17:00 (5 PM) on the 28th of every month:
```cron
# Run AquaTroll Pipeline monthly
0 17 28 * * /path/to/your/project/AquaTroll.Telemetry.Pipeline/venv/bin/python /path/to/your/project/AquaTroll.Telemetry.Pipeline/runPipeline.py >> /path/to/your/project/AquaTroll.Telemetry.Pipeline/cron.log 2>&1
```

   * **Important:** Replace `/path/to/your/project/` with the actual absolute path to where you cloned the `AquaTroll.Telemetry.Pipeline` directory (e.g., `/home/pi/`).
   * This command explicitly uses the Python interpreter inside your virtual environment (`venv/bin/python`).
   * Output and errors from the script will be appended (`>>`) to `cron.log` in the project directory.

3. Save and close the editor.
   * For `nano`: Press `Ctrl+O`, Enter, then `Ctrl+X`.
   * You should see a message like `crontab: installing new crontab`.

4. **Verify the cron job was added:**
```bash
crontab -l
```
You should see the line you just added listed in the output. This confirms the schedule is active.