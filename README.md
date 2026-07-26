# AquaTroll Logger Data Pipeline
<img src="images/gorgeMonitoring.jpg" alt="Gorge monitoring site" style="width:100%; max-height:400px; object-fit:cover; border-radius: 8px;">

## Project Overview

The industry standard for collecting surface pool depth readings relies on pressure based logger data, which requires regular calibration to remain accurate. In areas with challenging terrain and anthropological sensitivities, infrequent site access leads to unreliable data and can lead to delayed identification of issues. This makes it difficult or even impossible to reliably deliver the best environmental outcomes.

The environmental receptors in my project environment are located in either steep gorges or dense tree canopies. Historically, the off the shelf telemetry systems available were either too bulky, too power hungry, or struggled to maintain sufficent signal to operate reliably. To overcome this, I worked with [Maxy Engineering](https://maxyengineering.com.au/) to develop a highly portable, power independent logging system capable of transmitting data via 4G/5G or the Iridium satellite network. The final production version of this hardware has a battery life of ~2years and has eliminated the need for regular site visits.

This project is the data extraction and validation software that links Maxy's hardware to the company stakeholders. The scripts below automate the fetching of raw data, perform pressure based calibrations of the logger data & distribute the validated results. The end result is calibrated data that empowers the site’s environmental team to review anomalies and, if necessary, take corrective action in a timely manner. Results are delivered in an easy to read stoplight PDF, and a CSV file is sent directly to the company’s SQL database.

## Hardware Components

<table>
<tr>
<td width="50%" valign="top" align="center">
<strong>Maxy Remote Logging Unit</strong><br><br>
<img src="images/logger.jpg" alt="Remote Logger" width="100%">
</td>
<td width="50%" valign="top" align="center">
<strong>Headless Pi & AquaTroll 200</strong><br><br>
<img src="images/raspberry_pi.jpg" alt="Headless Raspberry Pi Setup" width="90%">
<br><br>
<img src="images/aquaTroll200.jpg" alt="AquaTroll 200" width="90%">
</td>
</tr>
</table>

> *To discuss the most suitable setup for your location, contact Maxy Engineering:*
> ✉️ **office@MAXYEngineering.com.au** | 📞 **0478 221 776**

---

## How It Works

The main script, `runPipeline.py`, orchestrates the following steps:

1. **Scrape logger data (`httpLoggerScraper.py`)** — Logs into the AquaTroll portal, navigates to each site & downloads the latest CSV files.
2. **Scrape weather data (`weatherStation.py`)** — Fetches daily barometric pressure readings from an external weather station via HTTP.
3. **Validate and calibrate (`dataValidation.py`)** — Reads raw CSVs, cleans and type-converts the data, merges weather station barometric readings & where appropriate, calculates adjusted depth using In-Situ's AquaTroll formula. Monitors battery voltage against low-voltage thresholds. Outputs validated CSVs & a PDF QAQV report to `transformed_data/`.
4. **Email results (`autoEmail.py`)** — Sends output files as attachments to configured recipients via Gmail SMTP.

All configuration — credentials, site definitions, email settings — is managed through a single `.env` file.

---

## Threshold Calculations

Depth adjustments use a four-tier system based on the pressure difference between the logger's internal barometric sensor and the external weather station:

| Pressure Difference | Behaviour |
|---|---|
| ≤ 5 hPa | No adjustment — within sensor noise tolerance |
| 5–20 hPa | Adjustment applied using AquaTroll formula |
| 20–50 hPa | No adjustment — excessive drift flagged in comments |
| > 50 hPa | No adjustment — possible sensor failure |

A minimum depth threshold of **0.3 metres** prevents corrections in very shallow or dry pools.

If significant discrepancies are observed between logger values and weather station data (e.g. due to elevation differences or microclimates), adjust these thresholds in `dataValidation.py` and `testsAB.py` to suit your environment.

---

## Quality Assurance

Each pipeline run generates a PDF report (`abTestsReport.pdf`) in `transformed_data/`, attached to the validation email. The report gives the environmental team a clear, stoplight view of whether field equipment is online & if this pipeline is running correctly.

Each check returns **PASS**, **WARNING**, or **FAIL**. If everything is green, no action is needed. A WARNING or FAIL is a prompt to investigate, not an immediate cause for concern. For example, a low battery flag may simply mean a site is approaching its normal replacement window.

The report is stamped with machine name, username, and timestamp for audit purposes.

---

## Installation

Clone the repository first, regardless of deployment method:

```bash
git clone https://github.com/muttaphilly/AquaTroll.Telemetry.Pipeline.git
cd AquaTroll.Telemetry.Pipeline
```

Then follow the path that matches your deployment:

- **[Docker (recommended for servers / navigating around Microslop)](#docker-deployment)** — runs in an isolated container, no system dependencies to manage
- **[Bare metal](#direct-raspberry-pi-deployment)** — runs natively with a Python virtual environment

---

## Configuration

Before running your pipeline, you need to populate the `.env` file.

Copy the example and modify to match your deployment:

```bash
cp .env.example .env
nano .env
```

### Environment Variables

```
# Logger portal
BASE_URL="https://yourAquaTrollWebsite.no"
LOGIN_URL="https://yourAquaTrollWebsite.no/login.aspx"
LOGIN_USERNAME="Bombadil"
LOGIN_PASSWORD="brightBlueJacket"
PAUSE_SECONDS=3

# Weather station
WEATHER_URL="yourIndependentWeather.no"

# Email
EMAIL_SENDER_ADDRESS="yourSMTPconfigured@EmailAddress.com"
EMAIL_SENDER_PASSWORD="bootsYellow"
EMAIL_SMTP_SERVER=smtp.gmail.com
EMAIL_SMTP_PORT=587

# Recipients
RECIPIENT_VALIDATION="stooge1@yourcompany.com,stooge2@yourcompany.com"
RECIPIENT_DATABASE="directInjectToSQL@yourcompany.com"
HARDWARE_CONTACT="razvan@maxyengineering.com.au"

# Email content
VALIDATION_EMAIL_SUBJECT=Logger Validation Report
DATABASE_EMAIL_SUBJECT={month} Depth Data
DATABASE_EMAIL_BODY=For Upload To Database: {filename}

# Sites - Source these by cross-checking with details from the AquaTroll website
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

### Site Configuration Reference

**Site ID (the key)** must match the CSV filename without extension. `"Site1"` produces `Site1.csv` and `baroSite1.csv`, and is used as the Sample Point name in output files.

**Depth conversion types:**

| Value | Behaviour |
|---|---|
| `"default"` | Feet to metres (× 0.3048). The default AquaTroll output |
| `"divide_by_100"` | Centimetres to metres (÷ 100) |
| `"metres"` | No conversion — data already in metres |

**Pressure units:**

| Value | Behaviour |
|---|---|
| `"hpa"` | Hectopascals. default for portable AquaTroll sensors |
| `"psi"` | Pounds per square inch. Default for the Starlink hub AquaTroll configurations |

**Enabled flag:** `true` includes the site in the pipeline run. `false` skips it — use for sites not yet deployed or temporarily offline.

**SITES_CONFIG fields:**

| Field | Description |
|---|---|
| `display_name` | Converts aquatroll website to a more friendly to read site name |
| `nav_option` | Upper navigation site ID from the logger portal |
| `depth_conversion_type` | Depth unit conversion to apply |
| `pressure_unit` | Pressure unit of the logger's barometric sensor |
| `enabled` | Include or exclude from pipeline run |
| `level_channel_id` | Channel ID for depth data |
| `baro_channel_id` | Channel ID for barometric pressure data |
| `battery_channel_id` | Channel ID for battery voltage data |

### Adding a New Site

1. Get the `nav_option` from the logger portal.
2. Confirm the pressure unit (`hpa` or `psi`).
3. Confirm the depth conversion needed.
4. Get the three channel IDs from the portal.
5. Add the entry to `SITES_CONFIG` in `.env`.

---

## Docker Deployment

Recommended. Handles all OS-level dependencies inside the container.  No virtual environment or system package installation required. Completely side-steps Microslop and/or company imposed security hurdles. Bare metal works well on Linux/MacOS.

### Prerequisites

- Docker and Docker Compose installed on the host
- `.env` populated (see [Configuration](#configuration) above)

### Directory Structure

```
/opt/aquatroll/
├── Dockerfile
├── docker-compose.yml
├── .env
├── repo/                  ← cloned repository
└── data/
    ├── transformed_data/  ← pipeline output (CSVs, PDF report)
    └── data_downloads/    ← raw CSVs downloaded from portal
```

```bash
mkdir -p /opt/aquatroll/data/transformed_data /opt/aquatroll/data/data_downloads
git clone https://github.com/muttaphilly/AquaTroll.Telemetry.Pipeline.git /opt/aquatroll/repo
cp /opt/aquatroll/repo/.env.example /opt/aquatroll/.env
# Edit /opt/aquatroll/.env with your credentials
```

### Dockerfile

Create `/opt/aquatroll/Dockerfile`:

```dockerfile
FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    libpango-1.0-0 \
    libpangoft2-1.0-0 \
    libharfbuzz0b \
    libffi-dev \
    libjpeg-dev \
    libopenjp2-7 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY repo/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY repo/ .

CMD ["python", "runPipeline.py"]
```

### Docker Compose

Create `/opt/aquatroll/docker-compose.yml`:

```yaml
services:
  aquatroll-pipeline:
    image: aquatroll-pipeline
    build:
      context: .
    volumes:
      - /opt/aquatroll/data/transformed_data:/app/transformed_data
      - /opt/aquatroll/data/data_downloads:/app/data_downloads
    env_file:
      - /opt/aquatroll/.env
    restart: "no"
```

### Build and Test

```bash
cd /opt/aquatroll
docker build -t aquatroll-pipeline .
docker compose run --rm aquatroll-pipeline
```

Output files appear in `/opt/aquatroll/data/transformed_data/`.

### Schedule with Cron

```bash
crontab -e
```

```cron
# AquaTroll Pipeline — runs 28th of each month at 7:00 PM
0 19 28 * * cd /opt/aquatroll && docker compose run --rm aquatroll-pipeline >> /opt/aquatroll/cron.log 2>&1

# Trim log if over 10MB
0 3 1 * * find /opt/aquatroll -name "cron.log" -size +10M -delete
```

### Updating

When new code is pushed to the repository:

```bash
cd /opt/aquatroll/repo && git pull
cd /opt/aquatroll && docker build -t aquatroll-pipeline .
```

The updated image will automaticallt be used on the next scheduled run.

---

## Bare Metal Deployment

For running natively on any Linux/MacOS host without Docker.

### Install System Dependencies

WeasyPrint requires OS-level libraries before `pip install` will work:

**Raspberry Pi / Debian:**
```bash
sudo apt install -y libpango-1.0-0 libpangoft2-1.0-0 libharfbuzz0b libffi-dev libjpeg-dev libopenjp2-7
```

**macOS (Homebrew):**
```bash
brew install pango libffi jpeg openjpeg
```

### Set Up Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Test Run

```bash
python runPipeline.py
```

### Schedule with Cron

```bash
crontab -e
```

```cron
# AquaTroll Pipeline — runs 28th of each month at 7:00 PM
0 19 28 * * /home/pi/AquaTroll.Telemetry.Pipeline/venv/bin/python /home/pi/AquaTroll.Telemetry.Pipeline/runPipeline.py >> /home/pi/AquaTroll.Telemetry.Pipeline/cron.log 2>&1
```

Replace `/home/pi/` with the actual path where you cloned the repository.