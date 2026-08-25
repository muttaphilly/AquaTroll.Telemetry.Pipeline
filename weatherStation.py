import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
import os
import logging
from typing import Optional
from collections import Counter
from dotenv import load_dotenv

def scrape_weather_data() -> Optional[pd.DataFrame]:
    """
    Retrieves barometric pressure and rainfall readings from weather forecasting website.
    Returns DataFrame with date, time, hPa values, and rainfall or None if scraping fails.
    """
    load_dotenv()
    logger = logging.getLogger()
    
    url = os.getenv("WEATHER_URL")
    if not url:
        logger.error("URL not found in environment variables.")
        return None
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    # Fetch and parse
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        logger.error(f"Failed to retrieve webpage: Status code {response.status_code}")
        return None
    
    soup = BeautifulSoup(response.text, 'html.parser')
    table = soup.select_one('table.data')
    if not table:
        logger.error("Table not found on the webpage.")
        return None
    
    # Constants
    current_month = datetime.now().month
    current_year = datetime.now().year
    RAINFALL_INDEX = 4
    HPA_AM_INDEX = 15
    HPA_PM_INDEX = 21
    SUMMARY_ROWS = {'Mean', 'Lowest', 'Highest', 'Total'}
    # Set expected range to help catch obvious bogus data
    PLAUSIBLE_HPA_MIN = 850.0
    PLAUSIBLE_HPA_MAX = 1100.0

    # Checkpoint for weather station data array. Want to catch things
    # silently pointing at wrong field (e.g. a temperature reading
    # instead of pressure)
    valid_rows = []
    for tr in table.select('tbody tr'):
        cells = tr.select('th, td')
        if not cells:
            continue
        date_value = cells[0].get_text(strip=True)
        if date_value in SUMMARY_ROWS or not date_value.isdigit():
            continue
        valid_rows.append((date_value, cells))

    if not valid_rows:
        logger.warning("No data rows found in weather table")
        return None

    cell_counts = Counter(len(cells) for _, cells in valid_rows)
    expected_cell_count = cell_counts.most_common(1)[0][0]

    # Extract data
    rows = []
    for date_value, cells in valid_rows:
        formatted_date = f"{int(date_value):02d}/{current_month:02d}/{current_year}"

        if len(cells) != expected_cell_count:
            logger.warning(
                f"Skipping {formatted_date}: row has {len(cells)} cells, "
                f"table's typical row has {expected_cell_count} - a missing "
                f"observation likely shifted this row's columns, so its "
                f"positional values can't be trusted"
            )
            continue

        # Extract rainfall (pandas handles conversion)
        rainfall = cells[RAINFALL_INDEX].get_text(strip=True) if len(cells) > RAINFALL_INDEX else None
        
        # Extract pressure readings (pandas does conversion and validation heavy lifting)
        readings = []
        if len(cells) > HPA_AM_INDEX:
            hpa_am = cells[HPA_AM_INDEX].get_text(strip=True)
            if hpa_am:
                readings.append(('12:00:00', hpa_am))
        
        if len(cells) > HPA_PM_INDEX:
            hpa_pm = cells[HPA_PM_INDEX].get_text(strip=True)
            if hpa_pm:
                readings.append(('18:00:00', hpa_pm))
        
        # Build rows
        for time_val, hpa_val in readings:
            rows.append({
                'Date': formatted_date,
                'Time': time_val,
                'hPa': hpa_val,
                'Rainfall': rainfall
            })
    
    if not rows:
        logger.warning("No valid barometric pressure readings found")
        return None
    
    # Create and clean DataFrame
    df = pd.DataFrame(rows)
    df['hPa'] = pd.to_numeric(df['hPa'], errors='coerce')
    df['Rainfall'] = pd.to_numeric(df['Rainfall'], errors='coerce')
    df = df.dropna(subset=['hPa'])

    # Reject if any weird hPa readings comeback (<980 or >1050).
    implausible = (df['hPa'] < PLAUSIBLE_HPA_MIN) | (df['hPa'] > PLAUSIBLE_HPA_MAX)
    if implausible.any():
        for _, r in df[implausible].iterrows():
            logger.warning(
                f"Rejected implausible hPa value {r['hPa']} on {r['Date']} "
                f"{r['Time']} (outside {PLAUSIBLE_HPA_MIN}-{PLAUSIBLE_HPA_MAX} hPa)"
            )
        df = df[~implausible]

    return df