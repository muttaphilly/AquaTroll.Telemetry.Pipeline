import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
import os
import logging
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv

def scrape_weather_data() -> Optional[pd.DataFrame]:
    """
    Retrieves barometric pressure and rainfall readings from weather forecasting website.
    Returns DataFrame with date, time, hPa values, and rainfall or None if scraping fails.
    """
    # Grab environment variables
    load_dotenv()   
    # Set up log for debugging
    logger = logging.getLogger()
    # Fetch URL
    url = os.getenv("WEATHER_URL")
    if not url:
        logger.error("URL not found in environment variables.")
        return None
    
    # Retrieve data by parsing the HTML of page with BeautifulSoup
    # Before fetching, have to set headers for request
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        # The request
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            logger.error(f"Failed to retrieve webpage: Status code {response.status_code}")
            return None
            
        # Parse HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.select_one('table.data')
        
        if not table:
            logger.error("Table not found on the webpage.")
            return None
        
        # Get current month and year for date formatting
        current_month = datetime.now().month
        current_year = datetime.now().year        
        
        # These are the index locations from the parsed data we want to extract.
        rainfall_index = 4   # Daily rainfall
        hpa_am_index = 15    # Morning hPa
        hpa_pm_index = 21    # Afternoon hPa       
        rows: List[Dict[str, Any]] = []
        data_rows = table.select('tbody tr')
        
        # Process data variables
        for tr in data_rows:
            cells = tr.select('th, td')
            if not cells:
                continue
        
        # Now we get two pressure readings, rainfall, and store them with date and time.        
            # Get date value
            date_value = cells[0].get_text(strip=True)          
            # Get back an unwanted summary row. Ditch this (and any non-integer dates)
            if date_value in ['Mean', 'Lowest', 'Highest', 'Total'] or not date_value.isdigit():
                continue               
            # Set format for date
            formatted_date = f"{int(date_value):02d}/{current_month:02d}/{current_year}"
            
            # Get rainfall value (once per day, not time-specific)
            rainfall_value = None
            if len(cells) > rainfall_index:
                rainfall_text = cells[rainfall_index].get_text(strip=True)
                try:
                    if rainfall_text:
                        rainfall_value = float(rainfall_text)
                except ValueError:
                    pass  # Keep as None if not a valid number
            
            # Initiate a list. Will store data as sets of tuples
            readings = []           
            # Get the morning reading (& remove any spaces)
            if len(cells) > hpa_am_index:
                hpa_am = cells[hpa_am_index].get_text(strip=True)
                try:
                    # If valid, stitch list together with tuples 
                    if hpa_am and float(hpa_am):
                        readings.append(('12:00:00', hpa_am))
                except ValueError:
                    pass
            
            # Jajaja, PM now
            if len(cells) > hpa_pm_index:
                hpa_pm = cells[hpa_pm_index].get_text(strip=True)
                try:
                    if hpa_pm and float(hpa_pm):
                        readings.append(('18:00:00', hpa_pm))
                except ValueError:
                    pass
            
            # Build the dictionary. 
            for time_val, hpa_val in readings:
                rows.append({
                    'Date': formatted_date,
                    'Time': time_val,
                    'hPa': hpa_val,
                    'Rainfall': rainfall_value 
                })
        
        # Create DataFrame (if we have data)
        if rows:
            df = pd.DataFrame(rows)
            # hPa to numeric 
            df['hPa'] = pd.to_numeric(df['hPa'], errors='coerce')
            df['Rainfall'] = pd.to_numeric(df['Rainfall'], errors='coerce')
            # Drop variables with NaNs in hPa only (rainfall can be 0 or missing)
            df = df.dropna(subset=['hPa'])
            #logger.info(f"Extracted {len(df)} barometric pressure readings from weather station")
            return df
        else:
            logger.warning("No valid barometric pressure readings found")
            return None
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error while retrieving weather data: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error while scraping weather data: {str(e)}", exc_info=True)
        return None