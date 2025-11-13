"""
This script processes raw logger data, validates it, and prepares it for database upload.

CONFIGURATION:
Gets the logger data from SITES_CONFIG environment variable (JSON format).
Each site needs the following fields:

- display_name: Human-readable site name
- nav_option: Portal navigation option ID
- depth_conversion_type: How to convert raw depth readings to metres
    * 'metres' - Data already in metres, no conversion needed 
    * 'default' - Convert from feet to metres (multiply by 0.3048)
    * 'divide_by_100' - Divide raw value by 100 to get metres
- pressure_unit: 'hpa' for hectopascals or 'psi' for pounds per square inch
- enabled: true/false to enable/disable processing
- level_channel_id: Channel ID for depth data
- baro_channel_id: Channel ID for barometric pressure data
"""

import os
import pandas as pd
import numpy as np
import datetime
from dateutil.relativedelta import relativedelta
import logging
from logging import FileHandler
from typing import Optional, Tuple, List
import weatherStation
import json
import csv

# --- Global Configuration ---
SITE_CONFIG = {}
EXPECTED_SITES = set()
# Iniatiate a logger object
logger_instance = logging.getLogger('data_validation')

# --- Load Site Configuration ---
def load_site_config():
    global SITE_CONFIG, EXPECTED_SITES
    logger_instance.info("Loading site configuration from environment variables.")
    try:
        # Load sites from .env
        sites_config_json = os.getenv('SITES_CONFIG', '{}')
        SITE_CONFIG = json.loads(sites_config_json)
        
        if not isinstance(SITE_CONFIG, dict):
            logger_instance.warning("SITES_CONFIG env variable not a valid JSON dictionary. Using empty config.")
            SITE_CONFIG = {}
        
        # Only include enabled sites in EXPECTED_SITES
        EXPECTED_SITES = set()
        for site_id, config in SITE_CONFIG.items():
            if config.get('enabled', False):
                EXPECTED_SITES.add(site_id)
            else:
                logger_instance.info(f"Site {site_id} is disabled, not expecting data file.")
        
        logger_instance.info(f"Loaded configuration for {len(EXPECTED_SITES)} expected enabled sites.")
        
        # Log which sites are expected for transparency
        if EXPECTED_SITES:
            logger_instance.info(f"Expected enabled sites: {', '.join(sorted(EXPECTED_SITES))}")
        else:
            logger_instance.warning("No enabled sites found in SITES_CONFIG")
        
    except json.JSONDecodeError:
        logger_instance.error("Failed to parse SITES_CONFIG env variable. Using empty config.", exc_info=True)
        SITE_CONFIG = {}
        EXPECTED_SITES = set()
    except Exception as e:
        logger_instance.error(f"Unexpected error loading site configuration: {e}", exc_info=True)
        SITE_CONFIG = {}
        EXPECTED_SITES = set()

def format_datetime_separated(dt_series: pd.Series) -> pd.Series:
    """
    Format datetime by separating date and time components, then joining with single space.
    Trying avoid Windows strftime spacing issues.
    """
    def format_single_datetime(dt):
        if pd.isna(dt):
            return ""
        # Format date part
        date_part = dt.strftime('%d/%m/%Y')
        # Manually format time part to avoid strftime AM/PM spacing issues
        hour_24 = dt.hour
        minute = dt.minute
        second = dt.second
        # Convert to 12-hour format
        if hour_24 == 0:
            hour_12 = 12
            am_pm = 'AM'
        elif hour_24 < 12:
            hour_12 = hour_24
            am_pm = 'AM'
        elif hour_24 == 12:
            hour_12 = 12
            am_pm = 'PM'
        else:
            hour_12 = hour_24 - 12
            am_pm = 'PM'
        # Format (again) time part. Please god let this work 
        time_part = f"{hour_12}:{minute:02d}:{second:02d} {am_pm}"
        # Join with a single space
        return f"{date_part} {time_part}"
    return dt_series.apply(format_single_datetime)

def extract_datetime_from_formula(formula_string: str) -> str:
    """
    Extract clean datetime string from Excel formula format.
    Converts '="11/07/2025 6:00:00 AM"' back to '11/07/2025 6:00:00 AM'
    """
    if pd.isna(formula_string) or formula_string == "":
        return ""
    # Remove Excel formula wrapper: '="..."' -> '...'
    if formula_string.startswith('="') and formula_string.endswith('"'):
        return formula_string[2:-1] # Remove first 2 chars and last char
    else:
        # If it's not wrapped in formula, return as-is
        return str(formula_string)

# --- Setup/configure data_validation logger object ---
def setup_logging(log_file_path: Optional[str] = None, enable_file_logging: bool = False) -> logging.Logger:
    """Sets up the logger for the data validation process."""
    logger = logger_instance
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.setLevel(logging.WARNING) # (choose .INFO or .DEBUG when things break)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.WARNING)
    logger.addHandler(console_handler)
    # File Handler
    if enable_file_logging and log_file_path:
        try:
            log_dir = os.path.dirname(log_file_path)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)
            file_handler = FileHandler(log_file_path, mode='w') # Overwrites each run
            file_handler.setFormatter(formatter)
            # Switch to DEBUG if specifically needed
            file_handler.setLevel(logging.WARNING)
            logger.addHandler(file_handler)
            logger.info(f"File logging enabled to: {log_file_path}")
        except Exception as e:
            logger.error(f"Failed to set up file logging to {log_file_path}: {e}", exc_info=True)
    return logger

# --- Utility Function ---
def convert_to_numeric(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Make listed columns numeric (errors become NaN)."""
    df_copy = df.copy()
    for col in columns:
        if col in df_copy.columns:
            df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
        else:
            logger_instance.warning(f"Column '{col}' not found for numeric conversion.")
    return df_copy

# --- Core Data Processing Functions ---
def get_bom_baro_data(logger: logging.Logger) -> Optional[pd.DataFrame]:
    """
    Retrieves and processes barometric data from weatherStation.
    Returns a DataFrame ready for merging or None on failure.
    """
    logger.info("Retrieving weather station barometric data")
    try:
        # Call weatherStation script
        baro_data = weatherStation.scrape_weather_data()
        if baro_data is None:
            logger.warning("Weather station scrape returned None.")
            return None
        if baro_data.empty:
            logger.warning("Weather station scrape returned an empty DataFrame.")
            return None
        required_cols = {'Date', 'Time', 'hPa'}
        if not required_cols.issubset(baro_data.columns):
            logger.error(f"Scraped weather data missing required columns ({required_cols}).")
            return None
        try:
            baro_data['Date Time (dd/mm/yyyy hh24:mi:ss)'] = pd.to_datetime(
                baro_data['Date'] + ' ' + baro_data['Time'],
                format='%d/%m/%Y %H:%M:%S',
                errors='coerce'
            )
            baro_data = (baro_data
                         .dropna(subset=['Date Time (dd/mm/yyyy hh24:mi:ss)'])
                         [['Date Time (dd/mm/yyyy hh24:mi:ss)', 'hPa']]
                         .rename(columns={'hPa': 'BomBaro'})
                         .copy()
                        )
            baro_data['BomBaro'] = pd.to_numeric(baro_data['BomBaro'], errors='coerce')
            baro_data = baro_data.dropna(subset=['BomBaro'])
            if baro_data.empty:
                logger.warning("BoM data: No rows remaining after initial processing.")
                return None
            # Just keep first entry for each date (remove any multiple observations in a day)
            baro_data = baro_data.sort_values('Date Time (dd/mm/yyyy hh24:mi:ss)')
            date_only = baro_data['Date Time (dd/mm/yyyy hh24:mi:ss)'].dt.date
            baro_data = baro_data.loc[~date_only.duplicated(keep='first')].copy()
            if baro_data.empty:
                logger.warning("BoM data: No rows remaining after daily deduplication.")
                return None
            logger.info(f"Successfully retrieved and processed {len(baro_data)} BoM data rows.")
            return baro_data
        except Exception as e:
            logger.error(f"Error processing retrieved BoM data: {e}", exc_info=True)
            return None
    except Exception as e:
        logger.error(f"Error calling or handling weatherStation.scrape_weather_data(): {e}", exc_info=True)
        return None

# Handles the flagging of bad or missing data (notifies with a comment)
def create_placeholder_data(site_name: str, reason: str, timestamp_for_entry: Optional[datetime.datetime] = None) -> pd.DataFrame:
    """
    Creates a row for a site with missing or invalid data.
    Call Hierarchy for {reason}:
        - Called by `process_site_file()` when:
            - "CSV file was empty"
            - "No valid Date/Time entries found"
            - "No valid numeric data found"
            - "No data found with Level(RAW)[Main Buffer] (ft) >= 0"
            - "Empty file" (from specific exception)
        - Called by `consolidate_csv_files()` if:
            - "Site CSV file not found" (expected files not present)
            - "No telemetry data received for [Month Year]" (for stale data)
    Args:
        site_name (str): The name of the sample point (site).
        reason (str): A descriptive string explaining why the placeholder is needed.
                      This is provided by the calling function.
    Returns:
        pd.DataFrame: A DataFrame containing one row with placeholder values and the
                      reason embedded in the 'OTHER - Comments - Text' column.
    """
    logger_instance.warning(f"Creating placeholder data for site '{site_name}': {reason}")
    # get date at time of running script. Done to filter out any old data
    current_time = timestamp_for_entry if timestamp_for_entry is not None else datetime.datetime.now()
    placeholder_datetime_entry = current_time.strftime('%d/%m/%Y 00:00:00')
    placeholder_datetime_comment = current_time.strftime('%d/%m/%Y %H:%M:%S')
    placeholder_df = pd.DataFrame({
        'Sample Point': [site_name],
        'Date Time (dd/mm/yyyy hh24:mi:ss)': [placeholder_datetime_entry],
        'Depth(m)raw': [np.nan],
        'Barometric Pressure(RAW)[Main Buffer] (hPa)': [np.nan],
        # reason variable passed in to provide the why
        'OTHER - Comments - Text': [f"{reason} on {placeholder_datetime_comment}"]
    })
    return placeholder_df

def detect_level_column(columns: List[str]) -> Optional[str]:
    """
    Detect the level column flexibly
    """
    candidates = [
        'Level in metres (m)',
        'Level(RAW)[Main Buffer] (ft)',
        'Level(RAW)[Main Buffer] (m)',
        'Level(RAW) (m)',
        'Level (m)'
    ]
    
    for cand in candidates:
        if cand in columns:
            return cand
    
    # Fallback: any column containing 'Level'
    level_cols = [c for c in columns if 'Level' in c.lower()]
    if level_cols:
        logger_instance.warning(f"Using fallback level column: {level_cols[0]}")
        return level_cols[0]
    
    return None

def detect_pressure_column(columns: List[str], pressure_unit: str) -> Optional[str]:
    """
    Detect pressure column based on unit
    """
    if pressure_unit.lower() == 'psi':
        candidates = ['Pressure(RAW)[Main Buffer] (PSI)', 'Pressure(RAW) [Main Buffer] (PSI)']
    else:
        candidates = ['Barometric Pressure(RAW)[Main Buffer] (hPa)', 'Barometric Pressure(RAW) [Main Buffer] (hPa)']
    
    for cand in candidates:
        if cand in columns:
            return cand
    
    pressure_cols = [c for c in columns if 'Pressure' in c]
    if pressure_cols:
        logger_instance.warning(f"Using fallback pressure column: {pressure_cols[0]}")
        return pressure_cols[0]
    
    return None

def process_site_file(file_path: str, logger: logging.Logger) -> Tuple[Optional[pd.DataFrame], str, bool]:
    """
    Process individual site CSV file with enhanced validation.
    Returns (processed_df, site_id, has_valid_data)
    """
    try:
        site_id = os.path.basename(file_path).split('.')[0]
        if site_id not in SITE_CONFIG:
            logger.warning(f"Site {site_id} not in config, skipping.")
            return None, site_id, False
        
        config = SITE_CONFIG[site_id]
        depth_conversion_type = config.get('depth_conversion_type', 'default')
        pressure_unit = config.get('pressure_unit', 'hpa')
        
        logger.info(f"Processing {site_id} ({config.get('display_name', site_id)})")
        logger.info(f"  Depth conversion: {depth_conversion_type}")
        logger.info(f"  Pressure unit: {pressure_unit}")
        
        df = pd.read_csv(file_path)
        if df.empty:
            logger.warning(f"{site_id}: CSV file was empty")
            return create_placeholder_data(site_id, "CSV file was empty"), site_id, False
        
        # Standardise columns
        df.columns = df.columns.str.strip()
        
        # Detect columns
        level_column = detect_level_column(df.columns)
        if level_column is None:
            logger.error(f"{site_id}: No level column detected")
            return create_placeholder_data(site_id, "No level column detected"), site_id, False
        
        pressure_column = detect_pressure_column(df.columns, pressure_unit)
        
        # Create Date Time column
        if 'Date' in df.columns and 'Time' in df.columns:
            # Normalise Time to have leading zeros
            df['Time'] = df['Time'].apply(lambda x: ':'.join(part.zfill(2) for part in x.split(':')))
            df['Date Time (dd/mm/yyyy hh24:mi:ss)'] = df['Date'] + ' ' + df['Time']
        else:
            logger.error(f"{site_id}: Missing Date/Time columns")
            return create_placeholder_data(site_id, "Missing Date/Time columns"), site_id, False
        
        # Convert to datetime
        df['Date Time (dd/mm/yyyy hh24:mi:ss)'] = pd.to_datetime(
            df['Date Time (dd/mm/yyyy hh24:mi:ss)'], 
            format='%d/%m/%Y %H:%M:%S', 
            errors='coerce'
        )
        
        invalid_dates = df['Date Time (dd/mm/yyyy hh24:mi:ss)'].isna().sum()
        if invalid_dates > 0:
            logger.warning(f"{site_id}: {invalid_dates} invalid dates removed")
            df = df.dropna(subset=['Date Time (dd/mm/yyyy hh24:mi:ss)'])
        
        if df.empty:
            logger.warning(f"{site_id}: No valid Date/Time entries found")
            return create_placeholder_data(site_id, "No valid Date/Time entries found"), site_id, False
        
        # Standardise level to Depth(m)raw
        df['Depth(m)raw'] = pd.to_numeric(df[level_column], errors='coerce')
        
        # Apply conversion
        if depth_conversion_type == 'metres':
            pass  # Already in metres
        elif depth_conversion_type == 'default':
            df['Depth(m)raw'] *= 0.3048
        elif depth_conversion_type == 'divide_by_100':
            df['Depth(m)raw'] /= 100
        else:
            logger.warning(f"{site_id}: Unknown conversion type, no conversion applied")
        
        invalid_depth = df['Depth(m)raw'].isna().sum()
        if invalid_depth > 0:
            logger.warning(f"{site_id}: {invalid_depth} invalid depth values set to NaN")
        
        # Handle pressure
        if pressure_column:
            if pressure_unit.lower() == 'psi':
                df['Pressure(RAW)[Main Buffer] (PSI) - Original'] = pd.to_numeric(df[pressure_column], errors='coerce')
                df['Barometric Pressure(RAW)[Main Buffer] (hPa)'] = df['Pressure(RAW)[Main Buffer] (PSI) - Original'] * 68.9476
            else:
                df['Barometric Pressure(RAW)[Main Buffer] (hPa)'] = pd.to_numeric(df[pressure_column], errors='coerce')
        else:
            logger.warning(f"{site_id}: No pressure column found")
            df['Barometric Pressure(RAW)[Main Buffer] (hPa)'] = np.nan
        
        # Set sample point
        df['Sample Point'] = site_id
        
        # Keep only necessary columns
        keep_cols = [
            'Sample Point',
            'Date Time (dd/mm/yyyy hh24:mi:ss)',
            'Depth(m)raw',
            'Barometric Pressure(RAW)[Main Buffer] (hPa)'
        ]
        if 'Pressure(RAW)[Main Buffer] (PSI) - Original' in df.columns:
            keep_cols.append('Pressure(RAW)[Main Buffer] (PSI) - Original')
        
        df = df[keep_cols]
        
        # Check for valid data
        has_valid_data = df['Depth(m)raw'].notna().any() or df['Barometric Pressure(RAW)[Main Buffer] (hPa)'].notna().any()
        
        if not has_valid_data:
            logger.warning(f"{site_id}: No valid numeric data found")
            return create_placeholder_data(site_id, "No valid numeric data found"), site_id, False
        
        logger.info(f"{site_id}: Processed {len(df)} rows")
        return df, site_id, True
    
    except pd.errors.EmptyDataError:
        logger.warning(f"{site_id}: Empty file")
        return create_placeholder_data(site_id, "Empty file"), site_id, False
    except Exception as e:
        logger.error(f"{site_id}: Error - {str(e)}", exc_info=True)
        return None, site_id, False

def calculate_adjusted_depth(
    df: pd.DataFrame, 
    water_density: float,
    logger: logging.Logger,
    reference_density: float = 1000.0
) -> pd.DataFrame:
    """
    Calculate adjusted depth using formula.
    """
    df_copy = df.copy()
    df_copy['Depth(m)adjusted'] = np.nan
    
    # Valid rows mask for adjustment
    valid_mask = (
        df_copy['Depth(m)raw'].notna() & 
        (df_copy['Depth(m)raw'] > 0.3) &
        df_copy['Barometric Pressure(RAW)[Main Buffer] (hPa)'].notna() &
        df_copy['BomBaro'].notna() &
        (abs(df_copy['BomBaro'] - df_copy['Barometric Pressure(RAW)[Main Buffer] (hPa)']) <= 50) &
        (abs(df_copy['BomBaro'] - df_copy['Barometric Pressure(RAW)[Main Buffer] (hPa)']) > 5)
    )
    
    if valid_mask.any():
        valid_df = df_copy[valid_mask]
        
        pressure_diff = valid_df['BomBaro'] - valid_df['Barometric Pressure(RAW)[Main Buffer] (hPa)']
        adjusted_depth = (pressure_diff * 100) / (9.81 * water_density) * (reference_density / 1000)
        
        df_copy.loc[valid_mask, 'Depth(m)adjusted'] = adjusted_depth
        
        num_adjusted = valid_mask.sum()
        logger.info(f"Calculated adjusted depth for {num_adjusted} rows")
        
        # Handle negative adjusted depths
        negative_adjusted = (df_copy['Depth(m)adjusted'] < 0) & df_copy['Depth(m)adjusted'].notna()
        if negative_adjusted.any():
            num_negative = negative_adjusted.sum()
            logger.warning(f"Found {num_negative} negative adjusted depths, setting to NaN")
            df_copy.loc[negative_adjusted, 'Depth(m)adjusted'] = np.nan
    
    else:
        logger.warning("No valid rows for adjusted depth calculation")
    
    # Handle skipped adjustments
    shallow_mask = (
        df_copy['Depth(m)raw'].notna() & 
        (df_copy['Depth(m)raw'] > 0) &
        (df_copy['Depth(m)raw'] <= 0.3)
    )
    
    large_diff_mask = (
        df_copy['Depth(m)raw'].notna() & 
        (abs(df_copy['BomBaro'] - df_copy['Barometric Pressure(RAW)[Main Buffer] (hPa)']) > 50) &
        df_copy['BomBaro'].notna() &
        df_copy['Barometric Pressure(RAW)[Main Buffer] (hPa)'].notna()
    )
    
    small_diff_mask = (
        df_copy['Depth(m)raw'].notna() & 
        (abs(df_copy['BomBaro'] - df_copy['Barometric Pressure(RAW)[Main Buffer] (hPa)']) <= 5) &
        df_copy['BomBaro'].notna() &
        df_copy['Barometric Pressure(RAW)[Main Buffer] (hPa)'].notna()
    )
    
    no_bom_mask = (
        df_copy['Depth(m)raw'].notna() & 
        (df_copy['Depth(m)raw'] > 0) &
        df_copy['BomBaro'].isna()
    )
    
    no_logger_baro_mask = (
        df_copy['Depth(m)raw'].notna() & 
        (df_copy['Depth(m)raw'] > 0) &
        df_copy['Barometric Pressure(RAW)[Main Buffer] (hPa)'].isna()
    )
    
    skip_masks = {
        'shallow': shallow_mask,
        'large_diff': large_diff_mask,
        'small_diff': small_diff_mask,
        'no_bom': no_bom_mask,
        'no_logger_baro': no_logger_baro_mask
    }
    
    skip_adjust_mask = shallow_mask | large_diff_mask | small_diff_mask | no_bom_mask | no_logger_baro_mask
    
    # Set adjusted to raw for skipped
    df_copy.loc[skip_adjust_mask, 'Depth(m)adjusted'] = df_copy.loc[skip_adjust_mask, 'Depth(m)raw']
    
    # Add comments if no existing comment
    comment_dict = {
        'shallow': "Shallow depth: no adjustment applied",
        'large_diff': "Large barometric difference observed: no adjustment applied",
        'small_diff': "Small barometric difference: no adjustment applied (possible sensor noise)",
        'no_bom': "No weather station data: Adjustments can not be applied",
        'no_logger_baro': "No AquaTroll pressure data.  Adjustments can not be applied"
    }
    
    for reason, mask in skip_masks.items():
        apply_comment = mask & (df_copy['OTHER - Comments - Text'].fillna('').str.strip() == '')
        if apply_comment.any():
            df_copy.loc[apply_comment, 'OTHER - Comments - Text'] = comment_dict[reason]
            logger.info(f"Applied '{comment_dict[reason]}' to {apply_comment.sum()} rows")
    
    return df_copy

def consolidate_comments(df: pd.DataFrame) -> pd.DataFrame:
    """
    Consolidate consecutive rows with comments and no adjusted depth to a single row per group.
    """
    if df.empty:
        return df
    
    df = df.sort_values(['Sample Point', 'Date Time (dd/mm/yyyy hh24:mi:ss)'])
    
    # Identify rows that are comment-only (no adjusted, has comment)
    df['is_comment_only'] = df['Depth(m)adjusted'].isna() & (df['OTHER - Comments - Text'].fillna('').str.strip() != '')
    
    # Detect changes in is_comment_only or comment text or site
    df['comment_text'] = df['OTHER - Comments - Text'].fillna('')
    change = (
        (df['Sample Point'] != df['Sample Point'].shift()) |
        (df['is_comment_only'] != df['is_comment_only'].shift()) |
        (df['comment_text'] != df['comment_text'].shift())
    )
    df['group'] = change.cumsum()
    
    aggregated = []
    for _, group in df.groupby('group'):
        if group['is_comment_only'].iloc[0]:
            # For comment groups, take the first row
            first_row = group.iloc[0].copy()
            aggregated.append(first_row)
        else:
            # For data groups, keep all rows
            aggregated.extend(group.to_dict('records'))
    
    consolidated_df = pd.DataFrame(aggregated)
    consolidated_df = consolidated_df.drop(columns=['is_comment_only', 'comment_text', 'group'], errors='ignore')
    
    return consolidated_df

# --- Primary Function ---
def consolidate_csv_files(
    input_folder: str,
    output_file: str,
    log_file: Optional[str] = None,
    enable_file_logging: bool = False,
    water_density: float = 1000.0,
    reference_density: float = 1000.0
) -> None:
    """
    Main function to consolidate site data, merge weather data, apply comments,
    calculate adjusted depth, and save output files.
    """
    logger = setup_logging(log_file, enable_file_logging)
    script_run_time = datetime.datetime.now()
    logger.info(f"Starting at {script_run_time.strftime('%Y-%m-%d %H:%M:%S')}")
    load_site_config()
    if not os.path.isdir(input_folder):
        logger.error(f"Input folder not found: {input_folder}")
        return
    logger.info(f"Input folder: {input_folder}")
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            logger.info(f"Created output directory: {output_dir}")
        except Exception as e:
            logger.error(f"Failed to create output directory: {e}", exc_info=True)
            return
    logger.info(f"Main output file: {output_file}")
    greater_pbo_output_file = os.path.join(output_dir, 'SWLVLGenericTemplate_greaterPBOPools.csv')
    logger.info(f"Secondary output: {greater_pbo_output_file}")
    try:
        all_files = os.listdir(input_folder)
        csv_files = [f for f in all_files if f.lower().endswith('.csv') and not f.lower().startswith('baro')]
        logger.info(f"Found {len(csv_files)} CSV files.")
    except Exception as e:
        logger.error(f"Error listing files: {e}", exc_info=True)
        return
    processed_sites = set()
    sites_with_data = set()
    sites_with_placeholder = set()
    processing_errors = []
    consolidated_data = []
    for filename in csv_files:
        file_path = os.path.join(input_folder, filename)
        site_df, site_name, has_valid_data = process_site_file(file_path, logger)
        processed_sites.add(site_name)
        if site_df is not None:
            consolidated_data.append(site_df)
            if has_valid_data:
                sites_with_data.add(site_name)
            else:
                sites_with_placeholder.add(site_name)
        else:
            processing_errors.append((site_name, "Processing failed"))
    missing_files = EXPECTED_SITES - processed_sites
    if missing_files:
        logger.warning(f"Missing files: {', '.join(sorted(missing_files))}")
        for site_name in missing_files:
            placeholder_df = create_placeholder_data(site_name, "Site CSV file not found", script_run_time)
            consolidated_data.append(placeholder_df)
            sites_with_placeholder.add(site_name)
    if not consolidated_data:
        logger.error("No data consolidated. No output files created.")
        return
    try:
        logger.info("Concatenating all site data...")
        final_df = pd.concat(consolidated_data, ignore_index=True)
        columns_to_drop = ['Date', 'Time', 'Level(RAW)[Main Buffer] (ft)', 'Pressure(RAW)[Main Buffer] (PSI)']
        existing_columns_to_drop = [col for col in columns_to_drop if col in final_df.columns]
        if existing_columns_to_drop:
            logger.info(f"Removing duplicate raw columns: {existing_columns_to_drop}")
            final_df = final_df.drop(columns=existing_columns_to_drop)
        final_df['Date Time (dd/mm/yyyy hh24:mi:ss)'] = pd.to_datetime(
            final_df['Date Time (dd/mm/yyyy hh24:mi:ss)'], errors='coerce'
        )
        invalid_dates = final_df['Date Time (dd/mm/yyyy hh24:mi:ss)'].isna().sum()
        if invalid_dates > 0:
            logger.warning(f"Removed {invalid_dates} rows with invalid Date/Time.")
            final_df.dropna(subset=['Date Time (dd/mm/yyyy hh24:mi:ss)'], inplace=True)
        if final_df.empty:
            logger.error("No valid data after Date/Time validation.")
            return
        final_df = final_df.sort_values(['Sample Point', 'Date Time (dd/mm/yyyy hh24:mi:ss)'])
        site_date_key = final_df['Sample Point'].astype(str) + final_df['Date Time (dd/mm/yyyy hh24:mi:ss)'].dt.date.astype(str)
        final_df = final_df.loc[~site_date_key.duplicated(keep='first')].copy()
        logger.info(f"Data ready for BoM merge: {final_df.shape[0]} rows.")
    except Exception as e:
        logger.error(f"Error during concatenation/deduplication: {e}", exc_info=True)
        return
    bom_baro_data = get_bom_baro_data(logger)
    final_df['BomBaro'] = np.nan
    if bom_baro_data is not None and not bom_baro_data.empty:
        logger.info("Merging BoM data...")
        try:
            final_df['merge_date'] = final_df['Date Time (dd/mm/yyyy hh24:mi:ss)'].dt.date
            bom_baro_data['merge_date'] = bom_baro_data['Date Time (dd/mm/yyyy hh24:mi:ss)'].dt.date
            final_df = pd.merge(
                final_df,
                bom_baro_data[['merge_date', 'BomBaro']],
                on='merge_date',
                how='left',
                suffixes=('', '_bom')
            )
            if 'BomBaro_bom' in final_df.columns:
                final_df['BomBaro'] = final_df['BomBaro'].fillna(final_df['BomBaro_bom'])
                final_df = final_df.drop(columns=['merge_date', 'BomBaro_bom'], errors='ignore')
            elif 'merge_date' in final_df.columns:
                final_df = final_df.drop(columns=['merge_date'], errors='ignore')
            bom_data_count = final_df['BomBaro'].notna().sum()
            logger.info(f"Merge complete. {bom_data_count} rows have BoM data.")
        except Exception as e:
            logger.error(f"Error during BoM merge: {e}", exc_info=True)
            if 'merge_date' in final_df.columns:
                final_df = final_df.drop(columns=['merge_date'], errors='ignore')
    else:
        logger.warning("Weather station data not available. Merge skipped.")
    if 'merge_date' in final_df.columns:
        final_df = final_df.drop(columns=['merge_date'], errors='ignore')
    logger.info("Applying comments based on data staleness and values...")
    if 'OTHER - Comments - Text' not in final_df.columns:
        final_df['OTHER - Comments - Text'] = ''
    final_df['OTHER - Comments - Text'] = final_df['OTHER - Comments - Text'].fillna('')
    original_comment_mask = final_df['OTHER - Comments - Text'] != ''
    final_df_original_placeholders = final_df[original_comment_mask].copy()
    final_df_data_to_process = final_df[~original_comment_mask].copy()
    if final_df_data_to_process.empty and not final_df_original_placeholders.empty:
        logger.info("Only placeholder data found. Skipping staleness/zero checks.")
        final_df_commented = final_df_original_placeholders
    elif final_df_data_to_process.empty and final_df_original_placeholders.empty:
        logger.error("No data remaining before comment logic.")
        return
    else:
        two_months_ago = script_run_time - relativedelta(months=2)
        current_month_year = script_run_time.strftime("%B %Y")
        if not final_df_data_to_process.empty:
            latest_indices = final_df_data_to_process.groupby('Sample Point')['Date Time (dd/mm/yyyy hh24:mi:ss)'].idxmax()
            latest_entries_df = final_df_data_to_process.loc[latest_indices]
            stale_mask = latest_entries_df['Date Time (dd/mm/yyyy hh24:mi:ss)'] < two_months_ago
            stale_indices_to_update = latest_entries_df[stale_mask].index
            if not stale_indices_to_update.empty:
                stale_site_names = final_df_data_to_process.loc[stale_indices_to_update, 'Sample Point'].unique().tolist()
                logger.info(f"Stale sites: {', '.join(stale_site_names)}")
                stale_reason = f"No telemetry data received for {current_month_year}"
                final_df_data_to_process.loc[stale_indices_to_update, 'OTHER - Comments - Text'] = stale_reason
                logger.info(f"Applied stale comment to {len(stale_indices_to_update)} sites.")
        if not final_df_data_to_process.empty:
            final_df_data_to_process['Depth(m)raw'] = pd.to_numeric(final_df_data_to_process['Depth(m)raw'], errors='coerce')
            zero_neg_mask = (final_df_data_to_process['Depth(m)raw'] <= 0) & final_df_data_to_process['Depth(m)raw'].notna()
            zero_neg_comment = "There is an equipment issue or the pool is dry"
            current_comment_empty = final_df_data_to_process['OTHER - Comments - Text'].isna() | (final_df_data_to_process['OTHER - Comments - Text'] == '')
            apply_comment_mask = zero_neg_mask & current_comment_empty
            final_df_data_to_process.loc[apply_comment_mask, 'OTHER - Comments - Text'] = zero_neg_comment
            num_zero_comments_applied = apply_comment_mask.sum()
            if num_zero_comments_applied > 0:
                logger.info(f"Applied zero/negative depth comment to {num_zero_comments_applied} rows.")
        all_processed_frames = []
        if not final_df_original_placeholders.empty:
            all_processed_frames.append(final_df_original_placeholders)
        if not final_df_data_to_process.empty:
            all_processed_frames.append(final_df_data_to_process)
        if not all_processed_frames:
            logger.error("No data frames to combine after comment logic.")
            return
        final_df_commented = pd.concat(all_processed_frames, ignore_index=True)
        final_df_commented['Date Time (dd/mm/yyyy hh24:mi:ss)'] = pd.to_datetime(
            final_df_commented['Date Time (dd/mm/yyyy hh24:mi:ss)'], errors='coerce'
        )
        final_df_commented.dropna(subset=['Date Time (dd/mm/yyyy hh24:mi:ss)'], inplace=True)
        if not final_df_commented.empty:
            final_df_commented = final_df_commented.sort_values(['Sample Point', 'Date Time (dd/mm/yyyy hh24:mi:ss)'])
        else:
            logger.error("DataFrame empty after comment processing.")
            return
    if final_df_commented.empty:
        logger.error("final_df_commented empty before adjusted depth.")
        return
    else:
        final_df_processed = calculate_adjusted_depth(final_df_commented, water_density, logger, reference_density)
    # Filter to current month
    current_month = script_run_time.month
    current_year = script_run_time.year
    final_df_processed = final_df_processed[
        (final_df_processed['Date Time (dd/mm/yyyy hh24:mi:ss)'].dt.month == current_month) &
        (final_df_processed['Date Time (dd/mm/yyyy hh24:mi:ss)'].dt.year == current_year)
    ]
    
    # Add placeholders for sites missing in current month
    present_sites = set(final_df_processed['Sample Point'].unique())
    missing_current_sites = EXPECTED_SITES - present_sites
    current_month_year = script_run_time.strftime("%B %Y")
    for site in missing_current_sites:
        placeholder = create_placeholder_data(site, f"No telemetry data received for {current_month_year}", script_run_time)
        final_df_processed = pd.concat([final_df_processed, placeholder], ignore_index=True)
    
    # Sort again
    final_df_processed = final_df_processed.sort_values(['Sample Point', 'Date Time (dd/mm/yyyy hh24:mi:ss)'])
    # Consolidate comment rows
    final_df_processed = consolidate_comments(final_df_processed)
    logger.info(f"Preparing SQL import file: {os.path.basename(greater_pbo_output_file)}")
    try:
        # Include rows with adjusted depth or comments
        pbo_df = final_df_processed[
            final_df_processed['Depth(m)adjusted'].notna() | 
            (final_df_processed['OTHER - Comments - Text'].fillna('').str.strip() != '')
        ].copy()
        if not pbo_df.empty:
            pbo_df.rename(columns={'Depth(m)adjusted': 'LEVEL - DEPTH TO WATER - m (INPUT)'}, inplace=True)
            pbo_column_order = [
                'Sample Point', 'Date Time (dd/mm/yyyy hh24:mi:ss)',
                'LEVEL - DEPTH TO WATER - m (INPUT)', 'LEVEL - WATER LEVEL - mAHD (INPUT)',
                'OTHER - Comments - Text'
            ]
            for col in pbo_column_order:
                if col not in pbo_df.columns: pbo_df[col] = np.nan
            pbo_df = pbo_df[pbo_column_order]
            pbo_df['Date Time (dd/mm/yyyy hh24:mi:ss)'] = format_datetime_separated(pbo_df['Date Time (dd/mm/yyyy hh24:mi:ss)'])
            pbo_df.to_csv(greater_pbo_output_file, index=False, na_rep='')
            logger.info("SQL output saved.")
        else:
            logger.warning("No valid adjusted depth data. SQL file not created.")
    except Exception as e:
        logger.error(f"Error creating SQL file: {e}", exc_info=True)
    logger.info(f"Preparing verification file: {os.path.basename(output_file)}")
    try:
        excel_df = final_df_processed.copy()
        desired_column_order = [
            'Sample Point',
            'Date Time (dd/mm/yyyy hh24:mi:ss)',
            'BomBaro',
            'Barometric Pressure(RAW)[Main Buffer] (hPa)',
            'Pressure(RAW)[Main Buffer] (PSI) - Original',
            'Depth(m)raw',
            'Depth(m)adjusted',
            'OTHER - Comments - Text'
        ]
        final_columns = [col for col in desired_column_order if col in excel_df.columns]
        remaining_cols = [col for col in excel_df.columns if col not in final_columns]
        final_columns.extend(remaining_cols)
        excel_df = excel_df[final_columns]
        excel_df['Date Time (dd/mm/yyyy hh24:mi:ss)'] = format_datetime_separated(excel_df['Date Time (dd/mm/yyyy hh24:mi:ss)'])
        excel_df.to_csv(output_file, index=False, na_rep='', quoting=csv.QUOTE_NONNUMERIC)
        logger.info(f"Verification file saved: {len(excel_df)} rows")
    except Exception as e:
        logger.error(f"Failed to create verification file: {e}", exc_info=True)
    logger.info("Consolidation complete.")
    if missing_files:
        logger.warning(f"Missing files: {', '.join(sorted(missing_files))}")
    if processing_errors:
        logger.error(f"Processing errors: {len(processing_errors)} sites")
        for site, err in processing_errors:
            logger.error(f"  {site}: {err}")
    logger.handlers = []