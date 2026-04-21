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
from typing import Optional, Tuple, List
import weatherStation
import json
import csv

# --- Global Configuration ---
SITE_CONFIG = {}
EXPECTED_SITES = set()
logger = logging.getLogger('data_validation')

# --- Configuration Loading ---
def load_site_config():
    """Load site configuration from environment variables."""
    global SITE_CONFIG, EXPECTED_SITES
    
    logger.info("Loading site configuration from environment variables.")
    sites_config_json = os.getenv('SITES_CONFIG', '{}')
    SITE_CONFIG = json.loads(sites_config_json)
    
    if not isinstance(SITE_CONFIG, dict):
        logger.warning("SITES_CONFIG env variable not a valid JSON dictionary. Using empty config.")
        SITE_CONFIG = {}
    
    # Only include enabled sites
    EXPECTED_SITES = {site_id for site_id, config in SITE_CONFIG.items() 
                      if config.get('enabled', False)}
    
    logger.info(f"Loaded configuration for {len(EXPECTED_SITES)} enabled sites.")
    if EXPECTED_SITES:
        logger.info(f"Expected enabled sites: {', '.join(sorted(EXPECTED_SITES))}")
    else:
        logger.warning("No enabled sites found in SITES_CONFIG")

# --- Logging Setup ---
def setup_logging(log_file: Optional[str] = None) -> logging.Logger:
    """Sets up the logger for the data validation process."""
    logger = logging.getLogger(__name__)
    
    if logger.hasHandlers():
        logger.handlers.clear()
    
    logger.setLevel(logging.WARNING)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Always add console handler
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)
    
    # Only add file handler if log_file is provided
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info(f"File logging enabled to: {log_file}")
    
    return logger

# --- Datetime Formatting ---
def format_datetime_separated(dt_series: pd.Series) -> pd.Series:
    """
    Format datetime by separating date and time components. 
    """
    def format_single(dt):
        if pd.isna(dt):
            return ""
        # Handle case where datetime might already be a string (from placeholder data)
        if isinstance(dt, str):
            return dt
        date_part = dt.strftime('%d/%m/%Y')
        hour = dt.hour
        # Dear old gods and the new pls let this patch the stoopid windows 11 double space bug    
        hour_12 = 12 if hour == 0 else (hour if hour <= 12 else hour - 12)
        am_pm = 'AM' if hour < 12 else 'PM'
        time_part = f"{hour_12:02d}:{dt.minute:02d}:{dt.second:02d} {am_pm}"
        # Please. 
        return f"{date_part} {time_part}"
    
    return dt_series.apply(format_single)

# --- Weather Data ---
def get_bom_baro_data() -> Optional[pd.DataFrame]:
    """Retrieve and process barometric data from weather station."""
    logger.info("Retrieving weather station barometric data")
    
    baro_data = weatherStation.scrape_weather_data()
    if baro_data is None or baro_data.empty:
        logger.warning("Weather station scrape returned no data.")
        return None
    
    required_cols = {'Date', 'Time', 'hPa', 'Rainfall'}
    if not required_cols.issubset(baro_data.columns):
        logger.error(f"Scraped weather data missing required columns ({required_cols}).")
        return None
    
    # Process datetime
    baro_data['Date Time (dd/mm/yyyy hh24:mi:ss)'] = pd.to_datetime(
        baro_data['Date'] + ' ' + baro_data['Time'],
        format='%d/%m/%Y %H:%M:%S',
        errors='coerce'
    )
    
    # Clean and deduplicate
    baro_data = (baro_data
                 .dropna(subset=['Date Time (dd/mm/yyyy hh24:mi:ss)'])
                 [['Date Time (dd/mm/yyyy hh24:mi:ss)', 'hPa', 'Rainfall']]
                 .rename(columns={'hPa': 'BomBaro'})
                 .copy())
    
    baro_data['BomBaro'] = pd.to_numeric(baro_data['BomBaro'], errors='coerce')
    baro_data['Rainfall'] = pd.to_numeric(baro_data['Rainfall'], errors='coerce')
    baro_data = baro_data.dropna(subset=['BomBaro'])
    
    # Keep first entry per day
    baro_data = baro_data.sort_values('Date Time (dd/mm/yyyy hh24:mi:ss)')
    date_only = baro_data['Date Time (dd/mm/yyyy hh24:mi:ss)'].dt.date
    baro_data = baro_data.loc[~date_only.duplicated(keep='first')].copy()
    
    if baro_data.empty:
        logger.warning("BoM data: No rows remaining after processing.")
        return None
    
    logger.info(f"Successfully retrieved {len(baro_data)} BoM data rows.")
    return baro_data

# --- Placeholder Data ---
def create_placeholder_data(site_name: str, reason: str, 
                           timestamp: Optional[datetime.datetime] = None) -> pd.DataFrame:
    """Create a placeholder row for sites with missing/invalid data."""
    logger.warning(f"Creating placeholder data for site '{site_name}': {reason}")
    
    current_time = timestamp or datetime.datetime.now()
    placeholder_datetime = current_time.strftime('%d/%m/%Y 00:00:00')
    comment_datetime = current_time.strftime('%d/%m/%Y %H:%M:%S')
    
    return pd.DataFrame({
        'Sample Point': [site_name],
        'Date Time (dd/mm/yyyy hh24:mi:ss)': [placeholder_datetime],
        'Depth(m)raw': [np.nan],
        'Barometric Pressure(RAW)[Main Buffer] (hPa)': [np.nan],
        'OTHER - Comments - Text': [f"{reason} on {comment_datetime}"]
    })

# --- Column Detection ---
def detect_level_column(columns: List[str]) -> Optional[str]:
    """Detect the level column flexibly."""
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
    
    # Fallback to any column containing 'Level'
    level_cols = [c for c in columns if 'Level' in c.lower()]
    if level_cols:
        logger.warning(f"Using fallback level column: {level_cols[0]}")
        return level_cols[0]
    
    return None

def detect_pressure_column(columns: List[str], pressure_unit: str) -> Optional[str]:
    """Detect pressure column based on unit."""
    if pressure_unit.lower() == 'psi':
        candidates = ['Pressure(RAW)[Main Buffer] (PSI)', 'Pressure(RAW) [Main Buffer] (PSI)']
    else:
        candidates = ['Barometric Pressure(RAW)[Main Buffer] (hPa)', 
                     'Barometric Pressure(RAW) [Main Buffer] (hPa)']
    
    for cand in candidates:
        if cand in columns:
            return cand
    
    # Fallback
    pressure_cols = [c for c in columns if 'Pressure' in c]
    if pressure_cols:
        logger.warning(f"Using fallback pressure column: {pressure_cols[0]}")
        return pressure_cols[0]
    
    return None

# --- Site File Processing ---
def process_site_file(file_path: str) -> Tuple[Optional[pd.DataFrame], str, bool]:
    """
    Process individual site CSV file with enhanced validation.
    Returns (processed_df, site_id, has_valid_data)
    """
    site_id = os.path.basename(file_path).split('.')[0]
    
    if site_id not in SITE_CONFIG:
        logger.warning(f"Site {site_id} not in config, skipping.")
        return None, site_id, False
    
    config = SITE_CONFIG[site_id]
    depth_conversion = config.get('depth_conversion_type', 'default')
    pressure_unit = config.get('pressure_unit', 'hpa')
    
    logger.info(f"Processing {site_id} ({config.get('display_name', site_id)})")
    logger.info(f"  Depth conversion: {depth_conversion}, Pressure unit: {pressure_unit}")
    
    # Read CSV
    df = pd.read_csv(file_path)
    if df.empty:
        logger.warning(f"{site_id}: CSV file was empty")
        return create_placeholder_data(site_id, "CSV file was empty"), site_id, False
    
    df.columns = df.columns.str.strip()
    
    # Detect columns
    level_column = detect_level_column(df.columns)
    if level_column is None:
        logger.error(f"{site_id}: No level column detected")
        return create_placeholder_data(site_id, "No level column detected"), site_id, False
    
    pressure_column = detect_pressure_column(df.columns, pressure_unit)
    
    # Create datetime column
    if 'Date' not in df.columns or 'Time' not in df.columns:
        logger.error(f"{site_id}: Missing Date/Time columns")
        return create_placeholder_data(site_id, "Missing Date/Time columns"), site_id, False
    
    # Normalize time format (add leading zeros)
    df['Time'] = df['Time'].apply(lambda x: ':'.join(part.zfill(2) for part in str(x).split(':')))
    df['Date Time (dd/mm/yyyy hh24:mi:ss)'] = pd.to_datetime(
        df['Date'] + ' ' + df['Time'],
        format='%d/%m/%Y %H:%M:%S',
        errors='coerce'
    )
    
    # Remove invalid dates
    invalid_dates = df['Date Time (dd/mm/yyyy hh24:mi:ss)'].isna().sum()
    if invalid_dates > 0:
        logger.warning(f"{site_id}: {invalid_dates} invalid dates removed")
        df = df.dropna(subset=['Date Time (dd/mm/yyyy hh24:mi:ss)'])
    
    if df.empty:
        logger.warning(f"{site_id}: No valid Date/Time entries found")
        return create_placeholder_data(site_id, "No valid Date/Time entries found"), site_id, False
    
    # Convert depth to metres
    df['Depth(m)raw'] = pd.to_numeric(df[level_column], errors='coerce')
    
    if depth_conversion == 'default':
        df['Depth(m)raw'] *= 0.3048
    elif depth_conversion == 'divide_by_100':
        df['Depth(m)raw'] /= 100
    # 'metres' - no conversion needed
    
    invalid_depth = df['Depth(m)raw'].isna().sum()
    if invalid_depth > 0:
        logger.warning(f"{site_id}: {invalid_depth} invalid depth values")
    
    # Handle pressure
    if pressure_column:
        if pressure_unit.lower() == 'psi':
            df['Pressure(RAW)[Main Buffer] (PSI)'] = pd.to_numeric(df[pressure_column], errors='coerce')
            df['Barometric Pressure(RAW)[Main Buffer] (hPa)'] = df['Pressure(RAW)[Main Buffer] (PSI)'] * 68.9476
        else:
            df['Barometric Pressure(RAW)[Main Buffer] (hPa)'] = pd.to_numeric(df[pressure_column], errors='coerce')
    else:
        logger.warning(f"{site_id}: No pressure column found")
        df['Barometric Pressure(RAW)[Main Buffer] (hPa)'] = np.nan
    
    df['Sample Point'] = site_id
    
    # Keep only necessary columns
    keep_cols = ['Sample Point', 'Date Time (dd/mm/yyyy hh24:mi:ss)', 
                 'Depth(m)raw', 'Barometric Pressure(RAW)[Main Buffer] (hPa)']
    if 'Pressure(RAW)[Main Buffer] (PSI)' in df.columns:
        keep_cols.append('Pressure(RAW)[Main Buffer] (PSI)')
    
    df = df[keep_cols]
    
    # Check for valid data
    has_valid_data = (df['Depth(m)raw'].notna().any() or 
                     df['Barometric Pressure(RAW)[Main Buffer] (hPa)'].notna().any())
    
    if not has_valid_data:
        logger.warning(f"{site_id}: No valid numeric data found")
        return create_placeholder_data(site_id, "No valid numeric data found"), site_id, False
    
    logger.info(f"{site_id}: Processed {len(df)} rows")
    return df, site_id, True

# --- Adjusted Depth Calculation ---
def calculate_adjusted_depth(df: pd.DataFrame, water_density: float = 1000.0,
                            reference_density: float = 1000.0) -> pd.DataFrame:
    """
    Calculate adjusted depth using AquaTroll drift correction formula.
    
    THRESHOLDS:
    - Drift ≤ 5 hPa: No adjustment (sensor noise)
    - 5 < Drift ≤ 20 hPa: Apply correction
    - 20 < Drift ≤ 50 hPa: No adjustment, flag excessive drift
    - Drift > 50 hPa: No adjustment (sensor failure)
    - Depth ≤ 0.3m: No adjustment (shallow water)
    """
    df = df.copy()
    df['Depth(m)adjusted'] = np.nan
    
    # Constants
    CONVERSION_FACTOR = 0.70307
    HPA_TO_PSI = 0.0145038
    
    # Calculate pressure difference
    def get_pressure_diff(row):
        if pd.notna(row.get('Pressure(RAW)[Main Buffer] (PSI)')):
            bom_psi = row['BomBaro'] * HPA_TO_PSI
            logger_psi = row['Pressure(RAW)[Main Buffer] (PSI)']
            return abs(bom_psi - logger_psi) / HPA_TO_PSI
        elif pd.notna(row.get('Barometric Pressure(RAW)[Main Buffer] (hPa)')):
            return abs(row['BomBaro'] - row['Barometric Pressure(RAW)[Main Buffer] (hPa)'])
        return np.nan
    
    df['pressure_diff_hpa'] = df.apply(get_pressure_diff, axis=1)
    
    # Valid adjustment mask (5 < diff ≤ 20 hPa, depth > 0.3m)
    valid_mask = (
        df['Depth(m)raw'].notna() & 
        (df['Depth(m)raw'] > 0.3) &
        df['BomBaro'].notna() &
        df['pressure_diff_hpa'].notna() &
        (df['pressure_diff_hpa'] > 5) &
        (df['pressure_diff_hpa'] <= 20)
    )
    
    # Apply adjustment
    if valid_mask.any():
        valid_df = df[valid_mask]
        
        if 'Pressure(RAW)[Main Buffer] (PSI)' in df.columns:
            bom_psi = valid_df['BomBaro'] * HPA_TO_PSI
            logger_psi = valid_df['Pressure(RAW)[Main Buffer] (PSI)']
            pressure_diff_psi = bom_psi - logger_psi
        else:
            pressure_diff_hpa = valid_df['BomBaro'] - valid_df['Barometric Pressure(RAW)[Main Buffer] (hPa)']
            pressure_diff_psi = pressure_diff_hpa * HPA_TO_PSI
        
        specific_gravity = water_density / reference_density
        depth_correction = (CONVERSION_FACTOR * pressure_diff_psi) / specific_gravity
        adjusted_depth = valid_df['Depth(m)raw'] + depth_correction
        
        df.loc[valid_mask, 'Depth(m)adjusted'] = adjusted_depth
        logger.info(f"Calculated adjusted depth for {valid_mask.sum()} rows")
        
        # Set negative values to NaN
        negative_mask = (df['Depth(m)adjusted'] < 0) & df['Depth(m)adjusted'].notna()
        if negative_mask.any():
            logger.warning(f"Set {negative_mask.sum()} negative adjusted depths to NaN")
            df.loc[negative_mask, 'Depth(m)adjusted'] = np.nan
    
    # For non-adjusted rows, copy raw to adjusted with appropriate comments
    df['OTHER - Comments - Text'] = df.get('OTHER - Comments - Text', '').fillna('')
    
    # Define skip conditions
    shallow = (df['Depth(m)raw'].notna()) & (df['Depth(m)raw'] > 0) & (df['Depth(m)raw'] <= 0.3)
    large_diff = (df['pressure_diff_hpa'] > 50) & df['pressure_diff_hpa'].notna()
    small_diff = (df['pressure_diff_hpa'] <= 5) & df['pressure_diff_hpa'].notna()
    excessive_drift = (df['Depth(m)raw'] > 0.3) & (df['pressure_diff_hpa'] > 20) & (df['pressure_diff_hpa'] <= 50)
    no_bom = (df['Depth(m)raw'] > 0) & df['BomBaro'].isna()
    no_logger_baro = (
        (df['Depth(m)raw'] > 0) &
        df['Barometric Pressure(RAW)[Main Buffer] (hPa)'].isna() &
        df.get('Pressure(RAW)[Main Buffer] (PSI)', pd.Series([np.nan] * len(df))).isna()
    )
    
    skip_conditions = [
        (shallow, "Very shallow depth: no adjustment applied"),
        (large_diff, "Large barometric difference observed: no adjustment applied"),
        (small_diff, "Small barometric difference: no adjustment applied (possible sensor noise)"),
        (excessive_drift, "Large sensor drift detected (>20 hPa): adjusted value formula not applied"),
        (no_bom, "No weather station data: Adjustments can not be applied"),
        (no_logger_baro, "No AquaTroll pressure data. Adjustments can not be applied")
    ]
    
    for condition, comment in skip_conditions:
        apply_mask = condition & (df['OTHER - Comments - Text'].str.strip() == '')
        if apply_mask.any():
            df.loc[apply_mask, 'Depth(m)adjusted'] = df.loc[apply_mask, 'Depth(m)raw']
            df.loc[apply_mask, 'OTHER - Comments - Text'] = comment
            logger.info(f"Applied '{comment}' to {apply_mask.sum()} rows")
    
    return df

# --- Comment Consolidation ---
def consolidate_comments(df: pd.DataFrame) -> pd.DataFrame:
    """Consolidate consecutive comment-only rows to single row per group."""
    if df.empty:
        return df
    
    df = df.sort_values(['Sample Point', 'Date Time (dd/mm/yyyy hh24:mi:ss)'])
    
    # Identify comment-only rows
    df['is_comment'] = df['Depth(m)adjusted'].isna() & (df['OTHER - Comments - Text'].fillna('').str.strip() != '')
    df['comment_text'] = df['OTHER - Comments - Text'].fillna('')
    
    # Create groups
    change = (
        (df['Sample Point'] != df['Sample Point'].shift()) |
        (df['is_comment'] != df['is_comment'].shift()) |
        (df['comment_text'] != df['comment_text'].shift())
    )
    df['group'] = change.cumsum()
    
    # Keep first row of comment groups, all rows of data groups
    aggregated = []
    for _, group in df.groupby('group'):
        if group['is_comment'].iloc[0]:
            aggregated.append(group.iloc[0].to_dict())
        else:
            aggregated.extend(group.to_dict('records'))
    
    result = pd.DataFrame(aggregated)
    return result.drop(columns=['is_comment', 'comment_text', 'group'], errors='ignore')

# --- Output File Creation ---
def save_output_file(df: pd.DataFrame, filepath: str, for_sql: bool = False):
    """Save output file with appropriate formatting."""
    output_df = df.copy()
    
    if for_sql:
        # SQL import file: include adjusted depth or comments
        output_df = output_df[
            output_df['Depth(m)adjusted'].notna() | 
            (output_df['OTHER - Comments - Text'].fillna('').str.strip() != '')
        ].copy()
        
        # Set missing adjusted depth to 0 (dry pools)
        output_df.loc[output_df['Depth(m)adjusted'].isna(), 'Depth(m)adjusted'] = 0.0
        
        # Round depth to 2 decimal places
        output_df['Depth(m)adjusted'] = output_df['Depth(m)adjusted'].round(2)
        
        output_df['OTHER - Comments - Text'] = ''  # Clear comments for SQL
        
        # Rename and reorder columns - put depth in WATER LEVEL column, not DEPTH TO WATER
        output_df.rename(columns={'Depth(m)adjusted': 'LEVEL - WATER LEVEL - mAHD (INPUT)'}, inplace=True)
        columns = [
            'Sample Point', 
            'Date Time (dd/mm/yyyy hh24:mi:ss)',
            'LEVEL - DEPTH TO WATER - m (INPUT)', 
            'LEVEL - WATER LEVEL - mAHD (INPUT)',
            'OTHER - Comments - Text'
        ]
        for col in columns:
            if col not in output_df.columns:
                output_df[col] = np.nan
        output_df = output_df[columns]
    else:
        # Verification file: drop internal columns
        output_df = output_df.drop(columns=['pressure_diff_hpa'], errors='ignore')
        
        # Reorder columns
        desired_order = [
            'Sample Point', 'Date Time (dd/mm/yyyy hh24:mi:ss)',
            'BomBaro', 'Rainfall', 'Barometric Pressure(RAW)[Main Buffer] (hPa)',
            'Pressure(RAW)[Main Buffer] (PSI)', 'Depth(m)raw',
            'Depth(m)adjusted', 'OTHER - Comments - Text'
        ]
        columns = [col for col in desired_order if col in output_df.columns]
        remaining = [col for col in output_df.columns if col not in columns]
        output_df = output_df[columns + remaining]
    
    # Format datetime
    output_df['Date Time (dd/mm/yyyy hh24:mi:ss)'] = format_datetime_separated(
        output_df['Date Time (dd/mm/yyyy hh24:mi:ss)']
    )
    
    # Save
    output_df.to_csv(filepath, index=False, na_rep='', 
                     quoting=csv.QUOTE_NONNUMERIC if not for_sql else csv.QUOTE_MINIMAL)
    logger.info(f"Saved {filepath}: {len(output_df)} rows")

# --- Main Consolidation Function ---
def consolidate_csv_files(input_folder: str, output_file: str, 
                         log_file: Optional[str] = None,
                         water_density: float = 1000.0,
                         reference_density: float = 1000.0) -> None:
    """
    Main function to consolidate site data, merge weather data, 
    calculate adjusted depth, and save output files.
    """
    setup_logging(log_file)
    script_run_time = datetime.datetime.now()
    logger.info(f"Starting at {script_run_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    load_site_config()
    
    if not os.path.isdir(input_folder):
        logger.error(f"Input folder not found: {input_folder}")
        return
    
    # Create output directory
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Output directory: {output_dir}")
    
    sql_output_file = os.path.join(output_dir, 'SWLVLGenericTemplate_greaterPBOPools.csv')
    
    # Process all CSV files
    csv_files = [f for f in os.listdir(input_folder) 
                 if f.lower().endswith('.csv') and not f.lower().startswith('baro')]
    logger.info(f"Found {len(csv_files)} CSV files.")
    
    processed_sites = set()
    consolidated_data = []
    
    for filename in csv_files:
        file_path = os.path.join(input_folder, filename)
        site_df, site_name, has_valid_data = process_site_file(file_path)
        processed_sites.add(site_name)
        if site_df is not None:
            consolidated_data.append(site_df)
    
    # Add placeholders for missing files
    missing_files = EXPECTED_SITES - processed_sites
    if missing_files:
        logger.warning(f"Missing files: {', '.join(sorted(missing_files))}")
        for site_name in missing_files:
            placeholder = create_placeholder_data(site_name, "Site CSV file not found", script_run_time)
            consolidated_data.append(placeholder)
    
    if not consolidated_data:
        logger.error("No data consolidated. Exiting.")
        return
    
    # Concatenate all data
    logger.info("Concatenating all site data...")
    final_df = pd.concat(consolidated_data, ignore_index=True)
    
    # Clean datetime and deduplicate
    final_df['Date Time (dd/mm/yyyy hh24:mi:ss)'] = pd.to_datetime(
        final_df['Date Time (dd/mm/yyyy hh24:mi:ss)'], errors='coerce'
    )
    final_df = final_df.dropna(subset=['Date Time (dd/mm/yyyy hh24:mi:ss)'])
    final_df = final_df.sort_values(['Sample Point', 'Date Time (dd/mm/yyyy hh24:mi:ss)'])
    
    # Deduplicate by site-date
    site_date_key = (final_df['Sample Point'].astype(str) + 
                     final_df['Date Time (dd/mm/yyyy hh24:mi:ss)'].dt.date.astype(str))
    final_df = final_df.loc[~site_date_key.duplicated(keep='first')].copy()
    
    logger.info(f"Data ready for BoM merge: {final_df.shape[0]} rows")
    
    # Merge weather data
    bom_data = get_bom_baro_data()
    final_df['BomBaro'] = np.nan
    final_df['Rainfall'] = np.nan
    
    if bom_data is not None:
        logger.info("Merging BoM data...")
        final_df['merge_date'] = final_df['Date Time (dd/mm/yyyy hh24:mi:ss)'].dt.date
        bom_data['merge_date'] = bom_data['Date Time (dd/mm/yyyy hh24:mi:ss)'].dt.date
        
        final_df = pd.merge(final_df, bom_data[['merge_date', 'BomBaro', 'Rainfall']], 
                           on='merge_date', how='left', suffixes=('', '_bom'))
        
        if 'BomBaro_bom' in final_df.columns:
            final_df['BomBaro'] = final_df['BomBaro'].fillna(final_df['BomBaro_bom'])
            final_df = final_df.drop(columns=['BomBaro_bom'])
        
        if 'Rainfall_bom' in final_df.columns:
            final_df['Rainfall'] = final_df['Rainfall'].fillna(final_df['Rainfall_bom'])
            final_df = final_df.drop(columns=['Rainfall_bom'])
        
        final_df = final_df.drop(columns=['merge_date'])
        logger.info(f"Merge complete. {final_df['BomBaro'].notna().sum()} rows with BoM data")
    else:
        logger.warning("Weather station data not available. Skipping merge.")
    
    # Apply comments
    logger.info("Applying comments...")
    if 'OTHER - Comments - Text' not in final_df.columns:
        final_df['OTHER - Comments - Text'] = ''
    final_df['OTHER - Comments - Text'] = final_df['OTHER - Comments - Text'].fillna('')
    
    # Stale data comments
    two_months_ago = script_run_time - relativedelta(months=2)
    current_month_year = script_run_time.strftime("%B %Y")
    
    latest_dates = final_df.groupby('Sample Point')['Date Time (dd/mm/yyyy hh24:mi:ss)'].transform('max')
    stale_mask = (latest_dates < two_months_ago) & (final_df['OTHER - Comments - Text'] == '')
    final_df.loc[stale_mask, 'OTHER - Comments - Text'] = f"No telemetry data received for {current_month_year}"
    
    # Zero/negative depth comments
    zero_mask = (final_df['Depth(m)raw'] <= 0) & final_df['Depth(m)raw'].notna() & (final_df['OTHER - Comments - Text'] == '')
    final_df.loc[zero_mask, 'OTHER - Comments - Text'] = "There is an equipment issue or the pool is dry"
    
    # Calculate adjusted depth
    logger.info("Calculating adjusted depth...")
    final_df = calculate_adjusted_depth(final_df, water_density, reference_density)
    
    # Filter to current month
    current_month = script_run_time.month
    current_year = script_run_time.year
    final_df = final_df[
        (final_df['Date Time (dd/mm/yyyy hh24:mi:ss)'].dt.month == current_month) &
        (final_df['Date Time (dd/mm/yyyy hh24:mi:ss)'].dt.year == current_year)
    ]
    
    # Add placeholders for sites missing in current month
    present_sites = set(final_df['Sample Point'].unique())
    missing_current = EXPECTED_SITES - present_sites
    if missing_current:
        logger.info(f"Adding placeholders for {len(missing_current)} sites missing in current month")
        for site in missing_current:
            placeholder = create_placeholder_data(
                site, 
                f"No telemetry data received for {current_month_year}", 
                script_run_time
            )
            final_df = pd.concat([final_df, placeholder], ignore_index=True)
    
    # Sort and consolidate comments
    final_df = final_df.sort_values(['Sample Point', 'Date Time (dd/mm/yyyy hh24:mi:ss)'])
    final_df = consolidate_comments(final_df)
    
    # Save output files
    logger.info("Saving output files...")
    
    try:
        save_output_file(final_df, sql_output_file, for_sql=True)
    except Exception as e:
        logger.error(f"Error creating SQL file: {e}", exc_info=True)
    
    try:
        save_output_file(final_df, output_file, for_sql=False)
    except Exception as e:
        logger.error(f"Error creating verification file: {e}", exc_info=True)
    
    logger.info("Consolidation complete.")
    
    if missing_files:
        logger.warning(f"Summary - Missing files: {', '.join(sorted(missing_files))}")


if __name__ == "__main__":
    # Example usage
    consolidate_csv_files(
        input_folder='./data/raw',
        output_file='./data/output/verification.csv',
        log_file='./logs/validation.log',
        water_density=1000.0
    )