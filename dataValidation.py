import os
import pandas as pd
import numpy as np
import datetime
from dateutil.relativedelta import relativedelta
import logging
from logging import FileHandler
from typing import Optional, Tuple, Set, Dict, List
import weatherStation 
import json

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
        site_config_json = os.getenv('SITE_CONFIG_JSON', '{}') # Default is an empty JSON string
        SITE_CONFIG = json.loads(site_config_json)
        if not isinstance(SITE_CONFIG, dict):
            logger_instance.warning("SITE_CONFIG_JSON env variable not a valid JSON dictionary. Using empty config.")
            SITE_CONFIG = {}
        EXPECTED_SITES = set(SITE_CONFIG.keys())
        logger_instance.info(f"Loaded configuration for {len(EXPECTED_SITES)} expected sites.")
    except json.JSONDecodeError:
        logger_instance.error("Failed to parse SITE_CONFIG_JSON env variable. Using empty config.", exc_info=True)
        SITE_CONFIG = {}
        EXPECTED_SITES = set()
    except Exception as e:
        logger_instance.error(f"Unexpected error loading site configuration: {e}", exc_info=True)
        SITE_CONFIG = {}
        EXPECTED_SITES = set()

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
                      reason embedded in the 'Other - Comments - Text' column.
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
        'Other - Comments - Text': [f"{reason} on {placeholder_datetime_comment}"]
    })

    return placeholder_df

def process_site_file(file_path: str, logger: logging.Logger) -> Tuple[Optional[pd.DataFrame], str, bool]:
    """
    Reads, validates, and processes a single site's merged CSV file.
    Returns (DataFrame | None, site_name, has_valid_data_flag).
    """
    site_name = os.path.splitext(os.path.basename(file_path))[0]
    logger.info(f"Processing site file: {site_name}")

    try:
        df = pd.read_csv(file_path)

        if df.empty:
            logger.warning(f"Site {site_name}: File is empty.")
            return create_placeholder_data(site_name, "CSV file was empty"), site_name, False

        expected_columns = ['Date', 'Time', 'Level(RAW)[Main Buffer] (ft)', 'Barometric Pressure(RAW)[Main Buffer] (hPa)']
        missing_columns = [col for col in expected_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Site {site_name}: Missing required columns: {missing_columns}. Cannot process.")
            return None, site_name, False

        df['Sample Point'] = site_name

        initial_row_count = len(df)
        try:
            df['Date Time (dd/mm/yyyy hh24:mi:ss)'] = pd.to_datetime(
                df['Date'] + ' ' + df['Time'],
                format='%d/%m/%Y %H:%M:%S',
                errors='coerce'
            )
            df.dropna(subset=['Date Time (dd/mm/yyyy hh24:mi:ss)'], inplace=True)
            dropped_datetime = initial_row_count - len(df)
            if dropped_datetime > 0:
                logger.warning(f"Site {site_name}: Removed {dropped_datetime} rows due to invalid source Date/Time format.")

            if df.empty:
                 logger.warning(f"Site {site_name}: No valid Date/Time rows remaining. Creating placeholder.")
                 return create_placeholder_data(site_name, "No valid Date/Time entries found"), site_name, False

        except Exception as e:
            logger.error(f"Site {site_name}: Error processing Date/Time columns: {e}.", exc_info=True)
            return None, site_name, False

        numeric_cols = ['Level(RAW)[Main Buffer] (ft)', 'Barometric Pressure(RAW)[Main Buffer] (hPa)']
        df = convert_to_numeric(df, numeric_cols)
        rows_before_nan_drop = len(df)
        df.dropna(subset=numeric_cols, inplace=True)
        dropped_numeric = rows_before_nan_drop - len(df)
        if dropped_numeric > 0:
             logger.warning(f"Site {site_name}: Removed {dropped_numeric} rows due to non-numeric values in required columns.")

        if df.empty:
            logger.warning(f"Site {site_name}: No rows remaining after numeric conversion. Creating placeholder.")
            return create_placeholder_data(site_name, "No valid numeric data found"), site_name, False

        level_col = 'Level(RAW)[Main Buffer] (ft)'
        rows_before_filter = len(df)
        df = df[df[level_col] >= 0].copy()
        dropped_filter = rows_before_filter - len(df)
        if dropped_filter > 0:
             logger.info(f"Site {site_name}: Removed {dropped_filter} rows where '{level_col}' was less than 0.")

        if df.empty:
            logger.warning(f"Site {site_name}: No valid data remaining after filtering negative levels. Creating placeholder.")
            return create_placeholder_data(site_name, f"No data found with {level_col} >= 0"), site_name, False

        config = SITE_CONFIG.get(site_name, {'depth_conversion_type': 'default'})
        conversion_type = config.get('depth_conversion_type', 'default')

        if conversion_type == 'divide_by_100':
            df['Depth(m)raw'] = (df[level_col] / 100).round(2)
        else: # Default is feet to meters conversion
            FEET_TO_METERS = 0.3048
            df['Depth(m)raw'] = (df[level_col] * FEET_TO_METERS).round(2)
            if conversion_type != 'default':
                 logger.warning(f"Site {site_name}: Unknown depth_conversion_type '{conversion_type}'. Using default (ft to m).")

        df = df.sort_values('Date Time (dd/mm/yyyy hh24:mi:ss)')
        logger.info(f"Site {site_name}: Successfully processed {len(df)} valid data rows.") 
        return df, site_name, True
    
    # Error handling blocks
    # Goal here is prevent failure due to single file. Manages a problematic site (logging, placeholders, or signaling failure with None).
    except pd.errors.EmptyDataError:
        logger.warning(f"Skipping {site_name}: File is empty (caught by pd.errors.EmptyDataError).")
        return create_placeholder_data(site_name, "Empty file"), site_name, False
    except pd.errors.ParserError:
        logger.error(f"Site {site_name}: Failed to parse CSV file.", exc_info=True)
        return None, site_name, False
    except KeyError as e:
         logger.error(f"Site {site_name}: Missing expected column during processing: {e}.", exc_info=True)
         return None, site_name, False
    except Exception as e:
        logger.error(f"Site {site_name}: Unexpected error during processing: {e}", exc_info=True)
        return None, site_name, False

def calculate_adjusted_depth(df: pd.DataFrame, water_density: float, logger: logging.Logger, reference_density: float = 1000.0) -> pd.DataFrame:
    """
    Calculates adjusted depth based on logger and external barometric pressure.
    Adds 'Depth(m)adjusted' column to the DataFrame.
    """
    logger.info("Calculating adjusted depth based on barometric differences...")
    df_copy = df.copy()

    HPA_TO_PSI = 0.0145038
    CONVERSION_FACTOR = 0.70307 # AquaTroll m/(PSI/SG)

    if reference_density <= 0:
        logger.error("Reference density must be positive. Cannot calculate adjusted depth.")
        df_copy['Depth(m)adjusted'] = np.nan
        return df_copy
    SG = water_density / reference_density
    if SG <= 0:
        logger.error(f"Calculated Specific Gravity (SG={SG:.4f}) is non-positive. Cannot calculate adjusted depth.")
        df_copy['Depth(m)adjusted'] = np.nan
        return df_copy

    required_cols = ['Depth(m)raw', 'Barometric Pressure(RAW)[Main Buffer] (hPa)', 'BomBaro']
    missing_cols = [col for col in required_cols if col not in df_copy.columns]
    if missing_cols:
        logger.error(f"Missing required columns for depth adjustment: {missing_cols}.")
        df_copy['Depth(m)adjusted'] = np.nan
        return df_copy

    df_copy = convert_to_numeric(df_copy, required_cols)
    df_copy['Depth(m)adjusted'] = np.nan

    mask = (
        df_copy['Depth(m)raw'].notna() &
        df_copy['Barometric Pressure(RAW)[Main Buffer] (hPa)'].notna() &
        df_copy['BomBaro'].notna()
    )

    eligible_rows_count = mask.sum()
    logger.info(f"Found {eligible_rows_count} rows eligible for depth adjustment calculation.")

    if eligible_rows_count > 0:
        delta_p_hpa = df_copy.loc[mask, 'Barometric Pressure(RAW)[Main Buffer] (hPa)'] - df_copy.loc[mask, 'BomBaro']
        delta_p_psi = delta_p_hpa * HPA_TO_PSI
        depth_adjustment_m = CONVERSION_FACTOR * delta_p_psi / SG
        df_copy.loc[mask, 'Depth(m)adjusted'] = df_copy.loc[mask, 'Depth(m)raw'] + depth_adjustment_m
        df_copy['Depth(m)adjusted'] = df_copy['Depth(m)adjusted'].round(2)
        logger.info(f"Applied depth adjustment to {eligible_rows_count} rows.")
    else:
        logger.warning("No rows found with complete data for depth adjustment calculation.")

    return df_copy

# --- Primary Function ---

def consolidate_csv_files(
    input_folder: str,
    output_file: str, # Path for validatedDepthData.csv
    log_file: Optional[str] = None,
    enable_file_logging: bool = False,
    water_density: float = 1000.0, # Default to freshwater density kg/m³
    reference_density: float = 1000.0 # Default reference density for SG
) -> None:
    """
    Main function to consolidate site data, merge weather data, apply specific comments,
    calculate adjusted depth, and save output files.
    """
    logger = setup_logging(log_file, enable_file_logging)
    script_run_time = datetime.datetime.now()
    logger.info(f"Starting data validation and consolidation process at {script_run_time.strftime('%Y-%m-%d %H:%M:%S')}")

    load_site_config() # Load SITE_CONFIG and EXPECTED_SITES

    if not os.path.isdir(input_folder):
        logger.error(f"Input folder not found: {input_folder}")
        return
    logger.info(f"Input data folder: {input_folder}")

    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            logger.info(f"Created output directory: {output_dir}")
        except Exception as e:
            logger.error(f"Failed to create output directory '{output_dir}': {e}", exc_info=True)
            return
    logger.info(f"Main output file target: {output_file}")
    greater_pbo_output_file = os.path.join(output_dir, 'SWLVLGenericTemplate_greaterPBOPools.csv')
    logger.info(f"Secondary output file target: {greater_pbo_output_file}")

    try:
        all_files = os.listdir(input_folder)
        csv_files = [
            f for f in all_files
            if f.lower().endswith('.csv') and not f.lower().startswith('baro')
        ]
        logger.info(f"Found {len(csv_files)} potential site data CSV files to process.")
    except Exception as e:
        logger.error(f"Error listing files in input folder '{input_folder}': {e}", exc_info=True)
        return

    processed_sites = set()
    sites_with_data = set()
    sites_with_placeholder = set()
    failed_processing_sites = set()
    consolidated_data = []

    # Process individual site files first
    for filename in csv_files:
        file_path = os.path.join(input_folder, filename)
        # Assuming process_site_file returns (DataFrame | None, site_name, has_valid_data_flag)
        # And that if has_valid_data is False, the DataFrame is a placeholder created by create_placeholder_data
        site_df, site_name, has_valid_data = process_site_file(file_path, logger)
        processed_sites.add(site_name)
        if site_df is not None:
            consolidated_data.append(site_df)
            if has_valid_data:
                sites_with_data.add(site_name)
            else:
                # Call from create_placeholder_data
                sites_with_placeholder.add(site_name)
        else:
            failed_processing_sites.add(site_name)

    # Handle expected sites that were completely missing
    missing_site_files = EXPECTED_SITES - processed_sites
    if missing_site_files:
        logger.warning(f"Expected site files not found: {', '.join(sorted(missing_site_files))}")
        for site_name in missing_site_files:
            # --- Pass script_run_time to create_placeholder_data ---
            placeholder_df = create_placeholder_data(
                site_name,
                "Site CSV file not found",
                timestamp_for_entry=script_run_time # Pass the consistent time
            )
            consolidated_data.append(placeholder_df)
            sites_with_placeholder.add(site_name)

    if not consolidated_data:
        logger.error("No data consolidated (including placeholders). No output files created.")
        return

    try:
        logger.info("Concatenating all site data (including initial placeholders)...")
        final_df = pd.concat(consolidated_data, ignore_index=True)

        # Convert Date/Time (handles placeholders potentially stored as string)
        # Do this early to allow filtering/grouping
        final_df['Date Time (dd/mm/yyyy hh24:mi:ss)'] = pd.to_datetime(
             final_df['Date Time (dd/mm/yyyy hh24:mi:ss)'], errors='coerce'
        )
        invalid_site_dates = final_df['Date Time (dd/mm/yyyy hh24:mi:ss)'].isna().sum()
        if invalid_site_dates > 0:
            logger.warning(f"Found {invalid_site_dates} rows with invalid or unparseable Date/Time in combined data. Dropping.")
            final_df.dropna(subset=['Date Time (dd/mm/yyyy hh24:mi:ss)'], inplace=True)

        if final_df.empty:
            logger.error("No valid data remaining after Date/Time conversion/validation. Cannot proceed.")
            return

        # Sort and Deduplicate (Per Site, Per Day) - Keep only the first record per site per day
        logger.info("Sorting and deduplicating data (keeping first record per site per day)...")
        final_df = final_df.sort_values(['Sample Point', 'Date Time (dd/mm/yyyy hh24:mi:ss)'])
        # Ensure datetime is available for key creation
        if pd.api.types.is_datetime64_any_dtype(final_df['Date Time (dd/mm/yyyy hh24:mi:ss)']):
            site_date_key = final_df['Sample Point'].astype(str) + final_df['Date Time (dd/mm/yyyy hh24:mi:ss)'].dt.date.astype(str)
            final_df = final_df.loc[~site_date_key.duplicated(keep='first')].copy()
            logger.info(f"Data ready for BoM merge after deduplication: {final_df.shape[0]} rows.")
        else:
            logger.error("Cannot deduplicate daily data as DateTime column is not valid.")
            return

        # --- Retrieve and Merge Weather Data ---
        bom_baro_data = get_bom_baro_data(logger)
        final_df['BomBaro'] = np.nan # Ensure column exists even if merge fails

        if bom_baro_data is not None and not bom_baro_data.empty:
            logger.info("Attempting to merge BoM data...")
            try:
                # Ensure merge keys are compatible types (date objects)
                final_df['merge_date'] = final_df['Date Time (dd/mm/yyyy hh24:mi:ss)'].dt.date
                bom_baro_data['merge_date'] = bom_baro_data['Date Time (dd/mm/yyyy hh24:mi:ss)'].dt.date

                final_df = pd.merge(
                    final_df,
                    bom_baro_data[['merge_date', 'BomBaro']],
                    on='merge_date',
                    how='left',
                    suffixes=('', '_bom') # Suffix avoids auto-renaming if 'BomBaro' already exists
                )

                # Use the merged data, preferring original if merge failed for a row
                if 'BomBaro_bom' in final_df.columns:
                    # Use fillna to keep original BomBaro if _bom version is NaN
                    final_df['BomBaro'] = final_df['BomBaro'].fillna(final_df['BomBaro_bom'])
                    final_df = final_df.drop(columns=['merge_date', 'BomBaro_bom'], errors='ignore')
                elif 'merge_date' in final_df.columns:
                     final_df = final_df.drop(columns=['merge_date'], errors='ignore') # Clean up merge key if _bom wasn't created

                bom_data_count = final_df['BomBaro'].notna().sum()
                logger.info(f"Merge complete. {bom_data_count} rows have BoM data after merge attempt.")

            except Exception as e:
                 logger.error(f"Error during the BoM data merge: {e}", exc_info=True)
                 # Clean up merge key regardless of error
                 if 'merge_date' in final_df.columns: final_df = final_df.drop(columns=['merge_date'], errors='ignore')
        else:
            logger.warning("BoM data not available or empty. Merge skipped.")
            # Clean up merge key if it exists from failed attempt before merge check
            if 'merge_date' in final_df.columns: final_df = final_df.drop(columns=['merge_date'], errors='ignore')
        
        # Begin dealing with any missing/bad data
        logger.info("Applying specific comments based on data staleness and values...")
        # Separate missing data observations (created by process_site_file or for missing files)
        # Ensure the 'Other - Comments - Text' column exists. If not, create it empty.
        if 'Other - Comments - Text' not in final_df.columns:
            final_df['Other - Comments - Text'] = '' # Initialise with empty strings
        # Fill NaN values with empty strings to allow safe use of .str accessor
        final_df['Other - Comments - Text'] = final_df['Other - Comments - Text'].fillna('')
        # Apply the .str method
        original_comment_mask = final_df['Other - Comments - Text'].str.startswith("Placeholder:")
        
        final_df_original_placeholders = final_df[original_comment_mask].copy()
        final_df_data_to_process = final_df[~original_comment_mask].copy()

        if final_df_data_to_process.empty and not final_df_original_placeholders.empty:
            logger.info("Only original placeholder data found. Skipping staleness/zero checks.")
            final_df_commented = final_df_original_placeholders # Use only original placeholders
        elif final_df_data_to_process.empty and final_df_original_placeholders.empty:
             logger.error("No data (including placeholders) remaining before comment logic. Cannot proceed.")
             return
        else:
            # --- Condition 1: Expired Data Check (>2months old) ---
            two_months_ago = script_run_time - relativedelta(months=2)
            current_month_year = script_run_time.strftime("%B %Y")

            latest_dates = final_df_data_to_process.groupby('Sample Point')['Date Time (dd/mm/yyyy hh24:mi:ss)'].max()
            stale_sites = latest_dates[latest_dates < two_months_ago].index.tolist()

            stale_placeholders = []
            if stale_sites:
                logger.info(f"Found stale sites (last data before {two_months_ago.date()}): {', '.join(stale_sites)}")
                stale_reason = f"No telemetry data received for {current_month_year}" # Use formatted month/year
                for site in stale_sites:
                    # Create a new placeholder row using the existing function
                    # This uses the current time, as requested for the placeholder row
                    placeholder_df = create_placeholder_data(site, stale_reason)
                    stale_placeholders.append(placeholder_df)

                # Remove the original (now identified as stale) data rows for these sites
                final_df_data_to_process = final_df_data_to_process[~final_df_data_to_process['Sample Point'].isin(stale_sites)]
                logger.info(f"Removed old data for stale sites. Processing remaining {len(final_df_data_to_process)} rows for zero check.")

            # --- Condition 2: Zero/Negative Depth Check (on remaining non-stale, non-placeholder data) ---
            if not final_df_data_to_process.empty:
                 # Ensure Depth(m)raw is numeric before comparison
                 final_df_data_to_process['Depth(m)raw'] = pd.to_numeric(final_df_data_to_process['Depth(m)raw'], errors='coerce')

                 # Mask for rows where Depth(m)raw is <= 0 AND is not NaN
                 zero_neg_mask = (final_df_data_to_process['Depth(m)raw'] <= 0) & final_df_data_to_process['Depth(m)raw'].notna()
                 zero_neg_comment = "There is an equipment issue or the pool is dry"

                 # Apply comment only where the condition is met and comment is currently empty/NaN
                 # This prevents overwriting original placeholders if logic changes later
                 current_comment_empty = final_df_data_to_process['Other - Comments - Text'].isna() | \
                                         (final_df_data_to_process['Other - Comments - Text'] == '')
                 apply_comment_mask = zero_neg_mask & current_comment_empty

                 final_df_data_to_process.loc[apply_comment_mask, 'Other - Comments - Text'] = zero_neg_comment
                 num_zero_comments_applied = apply_comment_mask.sum()
                 if num_zero_comments_applied > 0:
                    logger.info(f"Applied zero/negative depth comment to {num_zero_comments_applied} rows.")

            # --- Combine Processed Data ---
            all_processed_frames = []
            if not final_df_original_placeholders.empty:
                 all_processed_frames.append(final_df_original_placeholders)
            if not final_df_data_to_process.empty:
                 all_processed_frames.append(final_df_data_to_process)
            if stale_placeholders: # Check if list is not empty
                # Need to concat these new DFs which might have slightly different dtypes initially
                stale_df_combined = pd.concat(stale_placeholders, ignore_index=True)
                all_processed_frames.append(stale_df_combined)

            if not all_processed_frames:
                 logger.error("No data frames to combine after comment logic. Cannot proceed.")
                 return

            final_df_commented = pd.concat(all_processed_frames, ignore_index=True)
            # Ensure DateTime is still valid after concat, necessary for sorting
            final_df_commented['Date Time (dd/mm/yyyy hh24:mi:ss)'] = pd.to_datetime(
                 final_df_commented['Date Time (dd/mm/yyyy hh24:mi:ss)'], errors='coerce'
            )
            final_df_commented.dropna(subset=['Date Time (dd/mm/yyyy hh24:mi:ss)'], inplace=True) # Drop if conversion failed

            # Sort again for final output consistency
            if not final_df_commented.empty:
                 final_df_commented = final_df_commented.sort_values(['Sample Point', 'Date Time (dd/mm/yyyy hh24:mi:ss)'])
            else:
                 logger.error("DataFrame is empty after final comment processing and DateTime validation.")
                 return

        # --- Calculate Adjusted Depth ---
        # Use final_df_commented (the result from the comment logic block)
        if final_df_commented.empty:
            logger.error("final_df_commented is empty before calculating adjusted depth.")
            # Decide how to proceed - maybe create empty output files? For now, return.
            return
        else:
            # Ensure BomBaro is numeric before calculation
            if 'BomBaro' in final_df_commented.columns:
                 final_df_commented['BomBaro'] = pd.to_numeric(final_df_commented['BomBaro'], errors='coerce')
            # Pass the result of the comment logic to the calculation
            final_df_processed = calculate_adjusted_depth(final_df_commented, water_density, logger, reference_density)


        # --- Prepare Final Output Columns Order ---
        # Use final_df_processed (the result from calculate_adjusted_depth)
        final_column_order = [
            'Sample Point', 'Date Time (dd/mm/yyyy hh24:mi:ss)', 'BomBaro',
            'Barometric Pressure(RAW)[Main Buffer] (hPa)', 'Depth(m)raw',
            'Depth(m)adjusted', 'Other - Comments - Text'
        ]
        # Ensure all required columns exist in the final processed DataFrame
        for col in final_column_order:
             if col not in final_df_processed.columns:
                 final_df_processed[col] = np.nan # Add missing columns as NaN

        # Select and order columns for the final output DataFrame
        final_df_output = final_df_processed[final_column_order].copy() # Use copy to avoid SettingWithCopyWarning

        logger.info("Summary of rows per site in final dataset before saving:")
        # Use final_df_output for the summary
        site_counts = final_df_output['Sample Point'].value_counts().sort_index().to_dict()
        for site, count in site_counts.items(): logger.info(f"  - {site}: {count} row(s)")

        # --- Create and Save greaterPBOPools.csv ---
        # Base this on final_df_output (which has comments applied and adjusted depth calculated)
        logger.info(f"Preparing secondary output file: {os.path.basename(greater_pbo_output_file)}")
        pbo_df = pd.DataFrame() # Initialise empty DataFrame

        try:
            # Filter final_df_output for rows with valid (non-NaN) adjusted depth
            adjusted_depth_exists = 'Depth(m)adjusted' in final_df_output.columns
            if adjusted_depth_exists and final_df_output['Depth(m)adjusted'].notna().any():
                # Create pbo_df by selecting rows where adjusted depth is NOT NaN
                pbo_df = final_df_output.loc[
                    final_df_output['Depth(m)adjusted'].notna(), # Key filter to exclude placeholders
                    ['Sample Point', 'Date Time (dd/mm/yyyy hh24:mi:ss)', 'Depth(m)adjusted', 'Other - Comments - Text']
                ].copy() # Use .copy()

                if not pbo_df.empty:
                    logger.info(f"Created pbo_df with {len(pbo_df)} rows (excluding placeholders).")

                    # Rename the depth column specifically for this file
                    pbo_df.rename(columns={'Depth(m)adjusted': 'LEVEL - DEPTH TO WATER - m (INPUT)'}, inplace=True)

                    # Define the specific column order for this file
                    pbo_column_order = [
                        'Sample Point', 'Date Time (dd/mm/yyyy hh24:mi:ss)',
                        'LEVEL - DEPTH TO WATER - m (INPUT)', 'Other - Comments - Text'
                    ]
                    # Ensure all columns exist before reordering
                    for col in pbo_column_order:
                        if col not in pbo_df.columns:
                             pbo_df[col] = np.nan # Add if missing, although unlikely here
                    pbo_df = pbo_df[pbo_column_order] # Reorder/select columns

                    # Format DateTime specifically for pbo_df output
                    # Check column exists and is datetime type
                    if 'Date Time (dd/mm/yyyy hh24:mi:ss)' in pbo_df.columns and pd.api.types.is_datetime64_any_dtype(pbo_df['Date Time (dd/mm/yyyy hh24:mi:ss)']):
                         pbo_df['Date Time (dd/mm/yyyy hh24:mi:ss)'] = pbo_df['Date Time (dd/mm/yyyy hh24:mi:ss)'].dt.strftime('%d/%m/%Y %I:%M:%S %p')
                    else:
                        logger.warning("Could not format DateTime for greaterPBOPools.csv as it's missing or not datetime type in pbo_df.")

                    # Save the filtered and formatted DataFrame
                    pbo_df.to_csv(greater_pbo_output_file, index=False, na_rep='')
                    logger.info(f"Secondary output file saved: {greater_pbo_output_file}")
                else:
                    # This case occurs if adjusted depth existed but was NaN for all rows
                     logger.warning(f"No rows with valid adjusted depth data found after filtering. File '{os.path.basename(greater_pbo_output_file)}' not created.")

            else:
                 # This case occurs if 'Depth(m)adjusted' column didn't exist or was all NaN initially
                logger.warning(f"No data with valid adjusted depth found (column missing or all NaN). File '{os.path.basename(greater_pbo_output_file)}' not created.")

        except KeyError as e:
            logger.error(f"Error preparing '{os.path.basename(greater_pbo_output_file)}': Missing column {e}.", exc_info=True)
        except Exception as e:
            logger.error(f"Error creating or saving '{os.path.basename(greater_pbo_output_file)}': {e}.", exc_info=True)


        # --- Create and Save Environmental Database File (SWLVLGenericTemplate_validatedDepthData.csv) ---
        # Use final_df_output (which contains data rows AND placeholder rows with comments)
        logger.info(f"Preparing main output file: {os.path.basename(output_file)}")
        if 'Date Time (dd/mm/yyyy hh24:mi:ss)' in final_df_output.columns and pd.api.types.is_datetime64_any_dtype(final_df_output['Date Time (dd/mm/yyyy hh24:mi:ss)']):
            # Apply final string formatting to the DateTime column for the main output
            final_df_output['Date Time (dd/mm/yyyy hh24:mi:ss)'] = final_df_output['Date Time (dd/mm/yyyy hh24:mi:ss)'].dt.strftime('%d/%m/%Y %I:%M:%S %p')
        else:
            logger.warning("DateTime column not suitable for final string formatting in main output.")

        # Save the main output file, including placeholders and commented rows
        final_df_output.to_csv(output_file, index=False, na_rep='')
        logger.info(f"Main output file saved: {output_file}")


        # --- Final Summary Logging ---
        logger.info("Data validation and consolidation summary:")
        total_processed = len(processed_sites | missing_site_files) # Union of found and missing expected sites
        logger.info(f" - Considered {total_processed} expected or found sites.")
        # Recalculate placeholder sites based on final output
        final_placeholder_sites = set(final_df_output.loc[final_df_output['Other - Comments - Text'].notna() &
                                                        final_df_output['Other - Comments - Text'].str.contains("Placeholder:|No telemetry data|equipment issue", na=False), 'Sample Point'])
        final_data_sites = set(final_df_output['Sample Point']) - final_placeholder_sites

        logger.info(f" - Sites with data rows in final output: {len(final_data_sites)} ({', '.join(sorted(final_data_sites))})")
        logger.info(f" - Sites with placeholder/comment-only rows in final output: {len(final_placeholder_sites)} ({', '.join(sorted(final_placeholder_sites))})")
        if failed_processing_sites:
             # Note: Failed sites might end up as placeholders if caught correctly
             logger.error(f" - Sites with critical processing errors reported earlier: {len(failed_processing_sites)} ({', '.join(sorted(failed_processing_sites))})")

        logger.info("Processing Complete.")

    except Exception as e:
        logger.error(f"Critical error during consolidation, comment application, or saving: {e}", exc_info=True)
        logger.info("Processing Failed Critically.")