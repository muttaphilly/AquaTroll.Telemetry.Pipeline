import os
import pandas as pd
import numpy as np
import datetime
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
    Retrieves and processes barometric data from the weatherStation .
    Returns a DataFrame ready for merging or None on failure.
    """
    logger.info("Retrieving weather station barometric data")

    try:
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

            # Just keep first entry for each date
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

def create_placeholder_data(site_name: str, reason: str) -> pd.DataFrame:
    """Creates a DataFrame row for sites with missing data."""
    logger_instance.warning(f"Creating placeholder data for site '{site_name}': {reason}")
    placeholder_datetime_str = datetime.datetime.now().strftime('%d/%m/%Y 00:00:00')
    placeholder_df = pd.DataFrame({
        'Sample Point': [site_name],
        'Date Time (dd/mm/yyyy hh24:mi:ss)': [placeholder_datetime_str],
        'Depth(m)raw': [np.nan],
        'Barometric Pressure(RAW)[Main Buffer] (hPa)': [np.nan],
        'Other - Comments - Text': [f"Placeholder: {reason} on {datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')}"]
    })
    expected_cols = [
        'Sample Point', 'Date Time (dd/mm/yyyy hh24:mi:ss)', 'Depth(m)raw',
        'Barometric Pressure(RAW)[Main Buffer] (hPa)', 'Other - Comments - Text'
    ]
    for col in expected_cols:
        if col not in placeholder_df.columns:
            placeholder_df[col] = np.nan
    return placeholder_df[expected_cols]

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
            df['Depth(m)raw'] = (df[level_col] * FEET_TO_METERS).round(3)
            if conversion_type != 'default':
                 logger.warning(f"Site {site_name}: Unknown depth_conversion_type '{conversion_type}'. Using default (ft to m).")

        df['Other - Comments - Text'] = ""
        df = df.sort_values('Date Time (dd/mm/yyyy hh24:mi:ss)')

        output_cols = [
            'Sample Point', 'Date Time (dd/mm/yyyy hh24:mi:ss)', 'Depth(m)raw',
            'Barometric Pressure(RAW)[Main Buffer] (hPa)', 'Other - Comments - Text'
        ]
        for col in output_cols:
            if col not in df.columns:
                 df[col] = np.nan

        logger.info(f"Site {site_name}: Successfully processed {len(df)} valid data rows.")
        return df[output_cols], site_name, True

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
        df_copy['Depth(m)adjusted'] = df_copy['Depth(m)adjusted'].round(3)
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
    Main function to consolidate site data, merge weather data, calculate adjusted depth,
    and save output files.
    """
    logger = setup_logging(log_file, enable_file_logging)
    logger.info("Starting data validation and consolidation process")

    load_site_config()

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
    greater_pbo_output_file = os.path.join(output_dir, 'greaterPBOPools.csv')
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
            failed_processing_sites.add(site_name)

    missing_site_files = EXPECTED_SITES - processed_sites
    if missing_site_files:
        logger.warning(f"Expected site files not found: {', '.join(sorted(missing_site_files))}")
        for site_name in missing_site_files:
            placeholder_df = create_placeholder_data(site_name, "Site CSV file not found")
            consolidated_data.append(placeholder_df)
            sites_with_placeholder.add(site_name)

    if not consolidated_data:
        logger.error("No data consolidated. No output files created.")
        return

    try:
        logger.info("Concatenating data...")
        final_df = pd.concat(consolidated_data, ignore_index=True)

        # Convert Date/Time (handles placeholders potentially stored as string)
        final_df['Date Time (dd/mm/yyyy hh24:mi:ss)'] = pd.to_datetime(
             final_df['Date Time (dd/mm/yyyy hh24:mi:ss)'], errors='coerce'
        )
        invalid_site_dates = final_df['Date Time (dd/mm/yyyy hh24:mi:ss)'].isna().sum()
        if invalid_site_dates > 0:
            logger.warning(f"Found {invalid_site_dates} rows with invalid Date/Time in combined data. Dropping.")
            final_df.dropna(subset=['Date Time (dd/mm/yyyy hh24:mi:ss)'], inplace=True)

        if final_df.empty:
            logger.error("No valid data remaining after Date/Time conversion. Cannot proceed.")
            return

        # Sort and Deduplicate (Per Site, Per Day)
        logger.info("Sorting and deduplicating data...")
        final_df = final_df.sort_values(['Sample Point', 'Date Time (dd/mm/yyyy hh24:mi:ss)'])
        site_date_key = final_df['Sample Point'].astype(str) + final_df['Date Time (dd/mm/yyyy hh24:mi:ss)'].dt.date.astype(str)
        final_df = final_df.loc[~site_date_key.duplicated(keep='first')].copy()
        logger.info(f"Data ready for BoM merge: {final_df.shape[0]} rows.")

        # --- Retrieve and Merge Weather Data ---
        bom_baro_data = get_bom_baro_data(logger)
        final_df['BomBaro'] = np.nan

        if bom_baro_data is not None and not bom_baro_data.empty:
            logger.info("Attempting to merge BoM data...")
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
                    final_df = final_df.drop(columns=['merge_date', 'BomBaro_bom'])
                elif 'merge_date' in final_df.columns:
                     final_df = final_df.drop(columns=['merge_date'])

                bom_data_count = final_df['BomBaro'].notna().sum()
                logger.info(f"Merge complete. {bom_data_count} rows have BoM data.")

            except Exception as e:
                 logger.error(f"Error during the BoM data merge: {e}", exc_info=True)
                 if 'merge_date' in final_df.columns: final_df = final_df.drop(columns=['merge_date'], errors='ignore')
        else:
            logger.warning("BoM data not available or empty. Merge skipped.")
            if 'merge_date' in final_df.columns: final_df = final_df.drop(columns=['merge_date'], errors='ignore')

        # --- Calculate Adjusted Depth ---
        if final_df.empty:
            logger.error("final_df is empty before calculating adjusted depth.")
        else:
            if 'BomBaro' in final_df.columns: final_df['BomBaro'] = pd.to_numeric(final_df['BomBaro'], errors='coerce')
            final_df = calculate_adjusted_depth(final_df, water_density, logger, reference_density)

        # --- Prepare Final Output Columns Order ---
        final_column_order = [
            'Sample Point', 'Date Time (dd/mm/yyyy hh24:mi:ss)', 'BomBaro',
            'Barometric Pressure(RAW)[Main Buffer] (hPa)', 'Depth(m)raw',
            'Depth(m)adjusted', 'Other - Comments - Text'
        ]
        for col in final_column_order:
             if col not in final_df.columns: final_df[col] = np.nan
        final_df = final_df[final_column_order]

        logger.info("Summary of rows per site in final dataset:")
        site_counts = final_df['Sample Point'].value_counts().sort_index().to_dict()
        for site, count in site_counts.items(): logger.info(f"  - {site}: {count} row(s)")

        # --- Create and Save greaterPBOPools.csv ---
        logger.info(f"Preparing secondary output file: {os.path.basename(greater_pbo_output_file)}")
        pbo_df = pd.DataFrame()

        try:
            if 'Depth(m)adjusted' in final_df.columns and final_df['Depth(m)adjusted'].notna().any():
                pbo_df = final_df.loc[
                    final_df['Depth(m)adjusted'].notna(),
                    ['Sample Point', 'Date Time (dd/mm/yyyy hh24:mi:ss)', 'Depth(m)adjusted', 'Other - Comments - Text']
                ].copy()
                logger.info(f"Created pbo_df with {len(pbo_df)} rows.")

                pbo_df.rename(columns={'Depth(m)adjusted': 'LEVEL - DEPTH TO WATER - m (INPUT)'}, inplace=True)

                pbo_column_order = [
                    'Sample Point', 'Date Time (dd/mm/yyyy hh24:mi:ss)',
                    'LEVEL - DEPTH TO WATER - m (INPUT)', 'Other - Comments - Text'
                ]
                pbo_df = pbo_df[pbo_column_order]

                # Format DateTime specifically for pbo_df output
                pbo_df['Date Time (dd/mm/yyyy hh24:mi:ss)'] = pbo_df['Date Time (dd/mm/yyyy hh24:mi:ss)'].dt.strftime('%d/%m/%Y %I:%M:%S %p')

                pbo_df.to_csv(greater_pbo_output_file, index=False, na_rep='')
                logger.info(f"Secondary output file saved: {greater_pbo_output_file}")
            else:
                logger.warning(f"No data with valid adjusted depth found. File '{os.path.basename(greater_pbo_output_file)}' not created.")

        except KeyError as e:
            logger.error(f"Error preparing '{os.path.basename(greater_pbo_output_file)}': Missing column {e}.", exc_info=True)
        except Exception as e:
            logger.error(f"Error creating '{os.path.basename(greater_pbo_output_file)}': {e}.", exc_info=True)

        # --- Format DateTime and Save Main Output File (validatedDepthData.csv) ---
        logger.info(f"Preparing main output file: {os.path.basename(output_file)}")
        if 'Date Time (dd/mm/yyyy hh24:mi:ss)' in final_df.columns and pd.api.types.is_datetime64_any_dtype(final_df['Date Time (dd/mm/yyyy hh24:mi:ss)']):
            final_df['Date Time (dd/mm/yyyy hh24:mi:ss)'] = final_df['Date Time (dd/mm/yyyy hh24:mi:ss)'].dt.strftime('%d/%m/%Y %I:%M:%S %p')
        else:
            logger.warning("DateTime column not suitable for final string formatting.")

        final_df.to_csv(output_file, index=False, na_rep='')
        logger.info(f"Main output file saved: {output_file}")

        # --- Final Summary Logging ---
        logger.info("Data validation and consolidation summary:")
        total_processed = len(processed_sites)
        logger.info(f" - Processed {total_processed} sites found in files.")
        logger.info(f" - Sites with data: {len(sites_with_data)} ({', '.join(sorted(sites_with_data))})")
        placeholder_sites = sites_with_placeholder | (failed_processing_sites - sites_with_data)
        logger.info(f" - Sites as placeholders: {len(placeholder_sites)} ({', '.join(sorted(placeholder_sites))})")
        if failed_processing_sites:
             logger.error(f" - Sites with processing errors: {len(failed_processing_sites)} ({', '.join(sorted(failed_processing_sites))})")

        logger.info("Processing Complete.")

    except Exception as e:
        logger.error(f"Critical error during consolidation or saving: {e}", exc_info=True)
        logger.info("Processing Failed.")