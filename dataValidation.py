import os
import pandas as pd
import numpy as np
import datetime
import logging
from logging import FileHandler
from typing import Optional, Tuple, Set, Dict, List
import weatherStation
import json

# --- Load Sites from Environment Variable ---
SITE_CONFIG = {}
EXPECTED_SITES = set()
logger_instance = logging.getLogger('data_validation')

def load_site_config():
    """Load site configuration from environment variables."""
    global SITE_CONFIG, EXPECTED_SITES
    
    try:
        site_config_json = os.getenv('SITE_CONFIG_JSON', '')
        if site_config_json:
            SITE_CONFIG = json.loads(site_config_json)
            if not isinstance(SITE_CONFIG, dict):
                SITE_CONFIG = {}
            EXPECTED_SITES = set(SITE_CONFIG.keys())
    except Exception:
        SITE_CONFIG = {}
        EXPECTED_SITES = set()

# --- Setup logger for debugging. Uncomment to True if needed ---
def setup_logging(log_file_path: str = None, enable_file_logging: bool = False) -> logging.Logger:
    logger = logging.getLogger('data_validation')
    if logger.hasHandlers():
        logger.handlers.clear()  # Resets if re-running script in same session
    
    logger.setLevel(logging.INFO)
    
    # Create formatters and handlers
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')   
    
    # Initialise for the terminal
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)    
    
    # Only add file handler if enabled and path provided
    if enable_file_logging and log_file_path:
        file_handler = FileHandler(log_file_path, mode='w')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

# --- Utility functions ---
def parse_datetime(date_str: str, time_str: str) -> Optional[str]:
    """Parse date and time strings into a standardised datetime format."""
    try:
        dt = pd.to_datetime(f"{date_str} {time_str}", format='%d/%m/%Y %H:%M:%S', errors='coerce')
        if pd.isna(dt):
            return None
        return dt.strftime('%d/%m/%Y:00:00:00')
    except Exception:
        return None

def convert_to_numeric(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Convert specified columns to numeric values, handling errors."""
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

# --- Core data processing functions ---

def get_bom_baro_data(logger: logging.Logger) -> Optional[pd.DataFrame]:
    logger.info("Retrieving BoM barometric data")
    
    try:
        # Call weatherStation script.
        baro_data = weatherStation.scrape_weather_data()
           
        if baro_data is None or baro_data.empty:
            return None
        
        # Check required variables exist
        if not {'Date', 'Time', 'hPa'}.issubset(baro_data.columns):
            return None
        
        try:
            # Convert Date/Time to DateTime format
            baro_data['DateTime'] = pd.to_datetime(
                baro_data['Date'] + ' ' + baro_data['Time'], 
                format='%d/%m/%Y %H:%M:%S', 
                errors='coerce'
            )
            
            # Clean and prepare data
            baro_data = (baro_data
                .dropna(subset=['DateTime'])
                .assign(DateTime=lambda df: df['DateTime'].dt.strftime('%d/%m/%Y:00:00:00'))
                [['DateTime', 'hPa']]
                .rename(columns={'hPa': 'BomBaro'})
            )
            
            # Keep first entry for each date to remove duplicates
            date_only = baro_data['DateTime'].str.split(':', n=1).str[0]
            baro_data = baro_data.loc[~date_only.duplicated(keep='first')]
            
            return baro_data
            
        except Exception:
            return None
            
    except Exception:
        return None

# --- Flag site missing/invalid data with a comment ---
def create_placeholder_data(site_name: str, reason: str) -> pd.DataFrame:
    today_date = datetime.datetime.now().strftime('%d/%m/%Y')
    return pd.DataFrame({
        'Site': [site_name],
        'DateTime': [f"{today_date}:00:00:00"],
        'Depth(m)raw': [0.00],
        'Barometric Pressure(RAW)[Main Buffer] (hPa)': [np.nan],
        'Comments': [f"{reason} on {today_date}"]
    })[['Site', 'DateTime', 'Depth(m)raw', 'Barometric Pressure(RAW)[Main Buffer] (hPa)', 'Comments']]

# --- Process site with data --- 
def process_site_file(file_path: str, logger: logging.Logger) -> Tuple[Optional[pd.DataFrame], str, bool]:
    site_name = os.path.splitext(os.path.basename(file_path))[0]
    
    try:
        # Read
        df = pd.read_csv(file_path)       
        # Check if empty
        if df.empty:
            return create_placeholder_data(site_name, "No telemetry data found in file"), site_name, False
        
        # Check if required variables exist
        expected_columns = ['Date', 'Time', 'Level(RAW)[Main Buffer] (ft)', 'Barometric Pressure(RAW)[Main Buffer] (hPa)']
        missing_columns = [col for col in expected_columns if col not in df.columns]
        if missing_columns:
            return None, site_name, False
        
        initial_count = len(df)
        df['Site'] = site_name
        
        # Whips date into envirosy shape
        try:
            df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S', errors='coerce')
            df['DateTime'] = df['DateTime'].dt.strftime('%d/%m/%Y:00:00:00')
            df = df.dropna(subset=['DateTime'])
            
            if len(df) < initial_count:
                logger.warning(f"Removed {initial_count - len(df)} rows from {site_name} due to invalid Date/Time format.")
                initial_count = len(df)
        except Exception as e:
            logger.error(f"Error processing DateTime for {site_name}: {e}")
            return None, site_name, False
        
        # Convert numeric columns
        numeric_cols = ['Level(RAW)[Main Buffer] (ft)', 'Barometric Pressure(RAW)[Main Buffer] (hPa)']
        df = convert_to_numeric(df, numeric_cols)       
        
        # Drop any invalid values
        pre_filter_count = len(df)
        df = df.dropna(subset=numeric_cols)
        if len(df) < pre_filter_count:
            logger.warning(f"Removed {pre_filter_count - len(df)} rows from {site_name} due to non-numeric values.")       
        
        # Filter based on minimum water level
        df = df[df['Level(RAW)[Main Buffer] (ft)'] >= 0.0001].copy()       
        
        # Do we have any data this month? 
        if df.empty:
            logger.warning(f"No valid data remaining for {site_name} after filtering.")
            return create_placeholder_data(site_name, "No valid telemetry data found"), site_name, False
        
        # Loggers have not been setup the same way. This standardises everything 
        config = SITE_CONFIG.get(site_name, {'depth_conversion_type': 'default'}) 
        conversion_type = config['depth_conversion_type']

        if conversion_type == 'divide_by_100':
            df['Depth(m)raw'] = (df['Level(RAW)[Main Buffer] (ft)'] / 100).round(2)
        else:  # Default or unknown type
            df['Depth(m)raw'] = df['Level(RAW)[Main Buffer] (ft)'].round(2)
        
        # Add comments variable
        df['Comments'] = ""
        df = df.sort_values('DateTime')
        df = df[['Site', 'DateTime', 'Depth(m)raw', 'Barometric Pressure(RAW)[Main Buffer] (hPa)', 'Comments']]
        
        return df, site_name, True
        
    except pd.errors.EmptyDataError:
        logger.warning(f"Skipping {site_name}: File is empty")
        return create_placeholder_data(site_name, "Empty file"), site_name, False
    except pd.errors.ParserError:
        return None, site_name, False
    except Exception:
        return None, site_name, False

def calculate_adjusted_depth(df: pd.DataFrame, water_density: float, logger: logging.Logger, reference_density: float = 1000.0) -> pd.DataFrame:
    """
    This function removes the need to manually calibrate the logger in the field
    by using daily barometric pressure differences between our non-vented 
    AquaTroll 200 loggers and an independent Australian standard weather station.

    ASSUMPTION: Depth data from the loggers is already PARTIALLY COMPENSATED
    using their own internal barometer readings. 

    The Mathematical Justification:
    - Uses AquaTroll 200's hydrostatic pressure formula 
      as specified on page 4 of the product documentation:
      D = 0.70307 * P / SG
      Where:
      - D is depth in meters
      - P is pressure in PSI
      - SG is specific gravity (water_density / reference_density)
      - 0.70307 is the manufacturer's conversion factor

    - Reference: In-Situ Technical Note: Aqua TROLL 200 Measurement Methodology
      https://in-situ.com/pub/media/support/documents/Aqua-TROLL-200-Measurement-Methodology-Tech-Note.pdf

    Args:
        df: DataFrame containing the time series data. Expected columns:
            'Depth(m)raw': Partially compensated depth (m) using internal barometer.
            'Barometric Pressure(RAW)[Main Buffer] (hPa)': Logger's internal barometer reading (hPa).
            'BomBaro': External barometer reading (hPa): Sourced when weatherStaion script called.
        water_density: Density of the water (kg/m^3). Used to calculate specific gravity.
        logger: Uncomment where needed for debugging
        reference_density: For specific gravity calculation (default matches doucmentation: 1000.0 kg/m^3).

    Returns:
        DataFrame with an added 'Depth(m)adjusted' column.
    """
    # Constants for calculation
    HPA_TO_PSI = 0.0145038  # Conversion factor from hectopascals (hPa) to PSI
    CONVERSION_FACTOR = 0.70307  # AquaTroll's conversion factor from PSI to meters per specific gravity
    
    # Calculate specific gravity from water density
    SG = water_density / reference_density

    # --- Input Data Verification and Preparation ---
    required_cols = ['Depth(m)raw', 'Barometric Pressure(RAW)[Main Buffer] (hPa)', 'BomBaro']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.error(f"Missing required columns for depth adjustment: {missing_cols}")
        raise ValueError(f"Missing required columns: {missing_cols}")

    numeric_cols = ['Depth(m)raw', 'Barometric Pressure(RAW)[Main Buffer] (hPa)', 'BomBaro']
    df = convert_to_numeric(df, numeric_cols)

    # Initialise new variable
    df['Depth(m)adjusted'] = np.nan

    # --- Core Calculation Logic ---
    # Identify observations with necessary data for calculation (not NaN)
    mask = (
        df['BomBaro'].notna() &
        df['Barometric Pressure(RAW)[Main Buffer] (hPa)'].notna() &
        df['Depth(m)raw'].notna()
    )

    if mask.sum() > 0:
        # Calculate the difference between the logger's internal barometric reading and the external reading (in hPa)
        # Positive delta_p means logger measured higher pressure than weather station
        delta_p_hpa = df.loc[mask, 'Barometric Pressure(RAW)[Main Buffer] (hPa)'] - df.loc[mask, 'BomBaro']
        
        # Convert the pressure difference from hPa to PSI (as required by AquaTroll formula)
        delta_p_psi = delta_p_hpa * HPA_TO_PSI
        # Calculate depth adjustment using AquaTroll's formula: D = 0.70307 * P / SG
        depth_adjustment_m = CONVERSION_FACTOR * delta_p_psi / SG

        # Apply the adjustment to the partially compensated raw depth
        # Add adjustment because:
        # - If logger's baro reading > weather station: positive adjustment (water is deeper)
        # - If logger's baro reading < weather station: negative adjustment (water is shallower)
        df.loc[mask, 'Depth(m)adjusted'] = df.loc[mask, 'Depth(m)raw'] + depth_adjustment_m
        # Round the final adjusted depth
        df['Depth(m)adjusted'] = df['Depth(m)adjusted'].round(2)
        
        adjusted_count = df.loc[mask, 'Depth(m)adjusted'].notna().sum()
        logger.info(f"{adjusted_count} rows successfully adjusted based on barometer differences.")
    else:
        logger.warning("No rows found with complete data for depth adjustment calculation.")

    return df

# --- Primary function: Main orchestrator of the data validation process ---
def consolidate_csv_files(
    input_folder: str, 
    output_file: str, 
    log_file: Optional[str] = None, 
    enable_file_logging: bool = False,
    water_density: float = 1000, 
    gravity: float = 9.80665
) -> None:
    """
    This function coordinates the entire data validation process.
    It handles:
    1. Processing site files
    2. Retrieving and merging barometric data
    3. Calculating adjusted depths based on barometric differences
    4. Creating the final csv files
    """
    # Load site configuration
    load_site_config()
    
    # Set up logging (if set to True)
    if log_file is None and enable_file_logging:
        log_dir = os.path.dirname(output_file) if os.path.dirname(output_file) else '.'
        log_file = os.path.join(log_dir, 'data_validation.log')
    
    logger = setup_logging(log_file, enable_file_logging)
    logger.info("Starting data validation and consolidation process")
    
    # Check input folder exists
    if not os.path.exists(input_folder):
        return
    
    # Create output directory if needed
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
        except Exception:
            return
    
    # Find all CSV files
    try:
        csv_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.csv')]
        logger.info(f"Found {len(csv_files)} CSV files in the input folder")
    except FileNotFoundError:
        logger.error(f"Input folder not found or inaccessible: {input_folder}")
        return
    
    # Initialize tracking sets
    processed_sites = set()
    sites_with_data = set()
    sites_without_data = set()
    consolidated_data = []
    
    # Process each CSV file
    for filename in csv_files:
        file_path = os.path.join(input_folder, filename)
        site_df, site_name, has_valid_data = process_site_file(file_path, logger)
        
        processed_sites.add(site_name)
        
        if site_df is not None:
            consolidated_data.append(site_df)
            if has_valid_data:
                sites_with_data.add(site_name)
            else:
                sites_without_data.add(site_name)
    
    # Handle missing expected sites
    missing_sites = EXPECTED_SITES - processed_sites
    for site in missing_sites:
        placeholder_df = create_placeholder_data(site, "No telemetry data found")
        consolidated_data.append(placeholder_df)
        sites_without_data.add(site)
    
    # Combine all data and finalise output
    if consolidated_data:
        try:
            # Concatenate dfs 
            final_df = pd.concat(consolidated_data, ignore_index=True)
            
            # Sort and deduplicate
            final_df = final_df.sort_values(['Site', 'DateTime'])
            date_only = final_df['DateTime'].str.split(':', n=1).str[0]
            final_df = final_df.loc[~(final_df['Site'] + date_only).duplicated(keep='first')]
            
            # Get BoM barometric data
            bom_baro_data = get_bom_baro_data(logger)
            
            # Add BomBaro column
            final_df['BomBaro'] = np.nan
            
            # Merge weather data if available
            if bom_baro_data is not None and not bom_baro_data.empty:
                # Convert to numeric
                bom_baro_data['BomBaro'] = pd.to_numeric(bom_baro_data['BomBaro'], errors='coerce')
                
                # Merge data
                final_df = pd.merge(
                    final_df, 
                    bom_baro_data, 
                    on='DateTime', 
                    how='left',
                    suffixes=('', '_bom')
                )
                
                # Update BomBaro columns
                if 'BomBaro_bom' in final_df.columns:
                    final_df['BomBaro'] = final_df['BomBaro'].fillna(final_df['BomBaro_bom'])
                    final_df = final_df.drop(columns=['BomBaro_bom'])
                
                final_df['BomBaro'] = pd.to_numeric(final_df['BomBaro'], errors='coerce')
                bom_data_count = final_df['BomBaro'].notna().sum()
                logger.info(f"Successfully merged barometric data. {bom_data_count} rows have barometric pressure values.")
            else:
                logger.warning("Barometric data unsuccessful (retrieval failed or data was empty/invalid).")
            
            # Calculate adjusted depth
            final_df = calculate_adjusted_depth(final_df, water_density, logger)
            
            # Arrange columns in final order
            column_order = [
                'Site', 
                'DateTime', 
                'BomBaro',
                'Barometric Pressure(RAW)[Main Buffer] (hPa)', 
                'Depth(m)raw',
                'Depth(m)adjusted',
                'Comments'
            ]
            final_df = final_df[column_order]
            
            # Log site statistics
            site_counts = final_df['Site'].value_counts().to_dict()
            for site, count in sorted(site_counts.items()):
                logger.info(f"Site {site}: {count} row(s) in final output")           
            # Save output file
            final_df.to_csv(output_file, index=False, na_rep='')

            # --- Create csv for environmental database (only calibrated observations) ---
            try:
                # Ensure output_dir is defined correctly
                greater_pbo_output_file = os.path.join(output_dir, 'greaterPBOPools.csv')

                # Select required variables & filter for successfully adjusted observations
                pbo_df = final_df.loc[final_df['Depth(m)adjusted'].notna(),
                                     ['Site', 'DateTime', 'Depth(m)adjusted', 'Comments']].copy()

                if not pbo_df.empty:
                    # Rename the depth variable
                    pbo_df.rename(columns={'Depth(m)adjusted': 'Depth'}, inplace=True)

                    # Save the new CSV
                    pbo_df.to_csv(greater_pbo_output_file, index=False, na_rep='')
                    logger.info(f"Envirosys upload file saved to:\n {greater_pbo_output_file} \n")
                else:
                    logger.info(f"No data with adjusted depth found. {greater_pbo_output_file} Not created.")

            except KeyError as e:
                logger.error(f"Error creating greaterPBOPools.csv: Missing column {e}")
            except Exception as e:
                logger.error(f"Error creating greaterPBOPools.csv: {str(e)}", exc_info=True)
            # --- End of environmental database csv creation ---
            
            # Create summary message
            sites_with_data_count = len(sites_with_data)
            sites_without_data_count = len(sites_without_data | missing_sites)
            
            summary_parts = []
            if sites_with_data_count > 0:
                summary_parts.append(f"{sites_with_data_count} site{'s' if sites_with_data_count != 1 else ''} with processed data ({', '.join(sorted(sites_with_data))})")
            if sites_without_data_count > 0:
                summary_parts.append(f"{sites_without_data_count} site{'s' if sites_without_data_count != 1 else ''} with no data or missing files ({', '.join(sorted(sites_without_data | missing_sites))})")
            
            summary_message = ", ".join(summary_parts) + "."
            logger.info(f"Data validation and verification complete. Output saved to:\n {output_file}.\n \n {summary_message}")
            
        except Exception as e:
            logger.error(f"Error during final consolidation, calculation, or saving: {str(e)}", exc_info=True)
    else:
        logger.error("No valid dataframes. Output file not created.")