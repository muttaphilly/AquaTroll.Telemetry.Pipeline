import os
import json
import time
import logging
import pandas as pd
from playwright.sync_api import Playwright, sync_playwright, expect
from dotenv import load_dotenv

# --- Configuration ---

# Load environment variables from .env file
load_dotenv()

LOGIN_URL = os.getenv("LOGIN_URL")
LOGIN_USERNAME = os.getenv("LOGIN_USERNAME")
LOGIN_PASSWORD = os.getenv("LOGIN_PASSWORD")
PAUSE_SECONDS = int(os.getenv("PAUSE_SECONDS"))
sites_json = os.getenv("SITES_JSON", "[]")
sites_to_scrape = json.loads(sites_json)

# Setup debug logging
logging.basicConfig(
    filename="script.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Sauce pools
def get_enabled_sites():
    """Grab from ENABLED_SITES list"""
    sites_json = os.getenv("SITES_JSON", "[]")
    all_sites = json.loads(sites_json)
    enabled_sites_str = os.getenv("ENABLED_SITES", "")
    if enabled_sites_str:
        enabled_names = [name.strip() for name in enabled_sites_str.split(",")]
        return [site for site in all_sites if site["main_csv"].replace(".csv", "") in enabled_names]
    # If not set, return all sites
    return all_sites

# --- Main Scraping Logic ---
# Uses PlayWright to scrape the frontend
def run(playwright: Playwright, download_path: str) -> None:
    """
    Main function to orchestrate the scraping process for all configured sites.
    Accepts the target download directory path as an argument.
    """
    try:
        logger.info("--- Starting Playwright Scraper ---")
        # Validate environment variables & project folder
        if not all([LOGIN_URL, LOGIN_USERNAME, LOGIN_PASSWORD]):
            logger.error("Missing environment variables (URL, USERNAME, PASSWORD). Exiting.")
            return

        if not download_path:
             logger.error("Missing download_path argument. Exiting.")
             return
        # Ensure download directory exists
        os.makedirs(download_path, exist_ok=True)
        
        # Get only enabled sites
        sites_to_scrape = get_enabled_sites()
        logger.info(f"Processing {len(sites_to_scrape)} enabled sites")

        # Process each site
        for i, site in enumerate(sites_to_scrape): 
            browser = None
            context = None
            page = None
            site_name = site["name"]
            nav_option = site["nav_option"]
            main_csv_filename = site["main_csv"]
            baro_csv_filename = site["baro_csv"]
            
            # Check if this is a PSI-based site
            uses_psi = (nav_option == "18096")  # ERP3

            # Construct paths using download_path
            main_download_path = os.path.join(download_path, main_csv_filename)
            baro_download_path = os.path.join(download_path, baro_csv_filename)

            # The main URL for the site's data channels list
            target_url = f"https://neon1.unidata.com.au/data-channels.aspx?id={nav_option}"

            logger.info(f"--- Processing site: {site_name} ({i+1}/{len(sites_to_scrape)}) ---")

            try:
                # Launch browser & Login (For each site)
                logger.info(f"Launching browser for {site_name}")
                browser = playwright.firefox.launch(headless=True)
                context = browser.new_context()
                page = context.new_page()

                logger.info(f"Navigating to login page: {LOGIN_URL}")
                page.goto(LOGIN_URL)
                page.locator("#txtUsername").fill(LOGIN_USERNAME)
                page.locator("#txtPassword").click()
                page.locator("#txtPassword").fill(LOGIN_PASSWORD)
                page.get_by_role("button", name="Login").click()
                logger.info("Login successful. Waiting for initial page load to complete...")
                page.wait_for_load_state("networkidle", timeout=60000)

                # --- Download Main Data (Level RAW) ---
                logger.info(f"Navigating to {target_url} for Level data")
                page.goto(target_url)
                page.wait_for_load_state("networkidle", timeout=60000)
                
                logger.info(f"Downloading Level(RAW) data for {site_name}")
                page.get_by_role("link", name="Level(RAW)[Main Buffer]").click()
                page.wait_for_load_state("networkidle", timeout=60000)
                page.get_by_role("radio", name="Table").check()
                page.get_by_role("radio", name="30 Days").check()
                page.wait_for_load_state("networkidle", timeout=60000)
                with page.expect_download(timeout=120000) as download_info:
                    page.get_by_role("button", name="Export Data to CSV File").click()
                download = download_info.value
                download.save_as(main_download_path)
                logger.info(f"Level(RAW) data saved as: {main_download_path}")

                # --- Download Barometric Data ---
                logger.info("Resetting page state by navigating back to the main data channel list.")
                page.goto(target_url)
                page.wait_for_load_state("networkidle", timeout=60000)
                
                if uses_psi:
                # For sites using PSI (ERP3)
                    logger.info(f"Downloading Pressure(RAW) data for {site_name} (PSI-based site)")
                    page.get_by_role("link", name="Pressure(RAW)[Main").click()
                else:
                # Default sites using hPa
                    logger.info(f"Downloading Barometric Pressure(RAW) data for {site_name}")
                    page.get_by_role("link", name="Barometric Pressure(RAW)[Main").click()
            
                page.wait_for_load_state("networkidle", timeout=60000)
                page.get_by_role("radio", name="Table").check()
                page.get_by_role("radio", name="30 Days").check()
                page.wait_for_load_state("networkidle", timeout=60000)
                with page.expect_download(timeout=120000) as download1_info:
                    page.get_by_role("button", name="Export Data to CSV File").click()
                download1 = download1_info.value
                download1.save_as(baro_download_path)
                logger.info(f"Barometric data saved as: {baro_download_path}")

                # Merge the downloaded CSVs
                merge_csv_files(main_download_path, baro_download_path, nav_option)
                logger.info(f"--- Successfully processed site: {site_name} ---")

            except Exception as site_error:
                logger.exception(f"An error occurred while processing site {site_name}: {site_error}")
            finally:
                if context:
                    context.close()
                    logger.info(f"Playwright context closed for {site_name}.")
                if browser:
                    browser.close()
                    logger.info(f"Playwright browser closed for {site_name}.")

            if i < len(sites_to_scrape) - 1:
                 logger.info(f"Pausing for {PAUSE_SECONDS} seconds before next site...")
                 time.sleep(PAUSE_SECONDS)

    except Exception as e:
        logger.exception(f"A critical error occurred outside the site loop: {e}")
    finally:
        logger.info("--- Playwright Scraper Finished ---")

# --- Merge depth and barometric data into single csv ---

def merge_csv_files(main_data_path, baro_path, nav_option=None):
    """
    Merge the main site data CSV with the barometric/pressure data CSV using Date and Time join.
    Handles both hPa and PSI based sites.
    CRITICAL: Preserves all original columns from both files.
    """
    if not os.path.exists(main_data_path):
        logger.error(f"Main data file not found, skipping merge: {main_data_path}")
        return
    if not os.path.exists(baro_path):
        logger.error(f"Barometric/Pressure data file not found, skipping merge: {baro_path}")
        return

    try:
        logger.info(f"Attempting to merge {main_data_path} and {baro_path}")
        
        # Read both CSV files
        main_df = pd.read_csv(main_data_path)
        baro_df = pd.read_csv(baro_path)
        
        # Log the columns found in each file for debugging
        logger.info(f"Main CSV columns: {list(main_df.columns)}")
        logger.info(f"Baro CSV columns: {list(baro_df.columns)}")
        
        # Check if main data has required columns
        if not {'Date', 'Time'}.issubset(main_df.columns):
            logger.error(f"Missing 'Date' or 'Time' column in main data file: {main_data_path}")
            logger.error(f"Available columns: {list(main_df.columns)}")
            return
        
        # Check if baro data has required columns
        if not {'Date', 'Time'}.issubset(baro_df.columns):
            logger.error(f"Missing 'Date' or 'Time' column in baro file: {baro_path}")
            logger.error(f"Available columns: {list(baro_df.columns)}")
            return
        
        # Lock in PSI if JSON nav_option matches
        uses_psi = (nav_option == "18096" or nav_option == "15619")
        
        if uses_psi:
            pressure_col = 'Pressure(RAW)[Main Buffer] (PSI)'
        else:
            pressure_col = 'Barometric Pressure(RAW)[Main Buffer] (hPa)'
        
        # Check if the expected pressure column exists
        if pressure_col not in baro_df.columns:
            logger.warning(f"Expected column '{pressure_col}' not found in {baro_path}")
            logger.warning(f"Looking for any pressure-related column...")
            # Oh pressure, where you at?
            pressure_columns = [col for col in baro_df.columns if 'Pressure' in col or 'pressure' in col]
            if pressure_columns:
                pressure_col = pressure_columns[0]
                logger.info(f"Found pressure column: {pressure_col}")
            else:
                logger.error(f"No pressure column found in {baro_path}")
                return
        
        # Get the pressure column(s) to merge - no Date and Time cause they're the merge keys
        baro_columns_to_merge = [col for col in baro_df.columns if col not in ['Date', 'Time']]
        
        logger.info(f"Merging with pressure columns: {baro_columns_to_merge}")
        
        # Perform the merge - keep all columns from both dataframes
        merged_df = pd.merge(
            main_df,
            baro_df,
            on=['Date', 'Time'],
            how='left'  # Keep all rows from main_df even if no matching baro data
        )
        
        # Verify merge preserved expected columns
        essential_cols = ['Date', 'Time', 'Level(RAW)[Main Buffer] (ft)']
        missing_essential = [col for col in essential_cols if col not in merged_df.columns]
        
        if missing_essential:
            logger.error(f"Merge failed - missing essential columns: {missing_essential}")
            logger.error(f"Merged DataFrame columns: {list(merged_df.columns)}")
            logger.error("NOT overwriting original file due to failed merge")
            return
        
        # Check merge hasn't derped rows
        if merged_df.empty:
            logger.error(f"Merge resulted in empty DataFrame. NOT overwriting original file.")
            return
        
        # Log successful merge info
        logger.info(f"Successfully merged. Result has {len(merged_df)} rows and {len(merged_df.columns)} columns")
        logger.info(f"Final columns: {list(merged_df.columns)}")
        
        # Save the merged data back to the main file
        merged_df.to_csv(main_data_path, index=False)
        logger.info(f"Merged data saved to {main_data_path}")

    except Exception as e:
        logger.exception(f"Error merging CSV files ({main_data_path}, {baro_path}): {str(e)}")
        logger.error("Original file NOT overwritten due to merge error")