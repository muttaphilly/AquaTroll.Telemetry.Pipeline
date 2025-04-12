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

# --- Main Scraping Logic ---
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
        logger.info(f"Using download directory: {download_path}")
        logger.info(f"Pause between sites: {PAUSE_SECONDS} seconds")

        # Process each site
        for i, site in enumerate(sites_to_scrape): 
            browser = None
            context = None
            page = None
            site_name = site["name"]
            nav_option = site["nav_option"]
            main_csv_filename = site["main_csv"]
            baro_csv_filename = site["baro_csv"]

            # Construct paths using download_path
            main_download_path = os.path.join(download_path, main_csv_filename)
            baro_download_path = os.path.join(download_path, baro_csv_filename)

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
                logger.info("Waiting for post-login element...")
                page.wait_for_selector("#ctlMenu_ctlNodeMenu_gridNodeList_lstNodes_1", timeout=60000)
                logger.info(f"Logged in successfully for {site_name}")

                # Navigate to the correct channel
                logger.info(f"Navigating to node option: {nav_option}")
                page.locator("#ctlMenu_ctlNodeMenu_gridNodeList_lstNodes_1").select_option("-2")
                page.wait_for_load_state("networkidle", timeout=60000)
                page.locator("#ctlMenu_ctlNodeMenu_gridNodeList_lstNodes_1").select_option(nav_option)
                page.wait_for_load_state("networkidle", timeout=60000)
                page.locator("#ctlMenu_ctlNodeMenu_gridNodeList_hrefViewNode_1").click()
                page.wait_for_load_state("networkidle", timeout=60000)
                logger.info(f"Navigation complete for {site_name}")

                # Download Main Data (Level RAW)
                logger.info(f"Downloading Level(RAW) data for {site_name}")
                page.get_by_role("link", name="Level(RAW)[Main Buffer]").click()
                page.wait_for_load_state("networkidle", timeout=60000)
                page.get_by_role("radio", name="Table").check()
                page.get_by_role("radio", name="30 Days").check()
                page.wait_for_load_state("networkidle", timeout=60000)
                with page.expect_download(timeout=120000) as download_info:
                    page.get_by_role("button", name="Export Data to CSV File").click()
                download = download_info.value
                # Use main_download_path constructed with the argument
                download.save_as(main_download_path)
                logger.info(f"Level(RAW) data saved as: {main_download_path}")

                # Download Barometric Data
                logger.info(f"Downloading Barometric Pressure(RAW) data for {site_name}")
                # Navigate back or directly to the baro link if possible
                try:
                    page.get_by_role("link", name="Displayed").click(timeout=10000)
                except Exception:
                    logger.warning("Could not click 'Displayed' link, proceeding.")
                    pass # Continue even if 'Displayed' link isn't found or clickable

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
                merge_csv_files(main_download_path, baro_download_path)

                logger.info(f"--- Successfully processed site: {site_name} ---")

            except Exception as site_error:
                logger.exception(f"An error occurred while processing site {site_name}: {site_error}")
                # Decide if you want to continue to the next site or stop
                # continue # Uncomment to continue with the next site despite error
            finally:
                # Close browser context and instance FOR EACH SITE
                if context:
                    context.close()
                    logger.info(f"Playwright context closed for {site_name}.")
                if browser:
                    browser.close()
                    logger.info(f"Playwright browser closed for {site_name}.")

            # Pause before processing the next site
            # Use index 'i' from enumerate to check if it's not the last site
            if i < len(sites_to_scrape) - 1:
                 logger.info(f"Pausing for {PAUSE_SECONDS} seconds before next site...")
                 time.sleep(PAUSE_SECONDS)

    except Exception as e:
        # Catch exceptions outside the loop (e.g., initial setup errors)
        logger.exception(f"A critical error occurred outside the site loop: {e}")
    finally:
        logger.info("--- Playwright Scraper Finished ---")

# --- Merge depth and barometric data into single csv ---

def merge_csv_files(main_data_path, baro_path):
    """
    Merge the main site data CSV with the barometric data CSV using Date and Time join.
    """
    if not os.path.exists(main_data_path):
        logger.error(f"Main data file not found, skipping merge: {main_data_path}")
        return
    if not os.path.exists(baro_path):
        logger.error(f"Barometric data file not found, skipping merge: {baro_path}")
        return

    try:
        logger.info(f"Attempting to merge {main_data_path} and {baro_path}")
        # Read both CSV files
        main_df = pd.read_csv(main_data_path)
        baro_df = pd.read_csv(baro_path)

        # Check if required columns exist
        if not {'Date', 'Time'}.issubset(main_df.columns):
             logger.error(f"Missing 'Date' or 'Time' column in {main_data_path}")
             return
        if not {'Date', 'Time', 'Barometric Pressure(RAW)[Main Buffer] (hPa)'}.issubset(baro_df.columns):
             logger.error(f"Missing required columns in {baro_path}")
             return

        # Merge
        merged_df = pd.merge(
            main_df,
            baro_df[['Date', 'Time', 'Barometric Pressure(RAW)[Main Buffer] (hPa)']],
            on=['Date', 'Time'],
            how='left'
        )

        # Overwrite original depth data csv
        merged_df.to_csv(main_data_path, index=False)
        logger.info(f"Successfully merged data and saved to {main_data_path}")

    except Exception as e:
        logger.exception(f"Error merging CSV files ({main_data_path}, {baro_path}): {str(e)}")
