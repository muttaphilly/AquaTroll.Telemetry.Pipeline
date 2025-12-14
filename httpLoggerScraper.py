#!/usr/bin/env python3
"""
HTTP Logger Scraper
- Handles website authentication & both Tree Menu and Node Menu views.
- Retrieves AquaTroll sensor depth, barometric data, and battery voltage.
- Creates csv file for each site & saves to data_downloads folder
"""

import os
import re
import json
import time
import logging
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from datetime import datetime, timedelta
import pandas as pd
from urllib.parse import urlparse

# ------------ Load environment variables ------------
load_dotenv()

BASE_URL = os.getenv("BASE_URL", "").rstrip("/")
LOGIN_PATH = os.getenv("LOGIN_PATH", "/logon.aspx")
LOGIN_URL = f"{BASE_URL}{LOGIN_PATH}"
LOGIN_USERNAME = os.getenv("LOGIN_USERNAME")
LOGIN_PASSWORD = os.getenv("LOGIN_PASSWORD")
PAUSE_SECONDS = int(os.getenv("PAUSE_SECONDS"))

# ------------ Setup logging -------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler('http_scraper.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ------------ Meat & Potatoes -----------------------
class UnidataHTTPScraper:
    """Scraper with dual tree/node menu support"""
    
    def __init__(self, base_url, username, password):
        self.base_url = base_url.rstrip('/')
        self.username = username
        self.password = password
        self.session = requests.Session()
        
        # Use exact headers from browser
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:144.0) Gecko/20100101 Firefox/144.0',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Sec-GPC': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        })
        self.current_view = None
        self.login_successful = False
    
    def extract_form_fields(self, html_content, include_buttons=False):
        """Extract all form fields from HTML"""
        soup = BeautifulSoup(html_content, 'html.parser')
        fields = {}
        
        # Find the main form
        form = soup.find('form')
        if not form:
            logger.warning("No form found in page")
            return fields
        
        # Extract __VIEWSTATE and related fields
        for field_name in ['__VIEWSTATE', '__VIEWSTATEGENERATOR', '__EVENTVALIDATION']:
            field = soup.find('input', {'name': field_name})
            if field:
                fields[field_name] = field.get('value', '')
                logger.debug(f"Found {field_name}: {len(fields[field_name])} chars")
        
        # Get all input fields
        for input_elem in soup.find_all('input'):
            name = input_elem.get('name')
            if not name:
                continue
                
            input_type = input_elem.get('type', 'text').lower()
            
            # Handle different input types
            if input_type in ['hidden', 'text', 'password']:
                fields[name] = input_elem.get('value', '')
            elif input_type == 'checkbox':
                if input_elem.get('checked'):
                    fields[name] = 'on'
            elif input_type == 'radio':
                if input_elem.get('checked'):
                    fields[name] = input_elem.get('value', '')
            elif input_type == 'submit' and include_buttons:
                # Only include if specifically requested
                fields[name] = input_elem.get('value', '')
        
        # Get all select fields
        for select_elem in soup.find_all('select'):
            name = select_elem.get('name')
            if name:
                selected = select_elem.find('option', {'selected': True})
                if selected:
                    fields[name] = selected.get('value', '')
                else:
                    first_option = select_elem.find('option')
                    if first_option:
                        fields[name] = first_option.get('value', '')
        
        # Get all textarea fields
        for textarea in soup.find_all('textarea'):
            name = textarea.get('name')
            if name:
                fields[name] = textarea.get_text('', strip=True)
        
        return fields
    
    def login(self):
        max_attempts = 3
        
        for attempt in range(1, max_attempts + 1):
            try:
                logger.info(f"Login attempt {attempt}/{max_attempts} to logger portal...")
                
                # Clear any existing cookies
                self.session.cookies.clear()
                
                # Step 1: GET the login page
                logger.info(f"  Fetching login page: {LOGIN_URL}")
                login_page = self.session.get(LOGIN_URL, timeout=30)
                
                if login_page.status_code != 200:
                    logger.error(f"  Failed to access login page: {login_page.status_code}")
                    continue
                
                logger.info(f"  Login page loaded successfully ({len(login_page.text)} bytes)")
                
                # Step 2: Extract ALL form fields
                form_data = self.extract_form_fields(login_page.text)
                logger.info(f"  Extracted {len(form_data)} form fields")
                
                # Step 3: Add our credentials
                form_data['txtUsername'] = self.username
                form_data['txtPassword'] = self.password
                form_data['btnLogon'] = 'Login'
                
                # Remove any conflicting submit buttons
                for key in list(form_data.keys()):
                    if key.startswith('btn') and key != 'btnLogon':
                        del form_data[key]
                
                # Ensure __EVENTTARGET and __EVENTARGUMENT are set
                form_data['__EVENTTARGET'] = ''
                form_data['__EVENTARGUMENT'] = ''
                
                # Debug: Log form data (redact password)
                debug_form = {k: v if k != 'txtPassword' else '***' for k, v in form_data.items()}
                logger.debug(f"  Form data being submitted: {debug_form}")
                
                logger.info(f"  Submitting login for user: {self.username}")
                
                # Step 4: POST the login
                login_response = self.session.post(
                    LOGIN_URL, 
                    data=form_data,
                    headers={
                        'Content-Type': 'application/x-www-form-urlencoded',
                        'Referer': LOGIN_URL,
                        'Origin': self.base_url
                    },
                    timeout=30,
                    allow_redirects=True
                )
                
                logger.info(f"  Login response: {login_response.status_code}")
                logger.info(f"  Final URL: {login_response.url}")
                
                # Step 5: Check if login was successful
                # Success indicators:
                # - Redirected away from login page
                # - No login form fields in response
                # - Session cookie present
                
                current_path = urlparse(login_response.url).path.lower()
                
                # Check various success indicators
                login_failed = False
                
                # Check if still on login page
                if 'logon' in current_path or 'login' in current_path:
                    logger.warning("  Still on login page based on URL")
                    login_failed = True
                
                # Check for login form elements in response
                if 'txtUsername' in login_response.text or 'txtPassword' in login_response.text:
                    logger.warning("  Login form fields still present in response")
                    login_failed = True
                
                # Check for error messages
                if 'invalid' in login_response.text.lower() or 'incorrect' in login_response.text.lower():
                    logger.warning("  Possible error message in response")
                    # Extract and log any error messages
                    soup = BeautifulSoup(login_response.text, 'html.parser')
                    error_elem = soup.find('span', class_='error') or soup.find('div', class_='alert')
                    if error_elem:
                        logger.error(f"  Error message: {error_elem.get_text(strip=True)}")
                    login_failed = True
                
                # Check for expected post-login elements
                if 'logout' in login_response.text.lower() or 'logoff' in login_response.text.lower():
                    logger.info("  Found logout link - good sign!")
                    login_failed = False
                
                if not login_failed:
                    logger.info("✅ Login successful!")
                    self.login_successful = True
                    
                    # Save cookies for debugging
                    logger.debug(f"  Session cookies: {list(self.session.cookies.keys())}")
                    
                    return True
                else:
                    logger.error(f"  Login attempt {attempt} failed")
                    
                    # Save debug info
                    if attempt == max_attempts:
                        debug_file = f"login_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
                        with open(debug_file, 'w', encoding='utf-8') as f:
                            f.write(login_response.text)
                        logger.info(f"  Debug HTML saved to: {debug_file}")
                    
                    # Wait before retry
                    if attempt < max_attempts:
                        wait_time = attempt * 5
                        logger.info(f"  Waiting {wait_time} seconds before retry...")
                        time.sleep(wait_time)
                
            except requests.exceptions.RequestException as e:
                logger.error(f"  Network error during login: {e}")
                if attempt < max_attempts:
                    time.sleep(5)
                continue
            except Exception as e:
                logger.error(f"  Unexpected error during login: {e}")
                import traceback
                traceback.print_exc()
                if attempt < max_attempts:
                    time.sleep(5)
                continue
        
        logger.error("❌ All login attempts failed")
        return False
    
    def detect_menu_view(self, html_content):
        """Detect whether the page is in tree menu or node menu view"""
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Method 1: Check for menu mode checkbox
        menu_checkbox = soup.find('input', {'name': 'ctlMenu$chkMenuMode'})
        if menu_checkbox:
            if menu_checkbox.get('checked'):
                logger.info("  Detected: TREE MENU view (checkbox checked)")
                return 'tree'
            else:
                logger.info("  Detected: NODE MENU view (checkbox unchecked)")
                return 'node'
        
        # Method 2: Check for specific menu panels
        tree_panel = soup.find('div', {'id': 'ctlMenu_ctlTreeMenu_pnlTreeMenu'})
        node_panel = soup.find('div', {'id': 'ctlMenu_ctlNodeMenu_pnlNodeMenu'})
        
        if tree_panel and tree_panel.get('style', '').find('display:none') == -1:
            logger.info("  Detected: TREE MENU view (panel visible)")
            return 'tree'
        elif node_panel and node_panel.get('style', '').find('display:none') == -1:
            logger.info("  Detected: NODE MENU view (panel visible)")
            return 'node'
        
        # Method 3: Check for specific form elements
        tree_elements = soup.find('input', {'name': re.compile(r'ctlMenu\$ctlTreeMenu')})
        node_elements = soup.find('select', {'name': re.compile(r'ctlMenu\$ctlNodeMenu')})
        
        if node_elements:
            logger.info("  Detected: NODE MENU view (node elements found)")
            return 'node'
        elif tree_elements:
            logger.info("  Detected: TREE MENU view (tree elements found)")
            return 'tree'
        
        logger.warning("  Could not definitively detect menu view, assuming TREE")
        return 'tree'
    
    def switch_to_table_mode(self, channel_url, page_content):
        """Switch to table mode - handles both tree and node menu views"""
        logger.info("  Step 2: Switching to Table mode...")
        
        # Detect current view
        self.current_view = self.detect_menu_view(page_content)
        
        # Extract all current form fields
        fields = self.extract_form_fields(page_content)
        
        # Set up date range (30 days)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        # Core fields for switching to table mode
        fields['__EVENTTARGET'] = 'rdoDisplayType$1'
        fields['__EVENTARGUMENT'] = ''
        fields['rdoDisplayType'] = 'TABLE_DISPLAY'
        
        # Set date fields
        fields['dateStart$txt_Date'] = start_date.strftime('%d/%m/%Y')
        fields['dateEnd$txt_Date'] = end_date.strftime('%d/%m/%Y')
        
        # Set data period to 30 days
        fields['rblDataPeriod'] = '720'
        
        # Remove any image button coordinates that might interfere
        for key in list(fields.keys()):
            if key.endswith('.x') or key.endswith('.y'):
                del fields[key]
        
        # If in NODE menu, ensure menu fields are preserved
        if self.current_view == 'node':
            # Preserve node menu selections
            soup = BeautifulSoup(page_content, 'html.parser')
            for select_elem in soup.find_all('select'):
                name = select_elem.get('name')
                if name and 'ctlMenu$ctlNodeMenu' in name:
                    selected = select_elem.find('option', {'selected': True})
                    if selected:
                        fields[name] = selected.get('value', '')
        
        # Debug: Log form data
        logger.debug(f"  Form data for table switch: {fields}")
        
        logger.info(f"  Posting {len(fields)} fields to switch to table mode")
        
        # Submit the form with retry for 500 errors
        max_switch_retries = 2
        for retry in range(max_switch_retries):
            try:
                response = self.session.post(
                    channel_url,
                    data=fields,
                    headers={
                        'Content-Type': 'application/x-www-form-urlencoded',
                        'Referer': channel_url
                    },
                    timeout=30
                )
                response.raise_for_status()  # Raise on 4xx/5xx
                # Check if now in table mode
                if 'TABLE_DISPLAY' in response.text and 'btnExportToCSV' in response.text:
                    logger.info("  ✅ Successfully switched to table mode")
                    return response.text
                else:
                    logger.warning("  Response received but table mode not confirmed")
                    # Save for debugging
                    debug_file = f"table_switch_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
                    with open(debug_file, 'w', encoding='utf-8') as f:
                        f.write(response.text)
                    logger.info(f"  Debug HTML saved to: {debug_file}")
                    return response.text
            except requests.exceptions.HTTPError as e:
                if '500' in str(e) and retry < max_switch_retries - 1:
                    logger.warning(f"  500 error on table switch, retrying in 5s...")
                    time.sleep(5)
                    continue
                logger.error(f"  Failed to switch to table mode: {e}")
                return None
        
        return None
    
    def download_csv(self, channel_url, table_page_content):
        """Download CSV from table view"""
        logger.info("  Step 3: Downloading CSV...")
        
        # Extract form fields from table page
        fields = self.extract_form_fields(table_page_content)
        
        # Check if CSV button exists
        soup = BeautifulSoup(table_page_content, 'html.parser')
        csv_button = soup.find('input', {'name': 'btnExportToCSV'})
        
        if not csv_button:
            logger.error("  CSV export button not found on page")
            return None
        
        # Set up for CSV download
        if csv_button.get('type') == 'image':
            # Image button - need to send coordinates
            fields['btnExportToCSV.x'] = '50'
            fields['btnExportToCSV.y'] = '15'
        else:
            # Regular button
            fields['btnExportToCSV'] = 'Export to CSV'
        
        # Clear event target for button click
        fields['__EVENTTARGET'] = ''
        fields['__EVENTARGUMENT'] = ''
        
        # Debug: Log form data
        logger.debug(f"  Form data for CSV download: {fields}")
        
        logger.info(f"  Posting {len(fields)} fields to download CSV")
        
        # Submit for CSV download
        response = self.session.post(
            channel_url,
            data=fields,
            headers={
                'Content-Type': 'application/x-www-form-urlencoded',
                'Referer': channel_url
            },
            timeout=60
        )
        
        if response.status_code == 200:
            # Check there is CSV data
            content_type = response.headers.get('Content-Type', '').lower()
            
            if 'csv' in content_type or 'octet-stream' in content_type:
                logger.info(f"  ✅ CSV downloaded: {len(response.content):,} bytes")
                return response.content
            elif response.content.startswith(b'Date,Time') or b',' in response.content[:100]:
                # Sometimes CSV comes without proper content-type
                logger.info(f"  ✅ CSV downloaded (detected by content): {len(response.content):,} bytes")
                return response.content
            else:
                logger.warning("  Response doesn't appear to be CSV")
                # Check first few bytes
                preview = response.content[:500].decode('utf-8', errors='ignore')
                if 'Date' in preview and 'Time' in preview:
                    logger.info("  But content looks like CSV, using it anyway")
                    return response.content
                else:
                    logger.error(f"  Content preview: {preview[:200]}...")
                    return None
        else:
            logger.error(f"  CSV download failed: {response.status_code}")
            return None
    
    def download_channel_data(self, channel_id, save_path):
        """Download data for a specific channel with retry logic"""
        channel_url = f"{self.base_url}/node-data-channel-am.aspx?id={channel_id}"
        
        max_retries = 2
        for retry in range(max_retries):
            try:
                if retry > 0:
                    logger.info(f"  Retry {retry}/{max_retries-1} for channel {channel_id}")
                
                logger.info(f"  Step 1: Fetching channel page: {channel_url}")
                
                # Get channel page
                response = self.session.get(channel_url, timeout=30)
                if response.status_code != 200:
                    logger.error(f"  Failed to access channel page: {response.status_code}")
                    continue
                
                # Check still logged in..
                if 'logon.aspx' in response.url.lower() or 'txtUsername' in response.text:
                    logger.warning("  Session expired, need to re-login")
                    if self.login():
                        # Retry getting the channel page
                        response = self.session.get(channel_url, timeout=30)
                    else:
                        logger.error("  Re-login failed")
                        return False
                
                logger.info(f"  Channel page loaded ({len(response.text)} bytes)")
                
                # Switch to table mode
                table_page = self.switch_to_table_mode(channel_url, response.text)
                if not table_page:
                    logger.error("  Failed to switch to table mode")
                    continue
                
                # Download CSV
                csv_data = self.download_csv(channel_url, table_page)
                if csv_data:
                    # Save the file
                    with open(save_path, 'wb') as f:
                        f.write(csv_data)
                    file_size = os.path.getsize(save_path)
                    logger.info(f"  ✅ Saved: {save_path} ({file_size:,} bytes)")
                    return True
                else:
                    logger.error("  Failed to download CSV")
                    if retry < max_retries - 1:
                        logger.info("  Waiting before retry...")
                        time.sleep(10)
                
            except Exception as e:
                logger.error(f"  Error downloading channel {channel_id}: {e}")
                import traceback
                traceback.print_exc()
                if retry < max_retries - 1:
                    time.sleep(10)
        
        return False

def load_sites_config():
    """Load sites configuration from environment variable"""
    sites_config_json = os.getenv('SITES_CONFIG', '{}')
    try:
        config = json.loads(sites_config_json)
        return config if isinstance(config, dict) else {}
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse SITES_CONFIG: {e}")
        return {}

# grab JSON from .env
def get_enabled_sites():
    """Get list of enabled sites with channel IDs"""
    all_sites = load_sites_config()
    enabled_sites = []
    
    for site_id, config in all_sites.items():
        if config.get('enabled', False):
            site_info = {
                'site_id': site_id,
                'display_name': config.get('display_name', site_id),
                'level_channel_id': config.get('level_channel_id'),
                'baro_channel_id': config.get('baro_channel_id'),
                'battery_channel_id': config.get('battery_channel_id'),
                'pressure_unit': config.get('pressure_unit', 'hpa')
            }
            
            if site_info['level_channel_id'] and site_info['baro_channel_id'] and site_info['battery_channel_id']:
                enabled_sites.append(site_info)
            else:
                logger.warning(f"Site {site_id} missing channel IDs, skipping")
    
    return enabled_sites

def merge_csv_files(level_path, baro_path, battery_path, site_id):
    """Merge level, barometric, and battery data CSVs"""
    try:
        level_df = pd.read_csv(level_path)
        baro_df = pd.read_csv(baro_path)
        battery_df = pd.read_csv(battery_path)
        
        logger.info(f"  Level data: {len(level_df)} rows, columns: {list(level_df.columns)[:5]}...")
        logger.info(f"  Baro data: {len(baro_df)} rows, columns: {list(baro_df.columns)[:5]}...")
        logger.info(f"  Battery data: {len(battery_df)} rows, columns: {list(battery_df.columns)[:5]}...")
        
        # Merge on Date and Time columns
        if 'Date' in level_df.columns and 'Time' in level_df.columns:
            merged_df = pd.merge(level_df, baro_df, on=['Date', 'Time'], how='left')
            merged_df = pd.merge(merged_df, battery_df, on=['Date', 'Time'], how='left')
            
            # Save merged data
            merged_df.to_csv(level_path, index=False)
            logger.info(f"  ✅ Merged: {len(merged_df)} rows, {len(merged_df.columns)} columns")
            logger.info(f"  💾 Saved to {os.path.basename(level_path)}")
        else:
            logger.warning("  Date/Time columns not found, skipping merge")
        
    except Exception as e:
        logger.error(f"  Error merging CSV files: {e}")


def run(download_path):
    """Main function to run the HTTP scraper"""
    logger.info("="*70)
    logger.info("HTTP Logger Scraper Starting")
    logger.info("="*70)
    logger.info(f"Download directory: {download_path}")
    
    # Validate configuration
    if not BASE_URL:
        logger.error("BASE_URL not configured in .env file")
        return
    if not LOGIN_USERNAME or not LOGIN_PASSWORD:
        logger.error("LOGIN_USERNAME or LOGIN_PASSWORD not configured in .env file")
        return
    
    logger.info(f"Target server: {BASE_URL}")
    logger.info(f"Login user: {LOGIN_USERNAME}")
    
    # Create download directory
    os.makedirs(download_path, exist_ok=True)
    
    # Get enabled sites
    sites = get_enabled_sites()
    if not sites:
        logger.error("No enabled sites with channel IDs found")
        logger.info("Check SITES_CONFIG in your .env file")
        return
    
    logger.info(f"Processing {len(sites)} enabled sites")
    
    # Initialise scraper
    scraper = UnidataHTTPScraper(BASE_URL, LOGIN_USERNAME, LOGIN_PASSWORD)
    
    # Login once
    if not scraper.login():
        logger.error("Login failed. Exiting.")
        logger.info("Please check:")
        logger.info("  1. BASE_URL is correct")
        logger.info("  2. LOGIN_USERNAME and LOGIN_PASSWORD are correct")
        logger.info("  3. Network connectivity to the server")
        logger.info("  4. Check login_debug_*.html file for details")
        return
    
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    logger.info(f"Date range: {start_date.strftime('%d/%m/%Y')} to {end_date.strftime('%d/%m/%Y')}")
    
    # Process each site
    successful = 0
    failed = 0
    
    for i, site in enumerate(sites, 1):
        site_id = site['site_id']
        display_name = site['display_name']
        
        logger.info("")
        logger.info("="*70)
        logger.info(f"Site {i}/{len(sites)}: {display_name} ({site_id})")
        logger.info("="*70)
        logger.info(f"  Level channel: {site['level_channel_id']}")
        logger.info(f"  Baro channel: {site['baro_channel_id']}")
        logger.info(f"  Battery channel: {site['battery_channel_id']}")
        
        try:
            # Download Level data
            logger.info("📊 Downloading Level(RAW) data...")
            level_path = os.path.join(download_path, f"{site_id}.csv")
            level_ok = scraper.download_channel_data(site['level_channel_id'], level_path)
            
            if not level_ok:
                logger.error(f"Failed to download Level data for {site_id}")
                failed += 1
                continue
            
            # Download Barometric data
            baro_type = "Pressure(RAW)" if site['pressure_unit'] == 'psi' else "Barometric Pressure(RAW)"
            logger.info(f"🌡️ Downloading {baro_type} data...")
            baro_path = os.path.join(download_path, f"baro{site_id}.csv")
            baro_ok = scraper.download_channel_data(site['baro_channel_id'], baro_path)
            
            if not baro_ok:
                logger.error(f"Failed to download Barometric data for {site_id}")
                failed += 1
                continue
            
            # Download Battery Voltage data
            logger.info("🔋 Downloading Battery Voltage data...")
            battery_path = os.path.join(download_path, f"battery{site_id}.csv")
            battery_ok = scraper.download_channel_data(site['battery_channel_id'], battery_path)
            
            if not battery_ok:
                logger.error(f"Failed to download Battery Voltage data for {site_id}")
                failed += 1
                continue
            
            # Merge csv files
            logger.info("🔀 Merging data files...")
            merge_csv_files(level_path, baro_path, battery_path, site_id)
            
            # Clean up temporary files
            for temp_path in [baro_path, battery_path]:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    logger.info(f"  Cleaned up temporary file: {os.path.basename(temp_path)}")
            
            logger.info(f"✅ Successfully processed {display_name}")
            successful += 1
            
            # Pause between sites
            if i < len(sites):
                logger.info(f"⏸️ Pausing for {PAUSE_SECONDS} seconds...")
                time.sleep(PAUSE_SECONDS)
                
        except Exception as e:
            logger.error(f"Error processing site {site_id}: {e}")
            failed += 1
    
    # Summary
    logger.info("")
    logger.info("="*70)
    logger.info("SCRAPING SUMMARY")
    logger.info("="*70)
    logger.info(f"Total sites: {len(sites)}")
    logger.info(f"✅ Successful: {successful}")
    logger.info(f"❌ Failed: {failed}")
    logger.info("="*70)
    
    # Close session
    scraper.session.close()
    logger.info("HTTP session closed")
    logger.info("✨ HTTP Logger Scraper Finished")


if __name__ == "__main__":
    import sys
    
    # Get download path from command line or use default
    if len(sys.argv) > 1:
        download_path = sys.argv[1]
    else:
        download_path = "./data_downloads"
    
    run(download_path)