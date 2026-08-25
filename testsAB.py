"""
AquaTroll Depth Data Pipeline
A/B tests for the data scraping, validation & processing pipeline.

A monthly verification of script performance through testing of:
- Data connectivity and authentication
- Validation of returned data structures
- Battery health checks
- Verification of depth calibration calculations
- Hardware availability checks
- Flagging of any statistical anomalies
"""
# ============================================================================
# IMPORTS
# ============================================================================
import json
import logging
import os
import sys
import unittest
import random
from datetime import datetime
from getpass import getuser
from socket import gethostname

# Third Party Libraries
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# Suppress warnings during test execution
import warnings
warnings.filterwarnings('ignore')

# Add project directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ============================================================================
# GLOBAL CONFIGURATION
# ============================================================================
# ID the system and user running this script
COMPUTER_NAME = gethostname()
USERNAME = getuser()
# ============================================================================
# TEST DESCRIPTIONS
# ============================================================================
TEST_DESCRIPTIONS = {
    "Weather Website Access": (
        "Verifies if weather website accessible. "
        "Prints the received HTTP response code and time to execute."
    ),
    "Logger Portal Access": (
        "Checks connectivity & provided authentication credentials allow for a "
        "successful login to the AquaTroll website."
    ),
    "Logger Portal Navigation": (
        "Verifies ability to navigate and identify which display structure is in use (node or tables)"
    ),
    "Logger Portal Data Extraction": (
        "Confirms the presence and functionality of data download button."
    ),
    "Weather Data Structure": (
        "Verifies weather website accessibility and confirms the presence of data "
        "at the expected positions of the returned array (9 am at position 15, 3 pm at position 21) and rainfall "
        "data at position 4."
    ),
    "Logger Data Structure": (
        "Confirms downloaded data has the required columns for"
        "analysis (Date, Time, Level and Battery.)"
    ),
    "Network Power": (
        "Verifies battery voltage across all {n_sites} active sites."
    ),
    "Site Configuration": (
        "Checks which sites are setup by reading from the SITES_CONFIG env variable."
    ),
    "Network Availability": (
        "Locates last recorded depth reading for active site(s). "
        "Helpful for determining whether a pool has gone dry or if the equipment is faulty."
    ),
    "Depth Adjustment Verification": (
        "Independently tests barometric pressure adjustments using downloaded files "
        "then cross-checks these results against the pipeline's monthly output. "
        "Applies thresholds: depth > 0.3m and baro difference > 5 hPa to prevent "
        "corrections from very shallow/dry pools and sensor noise."
    ),
    "Monthly Statistical Anomalies": (
        "Picks up data for review with any 1 day depth changes >15%,  "
        "pressure changes >2% plus any other statistically dubious depth values"
    )
}
# ============================================================================
# HTML REPORT TEMPLATES
# ============================================================================
HTML_HEADER = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Pipeline Test Report - {timestamp}</title>
    <style>
        @page {{ margin: 10mm 12mm; }}
        body {{ font-family: Arial, sans-serif; font-size: 11px; margin: 0;
                background-color: #cfd6d1; color: #1c1917; }}
        .banner {{ width: 100%; height: 220px; object-fit: cover; border-radius: 6px 6px 0 0;
                   margin-bottom: 0; background-color: #b5bfba; display: block; }}
        .summary {{ background: white; padding: 0 0 16px 0; border-radius: 0 0 6px 6px;
                    margin-bottom: 12px; overflow: hidden; }}
        .summary-title-block {{ background-color: #d5cbae; padding: 12px 16px 10px 16px;
                                margin-bottom: 12px; }}
        .summary-title-block h1 {{ margin: 0 0 3px 0; font-size: 15px; font-weight: 600;
                                   letter-spacing: 0.01em; color: #1c1917; }}
        .summary-title-block p  {{ margin: 0; font-size: 9px; color: #555; }}
        .section-heading {{ background: #cfd6d1; border-radius: 6px;
                            margin-bottom: 8px; overflow: hidden; }}
        .section-heading h2 {{ font-size: 13px; font-weight: 600; color: #1c1917;
                               background-color: #cfd6d1; padding: 8px 16px;
                               margin: 0; border-radius: 0; text-align: center;
                               width: 100%; display: block; }}
        .test-section {{ background: white; padding: 12px 16px; margin-bottom: 10px;
                         border-radius: 6px; }}
        .test-section h3 {{ font-size: 12px; margin: 0 0 6px 0;
                            color: #1c1917; font-weight: 600; }}
        .test-description {{ color: #6c757d; font-style: italic; margin: 6px 0;
                             padding: 6px 10px; font-size: 10px;
                             background-color: #f0f2f1; border-left: 3px solid #bd3c35; }}
        h4 {{ font-size: 11px; margin: 8px 0 4px 0; }}
        .pass {{ color: #28a745; font-weight: bold; }}
        .fail {{ color: #dc3545; font-weight: bold; }}
        .warning {{ color: #b8860b; font-weight: bold; }}
        .skip {{ color: #6c757d; font-style: italic; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 6px;
                 font-size: 10px; }}
        th {{ background-color: #f0f2f1; padding: 6px 8px; text-align: left;
              border-bottom: 2px solid #d5cbae; font-size: 10px; }}
        td {{ padding: 5px 8px; border-bottom: 1px solid #dee2e6; }}
        .column-list {{ background: #f0f2f1; padding: 8px; border-radius: 4px;
                        margin: 6px 0; font-size: 10px; }}
        .calculation-table {{ margin-top: 8px; }}
        .anomaly {{ background-color: #fff3cd; }}
        .footer {{ text-align: center; color: #6c757d; margin-top: 20px;
                   padding: 12px; font-size: 10px; }}
        pre {{ background: #f0f2f1; padding: 8px; border-radius: 4px;
               overflow-x: auto; font-size: 9px; }}
        p {{ margin: 4px 0; font-size: 11px; }}
        ul {{ margin: 4px 0; padding-left: 18px; font-size: 11px; }}
        li {{ margin: 2px 0; }}
        /* Stoplight summary */
        .stoplight-intro {{ text-align: center; color: #555; font-size: 11px;
                            margin: 8px 16px 16px 16px; }}
        .stoplight {{ display: flex; justify-content: center; gap: 16px;
                      margin: 0 16px 4px 16px; }}
        .stoplight-cell {{ flex: 1; max-width: 120px; border-radius: 6px;
                           padding: 14px 8px 10px 8px; text-align: center; }}
        .stoplight-num   {{ font-size: 32px; font-weight: 700;
                            line-height: 1; display: block; }}
        .stoplight-label {{ font-size: 9px; font-weight: 600;
                            letter-spacing: 0.08em; text-transform: uppercase;
                            display: block; margin-top: 4px; }}
        .sl-pass    {{ background-color: #4a8c3f; }}
        .sl-pass .stoplight-num, .sl-pass .stoplight-label {{ color: white; }}
        .sl-warning {{ background-color: #c8960c; }}
        .sl-warning .stoplight-num, .sl-warning .stoplight-label {{ color: white; }}
        .sl-fail    {{ background-color: #bd3c35; }}
        .sl-fail .stoplight-num, .sl-fail .stoplight-label {{ color: white; }}
        @media print {{
            body {{ font-size: 8px; }}
            p, ul, li {{ font-size: 8px; }}
            table, th, td {{ font-size: 7.5px; }}
            .test-section h3 {{ font-size: 9px; }}
            .section-heading h2 {{ font-size: 10px; }}
            .summary-title-block h1 {{ font-size: 11px; }}
            .summary-title-block p {{ font-size: 7px; }}
            h4 {{ font-size: 8px; }}
            .test-description {{ font-size: 7.5px; }}
            .column-list {{ font-size: 7.5px; }}
            .stoplight-intro {{ font-size: 8px; }}
            .stoplight-num {{ font-size: 24px; }}
            .footer {{ font-size: 7.5px; }}
            pre {{ font-size: 7px; }}
        }}
    </style>
</head>
<body>
    <img src="{banner_path}" alt="Gorge Monitoring Banner" class="banner"
         onerror="this.style.display='none'">
"""
HTML_FOOTER = """
    <div class="footer">
        <p>End of Report: AquaTroll Depth Data Pipeline</p>
    </div>
</body>
</html>
"""
# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def load_csv_safely(filepath, logger=None):
    """
    Safely load a CSV file with error handling.
    Args:
        filepath (str): Path to CSV file
        logger: Optional logger for error reporting
    Returns:
        pd.DataFrame or None: Loaded dataframe or None on error
    """
    try:
        if not os.path.exists(filepath):
            if logger:
                logger.warning(f"File not found: {filepath}")
            return None
        return pd.read_csv(filepath)
    except Exception as e:
        if logger:
            logger.error(f"Error loading {filepath}: {str(e)}")
        return None
# ============================================================================
# MAIN TEST CLASS
# ============================================================================
class TestEnvironmentalPipeline(unittest.TestCase):
    """
    Suite for aquatroll data pipeline. Described in TEST_DESCRIPTIONS
    """
    @classmethod
    def setUpClass(cls):
        """
        Initialise test environment.
        Loads config & sets up paths.
        """
        load_dotenv()
        cls.results = []
        cls.test_start_time = datetime.now()
        # Get script directory to make paths absolute
        cls.script_dir = os.path.dirname(os.path.abspath(__file__))
        # Session storage for multi-step tests
        cls.auth_session = None
        cls.channel_response = None
        # Suppress logging during tests
        logging.getLogger().setLevel(logging.CRITICAL)
        # Load configurations from env
        cls.weather_url = os.getenv("WEATHER_URL", "")
        cls.login_url = os.getenv("LOGIN_URL", "")
        # Load logger sites
        cls.site_config = json.loads(os.getenv("SITES_CONFIG", "{}"))
        # Define data paths — now absolute relative to script location
        cls.data_downloads_path = os.path.join(cls.script_dir, 'data_downloads')
        cls.transformed_data_path = os.path.join(cls.script_dir, 'transformed_data')
        cls.validated_output_file = os.path.join(
            cls.transformed_data_path,
            'validatedDepthData.csv'
        )
    def record_result(self, test_name, category, status, details="", data=None):
        """
        Record test results for HTML report generation.
        Args:
            test_name (str): Name of the test
            category (str): Test category (Connectivity, Calculations, etc.)
            status (str): Test status (PASS, FAIL, WARNING, SKIP)
            details (str): Detailed description of test results
            data (dict): Additional data for HTML rendering
        """
        self.results.append({
            'test_name': test_name,
            'category': category,
            'status': status,
            'details': details,
            'data': data,
            'timestamp': datetime.now()
        })

    def _fail(self, test_name, category, message, data=None):
        """Record a FAIL result and raise the same message via self.fail()."""
        self.record_result(test_name, category, "FAIL", message, data)
        self.fail(message)

    def _skip(self, test_name, category, message):
        """Record a SKIP result and raise the same message via self.skipTest()."""
        self.record_result(test_name, category, "SKIP", message)
        self.skipTest(message)
   
    # ========================================================================
    # CONNECTIVITY TESTS
    # ========================================================================
    def test_01_weather_website_accessibility(self):
        """Test connectivity to weather website."""
        if not self.weather_url:
            self._fail("Weather Website Access", "Data Pipeline Tests",
                        "WEATHER_URL not configured in environment")
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        try:
            start_time = datetime.now()
            response = requests.get(self.weather_url, headers=headers, timeout=10)
            response_time = (datetime.now() - start_time).total_seconds()
            # Parse response to verify table structure exists
            soup = BeautifulSoup(response.text, 'html.parser')
            table = soup.select_one('table.data')
            if response.status_code == 200:
                if table:
                    self.record_result(
                        "Weather Website Access",
                        "Data Pipeline Tests",
                        "PASS",
                        f"Successfully connected (HTTP {response.status_code}, {response_time:.2f}s)"
                    )
                else:
                    self.record_result(
                        "Weather Website Access",
                        "Data Pipeline Tests",
                        "WARNING",
                        f"Connected but table.data not found (HTTP {response.status_code})"
                    )
            else:
                self._fail("Weather Website Access", "Data Pipeline Tests",
                            f"Weather website returned HTTP {response.status_code}")
        except (AssertionError, unittest.SkipTest):
            raise
        except requests.exceptions.Timeout:
            self._fail("Weather Website Access", "Data Pipeline Tests",
                        "Connection timeout after 10 seconds")
        except requests.exceptions.RequestException as e:
            self._fail("Weather Website Access", "Data Pipeline Tests", f"Connection error: {e}")
        except Exception as e:
            self._fail("Weather Website Access", "Data Pipeline Tests", f"Unexpected error: {e}")
    def test_02_logger_portal_authentication(self):
        """Test authentication to the logger portal using HTTP."""
        if not self.login_url:
            self._skip("Logger Portal Access", "Data Pipeline Tests",
                        "LOGIN_URL not configured in environment")
        login_username = os.getenv("LOGIN_USERNAME", "")
        login_password = os.getenv("LOGIN_PASSWORD", "")
        if not login_username or not login_password:
            self._fail("Logger Portal Access", "Data Pipeline Tests",
                        "LOGIN_USERNAME or LOGIN_PASSWORD not configured")
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        try:
            # Create session
            session = requests.Session()
            # Get login page
            response = session.get(self.login_url, headers=headers, timeout=10)
            if response.status_code != 200:
                self.__class__.auth_session = None
                self._fail("Logger Portal Access", "Data Pipeline Tests",
                            f"Login page not accessible (HTTP {response.status_code})")
            # Parse login form
            soup = BeautifulSoup(response.text, 'html.parser')
            form_data = {}
            # Get hidden fields (ASP.NET ViewState)
            for hidden_input in soup.find_all('input', type='hidden'):
                name = hidden_input.get('name')
                value = hidden_input.get('value', '')
                if name:
                    form_data[name] = value
            # Add credentials
            form_data['txtUsername'] = login_username
            form_data['txtPassword'] = login_password
            form_data['btnLogon'] = 'Logon'
            # Submit login
            login_response = session.post(
                self.login_url,
                data=form_data,
                headers=headers,
                timeout=15,
                allow_redirects=True
            )
            # Check if redirected away from login page
            if 'logon.aspx' in login_response.url.lower():
                self.__class__.auth_session = None
                self._fail("Logger Portal Access", "Data Pipeline Tests",
                            "Authentication failed - credentials rejected (remained on login page)")
            # Success. Store session for other tests (use class variable!)
            self.__class__.auth_session = session
            self.record_result(
                "Logger Portal Access",
                "Data Pipeline Tests",
                "PASS",
                f"Successfully authenticated with logger portal",
                {
                    'session_cookie': bool(session.cookies),
                    'final_url': login_response.url
                }
            )
        except (AssertionError, unittest.SkipTest):
            raise
        except requests.exceptions.Timeout:
            self.__class__.auth_session = None
            self._fail("Logger Portal Access", "Data Pipeline Tests", "Connection timeout")
        except requests.exceptions.RequestException as e:
            self.__class__.auth_session = None
            self._fail("Logger Portal Access", "Data Pipeline Tests", f"Connection error: {e}")
        except Exception as e:
            self.__class__.auth_session = None
            self._fail("Logger Portal Access", "Data Pipeline Tests", f"Authentication error: {e}")
    def test_03_logger_portal_node_selection(self):
        """Test ability to navigate to and select nodes/tables in the logger portal."""
        if not hasattr(self.__class__, 'auth_session') or self.__class__.auth_session is None:
            self._skip("Logger Portal Navigation", "Data Pipeline Tests",
                        "Authentication session not available")
        # Get first enabled site
        enabled_sites = {name: config for name, config in self.site_config.items()
                         if config.get('enabled', False)}
        if not enabled_sites:
            self._skip("Logger Portal Navigation", "Data Pipeline Tests",
                        "No enabled sites configured")
        # Pick a random enabled site
        test_site_name = random.choice(list(enabled_sites.keys()))
        test_site_config = enabled_sites[test_site_name]
        nav_option = test_site_config.get('nav_option')
        if not nav_option:
            self.__class__.channel_response = None
            self._fail("Logger Portal Navigation", "Data Pipeline Tests",
                        f"Site {test_site_name} missing nav_option in configuration")
        base_url = os.getenv("BASE_URL", "").rstrip('/')
        channel_url = f'{base_url}/data-channels.aspx?id={nav_option}'
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        try:
            # Navigate to channel page
            response = self.__class__.auth_session.get(channel_url, headers=headers, timeout=15)
            if response.status_code != 200:
                self.__class__.channel_response = None
                self._fail("Logger Portal Navigation", "Data Pipeline Tests",
                            f"Failed to navigate to channel (HTTP {response.status_code})")
            # Parse page to verify channel elements
            soup = BeautifulSoup(response.text, 'html.parser')
            # Check on a valid channel page
            page_text = response.text.lower()
            has_node_or_tree = ('node' in page_text or 'tree' in page_text)
            has_data_table = soup.find('table') is not None
            validation_score = sum([has_node_or_tree, has_data_table])
            # Accept if both elements present (page loaded with proper structure)
            if validation_score >= 2:
                # Store response for download test
                self.__class__.channel_response = response
                self.record_result(
                    "Logger Portal Navigation",
                    "Data Pipeline Tests",
                    "PASS",
                    f"Successfully navigated to channel {nav_option} for site {test_site_name} (validation score: {validation_score}/2)",
                    {
                        'site': test_site_name,
                        'nav_option': nav_option,
                        'node_or_tree_present': has_node_or_tree,
                        'data_table': has_data_table
                    }
                )
            else:
                self.__class__.channel_response = None
                self._fail("Logger Portal Navigation", "Data Pipeline Tests",
                            f"Channel page missing expected elements (score: {validation_score}/2)")
        except requests.exceptions.Timeout:
            self.__class__.channel_response = None
            self._fail("Logger Portal Navigation", "Data Pipeline Tests", "Navigation timeout")
        except requests.exceptions.RequestException as e:
            self.__class__.channel_response = None
            self._fail("Logger Portal Navigation", "Data Pipeline Tests", f"Navigation error: {e}")
    def test_04_logger_portal_csv_download(self):
        """Test presence of CSV download functionality."""
        if not hasattr(self.__class__, 'channel_response') or self.__class__.channel_response is None:
            self._skip("Logger Portal Data Extraction", "Data Pipeline Tests",
                        "No channel navigation response available")
        try:
            soup = BeautifulSoup(self.__class__.channel_response.text, 'html.parser')
            page_text = self.__class__.channel_response.text.lower()
            # Check for table/channel selection mechanism
            has_channel_selector = (
                soup.find('select') is not None or # Any dropdown
                soup.find('form') is not None or # Any form
                'table' in page_text # Reference to tables
            )
            # Check for CSV/export button
            has_csv_button = (
                'csv' in page_text or
                soup.find('input', attrs={'value': lambda x: x and 'csv' in str(x).lower()}) is not None or
                soup.find('button', attrs={'value': lambda x: x and 'csv' in str(x).lower()}) is not None or
                soup.find('a', attrs={'href': lambda x: x and 'csv' in str(x).lower()}) is not None or
                soup.find('input', attrs={'value': lambda x: x and 'export' in str(x).lower()}) is not None
            )
            # Both must be present
            if has_channel_selector and has_csv_button:
                self.record_result(
                    "Logger Portal Data Extraction",
                    "Data Pipeline Tests",
                    "PASS",
                    f"CSV download capability verified: Table selector present and CSV/export button found"
                )
            elif has_csv_button and not has_channel_selector:
                self.record_result(
                    "Logger Portal Data Extraction",
                    "Data Pipeline Tests",
                    "WARNING",
                    f"CSV button found but no table selection mechanism detected"
                )
            elif has_channel_selector and not has_csv_button:
                self.record_result(
                    "Logger Portal Data Extraction",
                    "Data Pipeline Tests",
                    "WARNING",
                    f"Table selection present but CSV/export button not found"
                )
            else:
                self._fail("Logger Portal Data Extraction", "Data Pipeline Tests",
                            "Neither table selection nor CSV button found on channel page")
        except (AssertionError, unittest.SkipTest):
            raise
        except Exception as e:
            self._fail("Logger Portal Data Extraction", "Data Pipeline Tests",
                        f"Error checking download functionality: {e}")
   
    # ========================================================================
    # DATA STRUCTURE TESTS
    # ========================================================================
    def test_05_weather_data_structure(self):
        """Test weather website scraping capability and data structure."""
        if not self.weather_url:
            self._skip("Weather Data Structure", "Data Pipeline Tests",
                        "WEATHER_URL not configured - skipping test")
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        try:
            response = requests.get(self.weather_url, headers=headers, timeout=10)
            if response.status_code != 200:
                self._fail("Weather Data Structure", "Data Pipeline Tests",
                            f"Could not access weather website (HTTP {response.status_code})")
            # Parse the page
            soup = BeautifulSoup(response.text, 'html.parser')
            table = soup.select_one('table.data')
            if not table:
                self._fail("Weather Data Structure", "Data Pipeline Tests",
                            "Table with class 'data' not found in HTML")
            # Get data rows from tbody
            data_rows = table.select('tbody tr')
            if len(data_rows) == 0:
                self._fail("Weather Data Structure", "Data Pipeline Tests",
                            "No data rows found in weather table")
            # Check first valid data row structure
            first_valid_row = None
            for row in data_rows:
                cells = row.select('th, td')
                if cells and len(cells) > 1:
                    date_cell = cells[0].get_text(strip=True)
                    if date_cell.isdigit(): # Valid day number
                        first_valid_row = cells
                        break
            if not first_valid_row:
                self._fail("Weather Data Structure", "Data Pipeline Tests",
                            "No valid data row found in weather table")
            # Validate structure
            total_cells = len(first_valid_row)
            position_4_valid = False
            position_15_valid = False
            position_21_valid = False
            position_4_value = "N/A"
            position_15_value = "N/A"
            position_21_value = "N/A"
            
            # Check position 4 (rainfall)
            if total_cells > 4:
                val_4 = first_valid_row[4].get_text(strip=True)
                try:
                    # Rainfall can be 0 or positive, empty is also valid (no rain)
                    if val_4 == '':
                        position_4_valid = True
                        position_4_value = "0.0"
                    else:
                        float_val = float(val_4)
                        position_4_valid = 0 <= float_val <= 500  # Reasonable rainfall range in mm
                        position_4_value = val_4
                except (ValueError, TypeError):
                    pass
            
            # Check position 15 (9am pressure)
            if total_cells > 15:
                val_15 = first_valid_row[15].get_text(strip=True)
                try:
                    float_val = float(val_15)
                    position_15_valid = 950 <= float_val <= 1050
                    position_15_value = val_15
                except (ValueError, TypeError):
                    pass
            # Check position 21 (3pm pressure)
            if total_cells > 21:
                val_21 = first_valid_row[21].get_text(strip=True)
                try:
                    float_val = float(val_21)
                    position_21_valid = 950 <= float_val <= 1050
                    position_21_value = val_21
                except (ValueError, TypeError):
                    pass
            # Need all three columns to be valid
            if position_4_valid and position_15_valid and position_21_valid:
                self.record_result(
                    "Weather Data Structure",
                    "Data Pipeline Tests",
                    "PASS",
                    f"Weather table structure verified: {total_cells} columns,\n rainfall at position 4 ({position_4_value} mm), pressure data at positions 15 ({position_15_value} hPa) and 21 ({position_21_value} hPa)"
                )
            else:
                issues = []
                if not position_4_valid:
                    issues.append(f"Position 4 (rainfall) invalid (got: '{position_4_value}')")
                if not position_15_valid:
                    issues.append(f"Position 15 invalid (got: '{position_15_value}')")
                if not position_21_valid:
                    issues.append(f"Position 21 invalid (got: '{position_21_value}')")
                self._fail("Weather Data Structure", "Data Pipeline Tests",
                            f"Weather table data validation failed: {', '.join(issues)}")
        except (AssertionError, unittest.SkipTest):
            raise
        except requests.exceptions.Timeout:
            self._fail("Weather Data Structure", "Data Pipeline Tests",
                        "Connection timeout accessing weather website")
        except requests.exceptions.RequestException as e:
            self._fail("Weather Data Structure", "Data Pipeline Tests", f"Connection error: {e}")
        except Exception as e:
            self._fail("Weather Data Structure", "Data Pipeline Tests",
                        f"Error parsing weather data: {e}")
    def test_06_logger_data_structure(self):
        """Test logger CSV data structure."""
        if not os.path.exists(self.data_downloads_path):
            self._fail("Logger Data Structure", "Data Pipeline Tests",
                        "data_downloads directory not found")
        csv_files = [f for f in os.listdir(self.data_downloads_path) if f.endswith('.csv') and f != 'weather_data.csv' and 'baro' not in f.lower()]
        if not csv_files:
            self._fail("Logger Data Structure", "Data Pipeline Tests",
                        "No logger CSV files found in data_downloads")
        test_file = os.path.join(self.data_downloads_path, csv_files[0])
        try:
            df = pd.read_csv(test_file)
            # Required columns from httpLoggerScraper.py output
            # Note: Level column varies by site config (metres vs feet)
            base_columns = ['Date', 'Time']
            level_column_options = [
                'Level(RAW)[Main Buffer] (ft)',
                'Level in metres (m)'
            ]
            battery_column_prefix = 'Main Battery(MIN)[Main Buffer]'
            # Check base columns
            base_present = all(col in df.columns for col in base_columns)
            # Check at least one level column variant exists
            level_present = any(col in df.columns for col in level_column_options)
            # Check battery column (hardware matched — units voltage varies by hardware type)
            battery_col = next(
                (col for col in df.columns if col.strip().startswith(battery_column_prefix)),
                None
            )
            battery_present = battery_col is not None
            if base_present and level_present:
                level_col = next((col for col in level_column_options if col in df.columns), None)
                batt_note = f", {battery_col}" if battery_present else " (battery column not found — WARNING)"
                overall_status = "PASS" if battery_present else "WARNING"
                self.record_result(
                    "Logger Data Structure",
                    "Data Pipeline Tests",
                    overall_status,
                    f"Logger data has the required columns: Date, Time, {level_col}{batt_note}"
                )
            else:
                missing = []
                if not base_present:
                    missing_base = [col for col in base_columns if col not in df.columns]
                    missing.extend(missing_base)
                if not level_present:
                    missing.append(f"Level column (expected one of: {', '.join(level_column_options)})")
                self._fail("Logger Data Structure", "Data Pipeline Tests",
                            f"Missing required columns: {', '.join(missing)}")
        except (AssertionError, unittest.SkipTest):
            raise
        except Exception as e:
            self._fail("Logger Data Structure", "Data Pipeline Tests",
                        f"Error reading logger data: {e}")
    def test_07_battery_voltage_check(self):
        """Check battery voltage for enabled sites.

        Thresholds (ERP3 / ERP4 use Starlink ~13V; all other sites use logger ~3.5V):
          WARNING : voltage < warn_threshold
          FAIL    : voltage < fail_threshold
          PASS    : voltage >= warn_threshold

        Status is based purely on the last returned voltage reading,
        however old it is. Time since last signal gets checked
        separately by test_10/Network Availability 

        The battery_results list carries a 'voltage_history' key (list of
        {date, voltage} dicts) for any site whose status is WARNING or FAIL,
        so the HTML renderer can draw a trend graph.
        """
        if not os.path.exists(self.data_downloads_path):
            self._fail("Network Power", "Hardware Tests", "data_downloads directory not found")

        enabled_sites = {
            name: config for name, config in self.site_config.items()
            if config.get('enabled', False)
        }
        if not enabled_sites:
            self._skip("Network Power", "Hardware Tests", "No enabled sites configured")

        # Per-site threshold config.
        # ERP3 / ERP4 are Starlink-powered; everything else is logger battery.
        STARLINK_SITES = {'ERP3', 'ERP4'}
        THRESHOLDS = {
            'starlink': {'warn': 13.2, 'fail': 13.05},
            'logger':   {'warn': 3.5,  'fail': 3.4},
        }
        STALE_WARN_DAYS = 7
        STALE_FAIL_DAYS = 28

        now = datetime.now()
        battery_results = []
        fail_count    = 0
        warning_count = 0

        for site_id, config in enabled_sites.items():
            csv_path = os.path.join(self.data_downloads_path, f"{site_id}.csv")

            # ── missing file ────────────────────────────────────────────────
            if not os.path.exists(csv_path):
                battery_results.append({
                    'site': site_id,
                    'status': 'FAIL',
                    'details': 'CSV file not found',
                    'threshold_warn': 'N/A',
                    'threshold_fail': 'N/A',
                })
                fail_count += 1
                continue

            try:
                df = pd.read_csv(csv_path)

                # ── locate battery column ───────────────────────────────────
                battery_col = next(
                    (col for col in df.columns
                     if col.strip().startswith('Main Battery(MIN)[Main Buffer]')),
                    None
                )

                site_type = 'starlink' if site_id in STARLINK_SITES else 'logger'
                thresh_warn = THRESHOLDS[site_type]['warn']
                thresh_fail = THRESHOLDS[site_type]['fail']

                if battery_col is None:
                    battery_results.append({
                        'site': site_id,
                        'status': 'WARNING',
                        'details': 'Battery voltage column not found',
                        'threshold_warn': thresh_warn,
                        'threshold_fail': thresh_fail,
                    })
                    warning_count += 1
                    continue

                # ── parse dates so we can check data recency ────────────────
                if 'Date' in df.columns and 'Time' in df.columns:
                    df['_datetime'] = pd.to_datetime(
                        df['Date'] + ' ' + df['Time'],
                        dayfirst=True,
                        errors='coerce'
                    )
                else:
                    df['_datetime'] = pd.NaT

                # ── build a clean series aligned on datetime ─────────────────
                battery_numeric = pd.to_numeric(df[battery_col], errors='coerce')
                valid_mask      = battery_numeric.notna()
                battery_valid   = battery_numeric[valid_mask]
                dates_valid     = df['_datetime'][valid_mask]

                # ── last datapoint ────────────────────────
                if df['_datetime'].notna().any():
                    latest_dt   = df['_datetime'].max()
                    days_since  = (now - latest_dt).days
                else:
                    days_since  = None   # unknown

                # ── no valid battery readings at all ────────────────────────
                if battery_valid.empty:
                    detail_msg = 'No valid battery data'
                    if days_since is not None:
                        detail_msg += f' (last datapoint {days_since}d ago)'
                    battery_results.append({
                        'site': site_id,
                        'status': 'WARNING',
                        'details': detail_msg,
                        'threshold_warn': thresh_warn,
                        'threshold_fail': thresh_fail,
                    })
                    warning_count += 1
                    continue

                # ── current voltage = most recent valid reading by datetime ──
                aligned = pd.DataFrame({
                    'datetime': dates_valid,
                    'voltage':  battery_valid
                }).dropna(subset=['datetime']).sort_values('datetime', ascending=True)

                if aligned.empty:
                    # Fall back to positional if no datetimes
                    current_voltage = round(float(battery_valid.iloc[-1]), 3)
                    last_reading_str = 'unknown'
                else:
                    current_voltage  = round(float(aligned['voltage'].iloc[-1]), 3)
                    latest_batt_dt   = aligned['datetime'].iloc[-1]
                    last_reading_str = latest_batt_dt.strftime('%d/%m/%Y %H:%M')

                # ── voltage value ─────────────────
                if current_voltage < thresh_fail:
                    effective_status = 'FAIL'
                elif current_voltage < thresh_warn:
                    effective_status = 'WARNING'
                else:
                    effective_status = 'PASS'

                if effective_status == 'FAIL':
                    fail_count += 1
                elif effective_status == 'WARNING':
                    warning_count += 1

                result = {
                    'site': site_id,
                    'status': effective_status,
                    'threshold_warn': thresh_warn,
                    'threshold_fail': thresh_fail,
                    'current_voltage': current_voltage,
                    'last_reading': last_reading_str,
                    'days_since': days_since,
                    'details': '',
                }

                # ── voltage history for trend graph (WARNING / FAIL only) ───
                if effective_status in ('WARNING', 'FAIL'):
                    history_df = pd.DataFrame({
                        'datetime': dates_valid,
                        'voltage': battery_valid
                    }).dropna().sort_values('datetime')
                    # Keep last 28 days
                    cutoff = now - pd.Timedelta(days=STALE_FAIL_DAYS)
                    history_df = history_df[history_df['datetime'] >= cutoff]
                    result['voltage_history'] = [
                        {
                            'date':    row['datetime'].strftime('%d/%m'),
                            'voltage': round(float(row['voltage']), 3)
                        }
                        for _, row in history_df.iterrows()
                    ]

                battery_results.append(result)

            except Exception as e:
                battery_results.append({
                    'site': site_id,
                    'status': 'FAIL',
                    'details': f"Error reading file: {str(e)}",
                    'threshold_warn': 'N/A',
                    'threshold_fail': 'N/A',
                })
                fail_count += 1

        # ── overall status ───────────────────────────────────────────────────
        if fail_count > 0:
            overall_status = "FAIL"
        elif warning_count > 0:
            overall_status = "WARNING"
        else:
            overall_status = "PASS"

        self.record_result(
            "Network Power",
            "Hardware Tests",
            overall_status,
            "",
            {'battery_results': battery_results}
        )
        if fail_count > 0:
            self.fail(
                f"{fail_count} site(s) failed battery check "
                "(low voltage, missing CSV, or stale data)"
            )
   
    # ========================================================================
    # CONFIGURATION TESTS
    # ========================================================================
    def test_08_site_configuration(self):
        """Test site configuration from environment."""
        try:
            enabled_sites = []
            disabled_sites = []
            # Parse enabled from SITES_CONFIG
            for site_name, config in self.site_config.items():
                is_enabled = config.get('enabled', False)
                site_display = site_name
                if is_enabled:
                    enabled_sites.append(site_display)
                else:
                    note = config.get('notes', '')
                    disabled_sites.append(f"{site_name}: {note}" if note else site_name)
            config_data = {
                'enabled': sorted(enabled_sites),
                'disabled': sorted(disabled_sites)
            }
            if len(enabled_sites) == 0:
                self._fail("Site Configuration", "Data Pipeline Tests",
                            "No enabled sites found in SITES_CONFIG", config_data)
            else:
                self.record_result(
                    "Site Configuration",
                    "Data Pipeline Tests",
                    "PASS",
                    f"Enabled: {len(enabled_sites)} sites; Disabled: {len(disabled_sites)} sites",
                    config_data
                )
        except (AssertionError, unittest.SkipTest):
            raise
        except Exception as e:
            self._fail("Site Configuration", "Data Pipeline Tests", f"Configuration error: {e}")
   
    # ========================================================================
    # CALCULATION TESTS
    # ========================================================================
    def test_09_depth_calculation_verification(self):
        """Validates result accuracy of depth calculations.

        Imports calculate_adjusted_depth() directly from pressureAdjustment so
        the test always exercises the same code path as the pipeline — not a
        local copy that could drift out of sync.

        Randomly samples 3 rows from validatedDepthData.csv, feeds each row as a
        single-row DataFrame to calculate_adjusted_depth(), then cross-checks
        the returned Depth(m)adjusted value against what the pipeline already
        recorded, within a 2 cm tolerance. The reason the pipeline did or did
        not apply an adjustment is captured from the OTHER - Comments - Text
        column and shown in the report.

        Skipped when PRESSURE_ADJUSTMENT is disabled, since Depth(m)adjusted
        is never written to the output in that case.
        """
        if not os.path.exists(self.validated_output_file):
            self._skip("Depth Adjustment Verification", "Data Pipeline Tests",
                        "No validated data found — pipeline has not run")

        # These SKIP checks live outside the try/except below deliberately:
        # skipTest() raises unittest.SkipTest, which is itself an Exception
        # subclass and would otherwise be caught by "except Exception" and
        # misreported as FAIL.
        validated_df = load_csv_safely(self.validated_output_file)
        if validated_df is None:
            self._skip("Depth Adjustment Verification", "Data Pipeline Tests",
                        "Could not load validated output file")

        if 'Depth(m)adjusted' not in validated_df.columns:
            self._skip("Depth Adjustment Verification", "Data Pipeline Tests",
                        "No weather station barometric pressure adjustments were applied. "
                        "To enable this feature set the PRESSURE_ADJUSTMENT variable to true in your .env file")

        # Filter for rows that have both raw and adjusted depth values
        valid_mask = (
            validated_df['Depth(m)raw'].notna() &
            validated_df['Depth(m)adjusted'].notna() &
            (pd.to_numeric(validated_df['Depth(m)raw'], errors='coerce') > 0) &
            (pd.to_numeric(validated_df['Depth(m)adjusted'], errors='coerce') > 0)
        )
        filtered_df = validated_df[valid_mask].copy()

        if len(filtered_df) == 0:
            self._skip("Depth Adjustment Verification", "Data Pipeline Tests",
                        "No valid depth data in validated output")

        try:
            # Lazy import — isolated to this test so a broken
            # pressureAdjustment module does not crash the entire test
            # runner at startup.
            from pressureAdjustment import calculate_adjusted_depth

            num_samples = min(3, len(filtered_df))
            test_samples = filtered_df.sample(n=num_samples, random_state=42)

            calculation_results = []
            all_passed = True
            tolerance = 0.02  # 2 cm

            for idx, row in test_samples.iterrows():
                site     = row['Sample Point']
                date     = row['Date Time (dd/mm/yyyy hh24:mi:ss)']
                expected_adjusted = float(row['Depth(m)adjusted'])

                # Build a single-row DataFrame matching the schema that
                # calculate_adjusted_depth() expects.
                row_df = pd.DataFrame([{
                    'Depth(m)raw':                                  pd.to_numeric(row['Depth(m)raw'], errors='coerce'),
                    'BomBaro':                                       pd.to_numeric(row.get('BomBaro'), errors='coerce'),
                    'Barometric Pressure(RAW)[Main Buffer] (hPa)':  pd.to_numeric(row.get('Barometric Pressure(RAW)[Main Buffer] (hPa)'), errors='coerce'),
                    'Pressure(RAW)[Main Buffer] (PSI)':             pd.to_numeric(row.get('Pressure(RAW)[Main Buffer] (PSI)'), errors='coerce'),
                    'OTHER - Comments - Text':                       '',
                }])

                result_df           = calculate_adjusted_depth(row_df)
                calculated_adjusted = float(result_df['Depth(m)adjusted'].iloc[0])
                reason              = result_df['OTHER - Comments - Text'].iloc[0] or ''

                # If calculate_adjusted_depth returned NaN (e.g. no weather data),
                # fall back to raw depth — matching what the pipeline itself does.
                if pd.isna(calculated_adjusted):
                    calculated_adjusted = float(row_df['Depth(m)raw'].iloc[0])

                adjustment_applied = (
                    abs(calculated_adjusted - float(row_df['Depth(m)raw'].iloc[0])) > 0.001
                )

                bom_val    = row_df['BomBaro'].iloc[0]
                logger_val = row_df['Barometric Pressure(RAW)[Main Buffer] (hPa)'].iloc[0]

                passed = abs(round(calculated_adjusted, 2) - round(expected_adjusted, 2)) < tolerance
                if not passed:
                    all_passed = False

                calculation_results.append({
                    'test_num':           len(calculation_results) + 1,
                    'site':               site,
                    'date':               date,
                    'depth_raw':          round(float(row_df['Depth(m)raw'].iloc[0]), 2),
                    'bom_baro':           round(float(bom_val), 1) if pd.notna(bom_val) else 'N/A',
                    'logger_baro':        round(float(logger_val), 1) if pd.notna(logger_val) else 'N/A',
                    'adjustment_applied': adjustment_applied,
                    'calculated':         round(calculated_adjusted, 2),
                    'expected':           round(expected_adjusted, 2),
                    'difference':         round(abs(calculated_adjusted - expected_adjusted), 3),
                    'passed':             passed,
                    'reason':             reason,
                })

            self.record_result(
                "Depth Adjustment Verification",
                "Data Pipeline Tests",
                "PASS" if all_passed else "FAIL",
                f"Tested {len(calculation_results)} calculations against pressureAdjustment.calculate_adjusted_depth()",
                {'calculations': calculation_results}
            )
            if not all_passed:
                self.fail("One or more calculated depths did not match expected values within tolerance")

        except (AssertionError, unittest.SkipTest):
            raise
        except ImportError:
            self._fail("Depth Adjustment Verification", "Data Pipeline Tests",
                        "Could not import calculate_adjusted_depth from pressureAdjustment.py — ensure it is on the Python path")
        except Exception as e:
            self._fail("Depth Adjustment Verification", "Data Pipeline Tests",
                        f"Error during calculation verification: {e}")
   
    # ========================================================================
    # DATA QUALITY TESTS
    # ========================================================================
 
    def test_10_data_recency_check(self):
        """Check the most recent depth data reading for each enabled site.

        Recency tiers:
          PASS    : data present and within 7 days
          WARNING : data present but 8–28 days old
          FAIL    : no CSV / empty CSV / no valid data / data older than 28 days
        """
        STALE_WARN_DAYS = 7
        STALE_FAIL_DAYS = 28

        if not os.path.exists(self.data_downloads_path):
            self._fail("Network Availability", "Hardware Tests", "data_downloads directory not found")

        enabled_sites = {
            name: config for name, config in self.site_config.items()
            if config.get('enabled', False)
        }

        if not enabled_sites:
            self._skip("Network Availability", "Hardware Tests", "No enabled sites configured")

        recency_results = []
        current_date  = datetime.now()
        fail_count    = 0
        warning_count = 0
        pass_count    = 0

        for site_id, config in enabled_sites.items():
            csv_path = os.path.join(self.data_downloads_path, f"{site_id}.csv")

            # ── missing file ────────────────────────────────────────────────
            if not os.path.exists(csv_path):
                recency_results.append({
                    'site': site_id, 'date': 'N/A', 'value': 'N/A',
                    'unit': 'N/A', 'days_since': 'No CSV file', 'status': 'FAIL'
                })
                fail_count += 1
                continue

            try:
                df = pd.read_csv(csv_path)

                if df.empty:
                    recency_results.append({
                        'site': site_id, 'date': 'N/A', 'value': 'N/A',
                        'unit': 'N/A', 'days_since': 'Empty CSV', 'status': 'FAIL'
                    })
                    fail_count += 1
                    continue

                # ── detect level column and unit ────────────────────────────
                level_column = None
                unit = None
                if 'Level in metres (m)' in df.columns:
                    level_column = 'Level in metres (m)'
                    unit = 'm'
                elif 'Level(RAW)[Main Buffer] (ft)' in df.columns:
                    level_column = 'Level(RAW)[Main Buffer] (ft)'
                    unit = 'ft'
                else:
                    for col in df.columns:
                        if 'Level' in col:
                            level_column = col
                            unit = 'm' if '(m)' in col else ('ft' if '(ft)' in col else 'unknown')
                            break

                if level_column is None:
                    recency_results.append({
                        'site': site_id, 'date': 'N/A', 'value': 'N/A',
                        'unit': 'N/A', 'days_since': 'No level column', 'status': 'FAIL'
                    })
                    fail_count += 1
                    continue

                if 'Date' not in df.columns:
                    recency_results.append({
                        'site': site_id, 'date': 'N/A', 'value': 'N/A',
                        'unit': unit, 'days_since': 'No date column', 'status': 'FAIL'
                    })
                    fail_count += 1
                    continue

                # ── parse and filter ────────────────────────────────────────
                df['parsed_date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
                df[level_column]  = pd.to_numeric(df[level_column], errors='coerce')
                valid_data = df[df['parsed_date'].notna() & df[level_column].notna()]

                if valid_data.empty:
                    recency_results.append({
                        'site': site_id, 'date': 'N/A', 'value': 'N/A',
                        'unit': unit, 'days_since': 'No valid data rows', 'status': 'FAIL'
                    })
                    fail_count += 1
                    continue

                # ── most recent reading ─────────────────────────────────────
                most_recent_idx   = valid_data['parsed_date'].idxmax()
                most_recent_date  = valid_data.loc[most_recent_idx, 'parsed_date']
                most_recent_value = valid_data.loc[most_recent_idx, level_column]
                days_since        = (current_date - most_recent_date).days
                date_str          = most_recent_date.strftime('%d/%m/%Y')

                # ── recency status ──────────────────────────────────────────
                if days_since > STALE_FAIL_DAYS:
                    status = 'FAIL'
                    fail_count += 1
                elif days_since > STALE_WARN_DAYS:
                    status = 'WARNING'
                    warning_count += 1
                else:
                    status = 'PASS'
                    pass_count += 1

                recency_results.append({
                    'site':       site_id,
                    'date':       date_str,
                    'value':      round(most_recent_value, 2),
                    'unit':       unit,
                    'days_since': f"{days_since} days ago",
                    'status':     status
                })

            except Exception as e:
                recency_results.append({
                    'site': site_id, 'date': 'N/A', 'value': 'N/A',
                    'unit': 'N/A', 'days_since': f'Error: {str(e)}', 'status': 'FAIL'
                })
                fail_count += 1

        # ── overall status ───────────────────────────────────────────────────
        if fail_count > 0:
            overall_status = "FAIL"
        elif warning_count > 0:
            overall_status = "WARNING"
        else:
            overall_status = "PASS"

        details = (
            f"Checked {len(enabled_sites)} active sites: "
            f"{warning_count} sporadically uploading data (>7 days), "
            f"{fail_count} failures (missing or no data in 28 days)"
        )

        self.record_result(
            "Network Availability",
            "Hardware Tests",
            overall_status,
            details,
            {'recency_results': recency_results}
        )

        if fail_count > 0:
            self.fail(
                f"{fail_count} site(s) failed network availability check"
            )
   
    def test_11_statistical_anomaly_detection(self):
        """Test for statistical anomalies in validated data.

        Depth % change suppression:
          - Percentage change is unstable when previous value is at or near
            zero. Rows where the previous depth reading <= MIN_DEPTH_FOR_PCT
            (0.05 m) are excluded from the % change check to prevent false flags.

        Depth validity:
          - Depth(m)adjusted > 3 m → WARNING  ('outside known historical depth range, verify sensor')
          - Depth(m)adjusted > 5 m → FAIL     ('data invalid, field calibration required')
          These are appended to depth_anomalies with a 'validity' key so
          the HTML renderer can distinguish them from % change anomalies.
        """
        MIN_DEPTH_FOR_PCT = 0.05   # metres — below this, % change is meaningless
        DEPTH_WARN_M      = 3.0    # Deepest recorded historical range 
        DEPTH_FAIL_M      = 5.0    # The pressure sensor is cooked & nNeeds a field calibration

        if not os.path.exists(self.validated_output_file):
            self._skip("Monthly Statistical Anomalies", "Hardware Tests",
                        "Statistical analysis requires validated data files")
        try:
            df = load_csv_safely(self.validated_output_file)
            if df is None:
                self.skipTest("Could not load validated output file")
                return

            # Depth(m)adjusted only exists when PRESSURE_ADJUSTMENT=true; BomBaro
            # and Rainfall are always present since the BoM merge are still needed.
            depth_adjustment_available = 'Depth(m)adjusted' in df.columns
            if depth_adjustment_available:
                df['Depth(m)adjusted'] = pd.to_numeric(df['Depth(m)adjusted'], errors='coerce')
            df['BomBaro']          = pd.to_numeric(df['BomBaro'], errors='coerce')
            df['Rainfall']         = pd.to_numeric(df.get('Rainfall', pd.Series()), errors='coerce')
            df = df.dropna(subset=['Sample Point'])
            df['parsed_date'] = pd.to_datetime(
                df['Date Time (dd/mm/yyyy hh24:mi:ss)'], errors='coerce'
            )

            depth_anomalies   = []
            pressure_anomalies = []
            has_fail          = False

            for site in df['Sample Point'].unique():
                site_data = df[df['Sample Point'] == site].copy()
                site_data = site_data.sort_values('Date Time (dd/mm/yyyy hh24:mi:ss)')

                if 'Depth(m)adjusted' in site_data.columns:

                    # ── % change anomalies (suppress near-zero previous values) ──
                    prev_depth = site_data['Depth(m)adjusted'].shift(1)
                    valid_pct_mask = prev_depth.abs() > MIN_DEPTH_FOR_PCT
                    site_data['depth_pct_change'] = np.where(
                        valid_pct_mask,
                        site_data['Depth(m)adjusted'].pct_change() * 100,
                        np.nan
                    )
                    depth_mask = abs(site_data['depth_pct_change']) > 15

                    for idx in site_data[depth_mask].index:
                        rainfall_val = site_data.loc[idx, 'Rainfall'] if 'Rainfall' in site_data.columns else None
                        rain_event   = rainfall_val > 0 if pd.notna(rainfall_val) else False
                        prev_val     = prev_depth.loc[idx]
                        depth_anomalies.append({
                            'site':       site,
                            'date':       str(site_data.loc[idx, 'Date Time (dd/mm/yyyy hh24:mi:ss)']),
                            'prev_value': round(float(prev_val), 3) if pd.notna(prev_val) else 'N/A',
                            'value':      round(site_data.loc[idx, 'Depth(m)adjusted'], 3),
                            'change_pct': round(site_data.loc[idx, 'depth_pct_change'], 1),
                            'rain_event': rain_event,
                            'validity':   'pct_change'
                        })

                    # ── depth validity check ────────────────────────────────
                    for idx, row in site_data.iterrows():
                        d = row['Depth(m)adjusted']
                        if pd.isna(d):
                            continue
                        if d > DEPTH_FAIL_M:
                            has_fail = True
                            depth_anomalies.append({
                                'site':       site,
                                'date':       str(row['Date Time (dd/mm/yyyy hh24:mi:ss)']),
                                'value':      round(d, 2),
                                'change_pct': None,
                                'rain_event': False,
                                'validity':   'fail',
                                'note':       'data invalid, field calibration required'
                            })
                        elif d > DEPTH_WARN_M:
                            depth_anomalies.append({
                                'site':       site,
                                'date':       str(row['Date Time (dd/mm/yyyy hh24:mi:ss)']),
                                'value':      round(d, 2),
                                'change_pct': None,
                                'rain_event': False,
                                'validity':   'warning',
                                'note':       'unusually deep, verify sensor'
                            })

                # ── logger pressure step-change anomalies ───────────────────
                # Flags step changes in the SITE'S OWN barometric sensor
                # (hardware fault, drift, disconnection). Uses
                # 'Barometric Pressure(RAW)[Main Buffer] (hPa)' since that
                # column is always populated regardless of whether the site's
                # own hardware reports natively in hPa or PSI. BomBaro is
                # provided just to give context to aid to the person reviewing
                logger_pressure_col = 'Barometric Pressure(RAW)[Main Buffer] (hPa)'
                if logger_pressure_col in site_data.columns:
                    prev_logger_pressure = site_data[logger_pressure_col].shift(1)
                    site_data['logger_pressure_pct_change'] = site_data[logger_pressure_col].pct_change() * 100
                    pressure_mask = abs(site_data['logger_pressure_pct_change']) > 2
                    for idx in site_data[pressure_mask].index:
                        if pd.notna(site_data.loc[idx, logger_pressure_col]):
                            prev_val = prev_logger_pressure.loc[idx]
                            weather_val = (
                                site_data.loc[idx, 'BomBaro']
                                if 'BomBaro' in site_data.columns else None
                            )
                            pressure_anomalies.append({
                                'site':         site,
                                'date':         str(site_data.loc[idx, 'Date Time (dd/mm/yyyy hh24:mi:ss)']),
                                'prev_value':   round(float(prev_val), 1) if pd.notna(prev_val) else 'N/A',
                                'value':        round(site_data.loc[idx, logger_pressure_col], 1),
                                'weather_value': round(float(weather_val), 1) if pd.notna(weather_val) else 'N/A',
                                'change_pct':   round(site_data.loc[idx, 'logger_pressure_pct_change'], 1)
                            })

            # ── Get stats then sort ───────────────────────────────────────────────
            pct_change_anomalies = [a for a in depth_anomalies if a['validity'] == 'pct_change']
            depth_stats = {
                'max_change_pct': round(
                    max([abs(a['change_pct']) for a in pct_change_anomalies], default=0), 1
                ),
                'count': len(pct_change_anomalies)
            }
            pressure_stats = {
                'max_change_pct': round(
                    max([abs(a['change_pct']) for a in pressure_anomalies], default=0), 1
                ),
                'count': len(pressure_anomalies)
            }

            # Sort: validity failures first, then by abs % change
            depth_anomalies.sort(
                key=lambda x: (
                    0 if x['validity'] == 'fail' else (1 if x['validity'] == 'warning' else 2),
                    -abs(x['change_pct']) if x['change_pct'] is not None else 0
                )
            )
            pressure_anomalies.sort(key=lambda x: abs(x['change_pct']), reverse=True)
            depth_anomalies    = depth_anomalies[:20]
            pressure_anomalies = pressure_anomalies[:10]

            validity_fail_count = sum(1 for a in depth_anomalies if a['validity'] == 'fail')
            validity_warn_count = sum(1 for a in depth_anomalies if a['validity'] == 'warning')

            if has_fail:
                status = "FAIL"
            elif len(depth_anomalies) > 0 or len(pressure_anomalies) > 0:
                status = "WARNING"
            else:
                status = "PASS"

            details_parts = []
            if depth_adjustment_available:
                details_parts.append(f"Depth % change anomalies: {depth_stats['count']} (>15%)")
            details_parts.append(f"Pressure anomalies: {pressure_stats['count']} (>2%)")
            if validity_fail_count:
                details_parts.append(f"Invalid depth readings: {validity_fail_count} (>5m)")
            if validity_warn_count:
                details_parts.append(f"Unusually deep readings: {validity_warn_count} (>3m)")

            self.record_result(
                "Monthly Statistical Anomalies",
                "Hardware Tests",
                status,
                ", ".join(details_parts),
                {
                    'depth_anomalies':   depth_anomalies,
                    'pressure_anomalies': pressure_anomalies,
                    'depth_stats':       depth_stats,
                    'pressure_stats':    pressure_stats
                }
            )

        except (AssertionError, unittest.SkipTest):
            raise
        except Exception as e:
            self._fail("Monthly Statistical Anomalies", "Hardware Tests",
                        f"Error during anomaly detection: {e}")
   
    # ========================================================================
    # REPORT GENERATION
    # ========================================================================
    @classmethod
    def tearDownClass(cls):
        """Generate HTML report after all tests complete."""
        cls.generate_html_report()
    @classmethod
    def generate_html_report(cls):
        """Generate the HTML A/B test report."""
        timestamp = cls.test_start_time.strftime("%Y-%m-%d %H:%M:%S")
        # Calculate summary stats
        total_tests = len(cls.results)
        passed = sum(1 for r in cls.results if r['status'] == 'PASS')
        failed = sum(1 for r in cls.results if r['status'] == 'FAIL')
        warnings = sum(1 for r in cls.results if r['status'] == 'WARNING')
        skipped = sum(1 for r in cls.results if r['status'] == 'SKIP')
        # Background banner needs to be absolute path for WeasyPrint to find
        banner_abs  = os.path.join(cls.script_dir, 'images', 'gorgeMonitoring.jpg')
        banner_path = banner_abs if os.path.exists(banner_abs) else ''

        # Start HTML
        html = HTML_HEADER.format(
            timestamp=timestamp,
            banner_path=banner_path
        )
        # Summary section: title block + stoplight
        run_month    = cls.test_start_time.strftime('%B %Y')
        skipped_text = f"{skipped} skipped" if skipped > 0 else "none were skipped"
        # Pressure adjustment status + rainfall summary from validatedDepthData.csv 
        pressure_status_text = "Pressure adjustment status unavailable."
        rain_text = None
        if os.path.exists(cls.validated_output_file):
            try:
                vdf = pd.read_csv(cls.validated_output_file)
                pressure_status_text = (
                    "Depth logger values have been calibrated using nearby AS weather station pressure data." if 'Depth(m)adjusted' in vdf.columns
                    else "Weather station corrected pressure adjustments were not applied."
                )

                if 'Rainfall' in vdf.columns and 'Date Time (dd/mm/yyyy hh24:mi:ss)' in vdf.columns:
                    vdf['_dt'] = pd.to_datetime(vdf['Date Time (dd/mm/yyyy hh24:mi:ss)'], errors='coerce')
                    this_month = vdf[
                        (vdf['_dt'].dt.year == cls.test_start_time.year) &
                        (vdf['_dt'].dt.month == cls.test_start_time.month)
                    ].copy()

                    if not this_month.empty:
                        this_month['_date'] = this_month['_dt'].dt.date
                        # Take a single rainfall figure per (site, date): it's a daily
                        # BOM value merged onto every intraday row for that
                        # date (summing raw rows would multiply it by
                        # however many readings-per-day a site has).
                        daily_by_site = (
                            this_month.groupby(['Sample Point', '_date'])['Rainfall']
                            .first()
                            .reset_index()
                        )
                        coverage = daily_by_site.groupby('Sample Point')['_date'].nunique()
                        qualifying = coverage[coverage >= 28]
                        rep_site = (
                            qualifying.index[0] if len(qualifying) > 0
                            else (coverage.idxmax() if len(coverage) > 0 else None)
                        )

                        if rep_site is not None:
                            site_days = daily_by_site[daily_by_site['Sample Point'] == rep_site]
                            rain_days = int((site_days['Rainfall'] > 0).sum())
                            total_rain = float(site_days['Rainfall'].sum())

                            if total_rain <= 0:
                                rain_text = "There was no recorded rainfall in the project area this month."
                            else:
                                day_word = "day" if rain_days == 1 else "days"
                                rain_text = f"It rained on {rain_days} {day_word} this month, totalling {total_rain:.1f}mm."
            except Exception as e:
                print(f"Could not compute rainfall summary for report: {e}")

        rain_html = f'\n            <p class="stoplight-intro">{rain_text}</p>' if rain_text else ''

        html += f"""
        <div class="summary">
            <div class="summary-title-block">
                <h1>AquaTroll Depth Data Pipeline: Test Report</h1>
                <p>Generated: {timestamp} on {COMPUTER_NAME} by {USERNAME}</p>
            </div>
            <p class="stoplight-intro">For the {run_month} validation, {total_tests} checks were run, {skipped_text}. {pressure_status_text}</p>{rain_html}
            <div class="stoplight">
                <div class="stoplight-cell sl-pass">
                    <span class="stoplight-num">{passed}</span>
                    <span class="stoplight-label">Passed</span>
                </div>
                <div class="stoplight-cell sl-warning">
                    <span class="stoplight-num">{warnings}</span>
                    <span class="stoplight-label">Warning</span>
                </div>
                <div class="stoplight-cell sl-fail">
                    <span class="stoplight-num">{failed}</span>
                    <span class="stoplight-label">Failed</span>
                </div>
            </div>
        </div>
        """
        # Group by category
        categories = {}
        for result in cls.results:
            cat = result['category']
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(result)
        # Set section order 
        month_name = cls.test_start_time.strftime('%B')
        sections = [
            {
                'heading': f'{month_name} Hardware Tests',
                'categories': ['Hardware Tests'],
                'order': [
                    'Network Availability',
                    'Network Power',
                    'Monthly Statistical Anomalies',
                ],
            },
            {
                'heading': f'{month_name} Data Pipeline Tests',
                'categories': ['Data Pipeline Tests'],
                'order': [
                    'Site Configuration',
                    'Logger Portal Access',
                    'Logger Portal Navigation',
                    'Logger Portal Data Extraction',
                    'Logger Data Structure',
                    'Weather Website Access',
                    'Weather Data Structure',
                    'Depth Adjustment Verification',
                ],
            },
        ]
        # Generate sections
        for section in sections:
            # Collect all tests belonging to this section's categories
            section_tests = []
            for cat in section['categories']:
                section_tests.extend(categories.get(cat, []))
            if not section_tests:
                continue
            html += f'<div style="text-align:center; font-size:13px; font-weight:600; color:#1c1917; padding:8px 16px; margin-bottom:8px;">{section["heading"]}</div>'
            # Emit tests in the defined order; any unlisted tests append at end
            ordered = []
            remaining = list(section_tests)
            for name in section['order']:
                for t in remaining:
                    if t['test_name'] == name:
                        ordered.append(t)
                        remaining.remove(t)
                        break
            ordered.extend(remaining)  # catch any not in the explicit order list
            for test in ordered:
                status_class = test['status'].lower()
                test_description = TEST_DESCRIPTIONS.get(test['test_name'], '')
                # Interpolate dynamic values into descriptions where needed
                if test['test_name'] == 'Network Power' and test.get('data'):
                    n_sites = len(test['data'].get('battery_results', []))
                    test_description = test_description.format(n_sites=n_sites)
                html += f"""
                    <div class="test-section">
                        <h3>{test['test_name']} - <span class="{status_class}">{test['status']}</span></h3>
                    """
                if test_description:
                    html += f'<div class="test-description">{test_description}</div>'
                html += f'<p>{test["details"]}</p>'
                # Add test-specific data
                if test['data']:
                    html += cls._generate_test_data_html(test)
                html += '</div>'
        # Footer
        html += HTML_FOOTER

        # ── Save HTML ─────────────────────────────────
        output_dir = os.path.dirname(cls.report_filename)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        html_path = cls.report_filename  # already ends in .html
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html)

        # ── Convert to PDF via WeasyPrint ──
        pdf_path = os.path.splitext(html_path)[0] + '.pdf'
        try:
            from weasyprint import HTML as WeasyprintHTML
            WeasyprintHTML(filename=html_path).write_pdf(pdf_path)
            print(f"\n{'='*60}")
            print(f"TEST REPORT GENERATED: {pdf_path}")
            print(f"HTML ALSO SAVED:       {html_path}")
        except Exception:
            print(f"\n{'='*60}")
            print(f"TEST REPORT GENERATED: {html_path}")
            print(f"PDF conversion unavailable — HTML report saved instead")
        print(f"{'='*60}")
        print(f"Total: {total_tests} | Passed: {passed} | Failed: {failed} | Warnings: {warnings} | Skipped: {skipped}")
        print(f"{'='*60}\n")

    @staticmethod
    def _render_table(headers, rows, table_class=""):
        """Build a <table> from a header list and rows of pre-formatted cell values.
        Any cell needing status/flag styling should already be wrapped, e.g.
        f'<span class="{cls}">{val}</span>' — see CSS .pass/.fail/.warning/.skip,
        which are text-only styles so this is visually identical to a class on <td>.
        """
        cls_attr = f' class="{table_class}"' if table_class else ''
        parts = [f'<table{cls_attr}><tr>']
        parts += [f'<th>{h}</th>' for h in headers]
        parts.append('</tr>')
        for row in rows:
            parts.append('<tr>' + ''.join(f'<td>{c}</td>' for c in row) + '</tr>')
        parts.append('</table>')
        return ''.join(parts)

    @classmethod
    def _generate_test_data_html(cls, test):
        """Generate HTML for test-specific data tables."""
        html = ""
        test_name = test['test_name']
        data = test['data']
        if test_name == 'Network Availability':
            if data and 'recency_results' in data and data['recency_results']:
                rows = []
                for res in sorted(data['recency_results'], key=lambda x: x['site']):
                    status_class = res['status'].lower()
                    rows.append([
                        res['site'], res['date'], res['value'], res['unit'], res['days_since'],
                        f'<span class="{status_class}">{res["status"]}</span>'
                    ])
                html += cls._render_table(
                    ['Site', 'Date', 'Depth Reading', 'Unit', 'Last Datapoint', 'Status'],
                    rows
                )
        elif test_name == 'Logger Portal Access':
            if data:
                html += '<h4>Authentication Query Result:</h4><ul>'
                if 'session_cookie' in data:
                    html += f'<li>Session Cookie: {data["session_cookie"]}</li>'
                if 'final_url' in data:
                    html += f'<li>Final URL: {data["final_url"]}</li>'
                html += '</ul>'
        elif test_name == 'Logger Portal Navigation':
            if data:
                html += '<h4>Channel Navigation:</h4><ul>'
                if 'site' in data:
                    html += f'<li>Site: {data["site"]}</li>'
                if 'nav_option' in data:
                    html += f'<li>Nav Option: {data["nav_option"]}</li>'
                if 'node_or_tree_present' in data:
                    html += f'<li>Node/Tree Reference: {"&check;" if data["node_or_tree_present"] else "&cross;"}</li>'
                if 'data_table' in data:
                    html += f'<li>Data Table: {"&check;" if data["data_table"] else "&cross;"}</li>'
                html += '</ul>'
        elif test_name == 'Network Power':
            if 'battery_results' in data:
                html += '<h4>Battery Voltage Results:</h4>'
                rows = []
                for res in data['battery_results']:
                    status_class = res['status'].lower()
                    details      = res.get('details', '')
                    voltage_cell = res.get('current_voltage', 'N/A') if details == '' else details
                    last_reading = res.get('last_reading', '—')
                    rows.append([
                        res['site'], voltage_cell, last_reading,
                        f'<span class="{status_class}">{res["status"]}</span>'
                    ])
                html += cls._render_table(
                    ['Site', 'Last Voltage (V)', 'Last Reading', 'Status'],
                    rows,
                    table_class="calculation-table"
                )

                # ── Voltage trend graphs (on WARNING / FAIL only) ──
                sites_needing_graph = [
                    r for r in data['battery_results']
                    if r.get('status') in ('WARNING', 'FAIL')
                    and r.get('voltage_history')
                ]
                if sites_needing_graph:
                    html += '<h4 style="page-break-before:always;">Voltage Trend (28-day) &mdash; Sites Requiring Attention:</h4>'
                    for res in sites_needing_graph:
                        history = res['voltage_history']
                        site_id = res['site']
                        volts   = [p['voltage'] for p in history]
                        labels  = [p['date']    for p in history]
                        if not volts:
                            continue

                        v_min   = min(volts)
                        v_max   = max(volts)
                        v_range = v_max - v_min if v_max != v_min else 0.1

                        W, H         = 640, 200
                        PAD_L, PAD_R = 60, 50
                        PAD_T, PAD_B = 20, 40
                        plot_w       = W - PAD_L - PAD_R
                        plot_h       = H - PAD_T - PAD_B
                        n            = len(volts)

                        def px_x(i):
                            return PAD_L + (i / max(n - 1, 1)) * plot_w

                        def px_y(v):
                            return PAD_T + plot_h - ((v - v_min) / v_range) * plot_h

                        tw_val = res.get('threshold_warn')
                        tf_val = res.get('threshold_fail')

                        def threshold_y(tv):
                            clamped = max(v_min, min(v_max, float(tv)))
                            return px_y(clamped)

                        points = ' '.join(
                            f"{px_x(i):.1f},{px_y(v):.1f}"
                            for i, v in enumerate(volts)
                        )

                        y_ticks = ''
                        for t in range(5):
                            tick_v = v_min + t * v_range / 4
                            ty     = px_y(tick_v)
                            y_ticks += (
                                f'<line x1="{PAD_L-4}" y1="{ty:.1f}" '
                                f'x2="{PAD_L}" y2="{ty:.1f}" stroke="#999" stroke-width="1"/>'
                                f'<text x="{PAD_L-6}" y="{ty+4:.1f}" text-anchor="end" '
                                f'font-size="7" fill="#555">{tick_v:.2f}</text>'
                            )

                        x_labels = ''
                        step = max(1, n // 8)
                        for i in range(0, n, step):
                            lx = px_x(i)
                            x_labels += (
                                f'<text x="{lx:.1f}" y="{H - PAD_B + 14}" '
                                f'text-anchor="middle" font-size="7" fill="#555">{labels[i]}</text>'
                            )

                        thresh_svg = ''
                        label_x = W - PAD_R + 4  # inside right margin, left-anchored
                        if tw_val and isinstance(tw_val, (int, float)) and v_min <= tw_val <= v_max:
                            ty = threshold_y(tw_val)
                            thresh_svg += (
                                f'<line x1="{PAD_L}" y1="{ty:.1f}" x2="{W-PAD_R}" y2="{ty:.1f}" '
                                f'stroke="#c8960c" stroke-width="1.5" stroke-dasharray="5,3"/>'
                                f'<text x="{label_x}" y="{ty+4:.1f}" text-anchor="start" '
                                f'font-size="9" fill="#c8960c">warn</text>'
                            )
                        if tf_val and isinstance(tf_val, (int, float)) and v_min <= tf_val <= v_max:
                            ty = threshold_y(tf_val)
                            thresh_svg += (
                                f'<line x1="{PAD_L}" y1="{ty:.1f}" x2="{W-PAD_R}" y2="{ty:.1f}" '
                                f'stroke="#bd3c35" stroke-width="1.5" stroke-dasharray="5,3"/>'
                                f'<text x="{label_x}" y="{ty+4:.1f}" text-anchor="start" '
                                f'font-size="9" fill="#bd3c35">fail</text>'
                            )

                        last_x     = px_x(n - 1)
                        last_y     = px_y(volts[-1])
                        dot_colour = '#dc3545' if res['status'] == 'FAIL' else '#ffc107'

                        html += f"""
                        <div style="margin: 16px 0;">
                            <p style="margin:4px 0; font-weight:bold; font-size:13px;">{site_id}</p>
                            <svg viewBox="0 0 {W} {H}" width="100%" style="max-width:{W}px;
                                 background:#fff; border:1px solid #dee2e6; border-radius:4px;"
                                 xmlns="http://www.w3.org/2000/svg">
                                <line x1="{PAD_L}" y1="{PAD_T}" x2="{PAD_L}" y2="{PAD_T+plot_h}"
                                      stroke="#ccc" stroke-width="1"/>
                                <line x1="{PAD_L}" y1="{PAD_T+plot_h}" x2="{W-PAD_R}" y2="{PAD_T+plot_h}"
                                      stroke="#ccc" stroke-width="1"/>
                                {y_ticks}
                                {x_labels}
                                {thresh_svg}
                                <polyline points="{points}" fill="none"
                                          stroke="#bd3c35" stroke-width="2" stroke-linejoin="round"/>
                                <circle cx="{last_x:.1f}" cy="{last_y:.1f}" r="4"
                                        fill="{dot_colour}" stroke="white" stroke-width="1.5"/>
                                <text x="{last_x:.1f}" y="{last_y-8:.1f}" text-anchor="middle"
                                      font-size="10" fill="{dot_colour}" font-weight="bold">
                                    {volts[-1]:.3f}V
                                </text>
                            </svg>
                        </div>"""
        elif test_name == 'Site Configuration':
            if data.get('enabled'):
                html += f'<h4>Enabled Sites ({len(data["enabled"])}):</h4><ul>'
                for site in data['enabled']:
                    html += f'<li>&check; {site}</li>'
                html += '</ul>'
            if data.get('disabled'):
                html += f'<h4>Disabled Sites ({len(data["disabled"])}):</h4><ul>'
                for site in data['disabled']:
                    html += f'<li>&check; {site}</li>'
                html += '</ul>'
        elif test_name == 'Depth Adjustment Verification':
            if 'calculations' in data:
                html += '<h4>Calculation Results:</h4>'
                rows = []
                for calc in data['calculations']:
                    result   = '<span class="pass">&check;</span>' if calc['passed'] else '<span class="fail">&cross;</span>'
                    adj_text = 'Yes' if calc['adjustment_applied'] else 'No'
                    reason   = calc.get('reason', '')
                    rows.append([
                        calc['test_num'], calc['site'], calc['depth_raw'], calc['bom_baro'],
                        calc['logger_baro'], adj_text, calc['calculated'], calc['expected'],
                        calc['difference'],
                        f'<span style="color:#6c757d;font-style:italic;font-size:0.9em;">{reason}</span>',
                        result
                    ])
                html += cls._render_table(
                    ['Test', 'Site', 'Depth', 'Weather Station', 'AquaTroll', 'Adj?',
                     'Calc', 'Expected', 'Diff', 'Reason', '&check;'],
                    rows,
                    table_class="calculation-table"
                )
        elif test_name == 'Monthly Statistical Anomalies':
            pct_anomalies      = [a for a in data.get('depth_anomalies', []) if a.get('validity') == 'pct_change']
            validity_anomalies = [a for a in data.get('depth_anomalies', []) if a.get('validity') in ('fail', 'warning')]

            if pct_anomalies:
                html += '<h4>Depth % Change Anomalies:</h4><div class="anomaly">'
                html += f"<p>Max change: {data['depth_stats']['max_change_pct']}%</p>"
                rows = []
                for a in pct_anomalies:
                    rain_status = 'Yes' if a.get('rain_event', False) else 'No'
                    rows.append([
                        a['site'], a['date'], a.get('prev_value', 'N/A'), a['value'],
                        f"{a['change_pct']}%", rain_status
                    ])
                html += cls._render_table(
                    ['Site', 'Date', 'Previous (m)', 'New (m)', 'Change %', 'Rain Event (Y/N)'],
                    rows
                )
                html += '</div>'

            if validity_anomalies:
                html += '<h4>Invalid Depth Readings:</h4><div class="anomaly">'
                rows = []
                for a in validity_anomalies:
                    flag_class = 'fail' if a['validity'] == 'fail' else 'warning'
                    flag_label = 'FAIL' if a['validity'] == 'fail' else 'WARNING'
                    rows.append([
                        a['site'], a['date'], a['value'],
                        f'<span class="{flag_class}">{flag_label}</span>',
                        a.get('note', '')
                    ])
                html += cls._render_table(
                    ['Site', 'Date', 'Depth (m)', 'Flag', 'Note'],
                    rows
                )
                html += '</div>'

            if data.get('pressure_anomalies'):
                html += '<h4>Logger Pressure Anomalies:</h4><div class="anomaly">'
                html += f"<p>Max change: {data['pressure_stats']['max_change_pct']}%</p>"
                rows = [
                    [a['site'], a['date'], a.get('prev_value', 'N/A'), a['value'],
                     a.get('weather_value', 'N/A'), f"{a['change_pct']}%"]
                    for a in data['pressure_anomalies']
                ]
                html += cls._render_table(
                    ['Site', 'Date', 'Previous (hPa)', 'New (hPa)', 'Weather Station (hPa)', 'Change %'],
                    rows
                )
                html += '</div>'
        return html
# ============================================================================
# MAIN EXECUTION
# ============================================================================
def run_tests(output_path='transformed_data/abTestsReport.html'):
    """
    Execute tests and generate HTML report.
    Args:
        output_path (str): Path for the HTML report
    Returns:
        bool: True if all tests passed
    """
    TestEnvironmentalPipeline.report_filename = output_path
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestEnvironmentalPipeline)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()
if __name__ == '__main__':
    success = run_tests()
    exit(0 if success else 1)