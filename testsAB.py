"""
AquaTroll Depth Data Pipeline
A/B tests for the data scraping, validation & processing pipeline.
Monthly verification of script performance through testing of:
- Data connectivity and authentication
- Validation of returned data structures
- Battery health checks
- Verification of depth calculations
- Site availability checks
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

# Third-Party Libraries
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
        "Verifies if weather website accessible "
        "and returns the received HTTP response code."
    ),
    "Logger Portal Access": (
        "Checks connectivity & provided authentication credentials allow for a "
        "successful login to the portal website."
    ),
    "Logger Portal Navigation": (
        "Verifies ability to navigate and identify which display structure is in use (node or tables)"
    ),
    "Logger Portal Data Extraction": (
        "Confirms the presence and functionality of CSV download button for data "
        "extraction."
    ),
    "Weather Data Structure": (
        "Verifies weather website accessibility and confirms the presence of pressure data "
        "at the expected positions (9 am at position 15, 3 pm at position 21) and rainfall "
        "data at position 4."
    ),
    "Logger Data Structure": (
        "Checks that downloaded logger CSV files contain all required columns for"
        "for analysis (Date, Time, Level and Battery.)"
    ),
    "Network Power": (
        "Verifies battery voltage across {n_sites} active sites."
    ),
    "Site Configuration": (
        "Checks which sites are setup by reading from the SITES_CONFIG env variable, "
    ),
    "Network Availability": (
        "Locates last recorded depth reading for each enabled site. "
        "Helpful for determining whether a pool has gone dry or if the equipment is faulty."
    ),
    "Depth Adjustment Verification": (
        "Independently tests barometric pressure adjustments using downloaded files "
        "then cross-checks results against the pipeline's final csv results. "
        "Applies thresholds: depth > 0.3m and baro difference > 5 hPa to prevent "
        "corrections from very shallow/dry pools and sensor noise."
    ),
    "Monthly Statistical Anomalies": (
        "Analyses validated data for any statistical anomalies; Namely by flagging "
        "depth changes >15%, unrealistic values and/or pressure changes >2% for review."
    )
}
# ============================================================================
# HTML REPORT TEMPLATES
# ============================================================================
HTML_HEADER = """
<!DOCTYPE html>
<html>
<head>
    <title>Pipeline Test Report - {timestamp}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                  color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
        .banner {{ width: 100%; height: 150px; object-fit: cover; border-radius: 8px;
                   margin-bottom: 20px; background-color: #e0e0e0; }}
        .summary {{ background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px;
                   box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .test-section {{ background: white; padding: 20px; margin-bottom: 20px;
                        border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .test-description {{ color: #6c757d; font-style: italic; margin: 10px 0; padding: 10px;
                            background-color: #f8f9fa; border-left: 4px solid #667eea; }}
        .pass {{ color: #28a745; font-weight: bold; }}
        .fail {{ color: #dc3545; font-weight: bold; }}
        .warning {{ color: #ffc107; font-weight: bold; }}
        .skip {{ color: #6c757d; font-style: italic; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
        th {{ background-color: #f8f9fa; padding: 10px; text-align: left; border-bottom: 2px solid #dee2e6; }}
        td {{ padding: 8px; border-bottom: 1px solid #dee2e6; }}
        .column-list {{ background: #f8f9fa; padding: 10px; border-radius: 4px; margin: 10px 0; }}
        .calculation-table {{ margin-top: 15px; }}
        .anomaly {{ background-color: #fff3cd; }}
        .footer {{ text-align: center; color: #6c757d; margin-top: 40px; padding: 20px; }}
        pre {{ background: #f8f9fa; padding: 10px; border-radius: 4px; overflow-x: auto; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🔬 AquaTroll Depth Data Pipeline: Test Report</h1>
        <p>Generated: {timestamp} on {computer_name} by {username}</p>
    </div>
    <img src="images/gorgeMonitoring.jpg" alt="Gorge Monitoring Banner" class="banner" onerror="this.style.display='none'">
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
    @classmethod
    def _ensure_downloads_exist(cls):
        """
        Diagnostic fallback: if data_downloads/ is missing or empty, silently
        run httpLoggerScraper.run() to produce the CSVs.  This lets testsAB.py
        function as a standalone diagnostic tool even when the pipeline hasn't run.
        """
        csv_files = []
        if os.path.exists(cls.data_downloads_path):
            csv_files = [
                f for f in os.listdir(cls.data_downloads_path)
                if f.endswith('.csv') and 'baro' not in f.lower()
                and f != 'weather_data.csv'
            ]
        if not csv_files:
            try:
                import httpLoggerScraper
                logging.getLogger(__name__).warning(
                    "data_downloads/ empty — running httpLoggerScraper in diagnostic mode"
                )
                httpLoggerScraper.run(cls.data_downloads_path)
            except Exception as e:
                logging.getLogger(__name__).error(
                    f"Diagnostic scrape failed: {e}"
                )

    @classmethod
    def _ensure_validated_exists(cls):
        """
        Diagnostic fallback: if validatedDepthData.csv is missing, silently run
        dataValidation.consolidate_csv_files() to produce it.
        """
        if not os.path.exists(cls.validated_output_file):
            try:
                import dataValidation
                os.makedirs(cls.transformed_data_path, exist_ok=True)
                logging.getLogger(__name__).warning(
                    "validatedDepthData.csv not found — running dataValidation in diagnostic mode"
                )
                dataValidation.consolidate_csv_files(
                    cls.data_downloads_path,
                    cls.validated_output_file
                )
            except Exception as e:
                logging.getLogger(__name__).error(
                    f"Diagnostic validation failed: {e}"
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
   
    # ========================================================================
    # CONNECTIVITY TESTS
    # ========================================================================
    def test_01_weather_website_accessibility(self):
        """Test connectivity to weather website."""
        if not self.weather_url:
            self.record_result(
                "Weather Website Access",
                "Data Pipeline Tests",
                "FAIL",
                "WEATHER_URL not configured in environment"
            )
            self.fail("WEATHER_URL not configured")
            return
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
                self.record_result(
                    "Weather Website Access",
                    "Data Pipeline Tests",
                    "FAIL",
                    f"Weather website returned HTTP {response.status_code}"
                )
                self.fail(f"Weather website returned HTTP {response.status_code}")
        except requests.exceptions.Timeout:
            self.record_result(
                "Weather Website Access",
                "Data Pipeline Tests",
                "FAIL",
                "Connection timeout after 10 seconds"
            )
            self.fail("Connection timeout")
        except requests.exceptions.RequestException as e:
            self.record_result(
                "Weather Website Access",
                "Data Pipeline Tests",
                "FAIL",
                f"Connection error: {str(e)}"
            )
            self.fail(f"Connection error: {e}")
        except Exception as e:
            self.record_result(
                "Weather Website Access",
                "Data Pipeline Tests",
                "FAIL",
                f"Unexpected error: {str(e)}"
            )
            self.fail(f"Unexpected error: {e}")
    def test_02_logger_portal_authentication(self):
        """Test authentication to the logger portal using HTTP."""
        if not self.login_url:
            self.record_result(
                "Logger Portal Access",
                "Data Pipeline Tests",
                "SKIP",
                "LOGIN_URL not configured in environment"
            )
            self.skipTest("LOGIN_URL not configured")
            return
        login_username = os.getenv("LOGIN_USERNAME", "")
        login_password = os.getenv("LOGIN_PASSWORD", "")
        if not login_username or not login_password:
            self.record_result(
                "Logger Portal Access",
                "Data Pipeline Tests",
                "FAIL",
                "LOGIN_USERNAME or LOGIN_PASSWORD not configured"
            )
            self.fail("Login credentials not configured")
            return
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        try:
            # Create session
            session = requests.Session()
            # Get login page
            response = session.get(self.login_url, headers=headers, timeout=10)
            if response.status_code != 200:
                self.record_result(
                    "Logger Portal Access",
                    "Data Pipeline Tests",
                    "FAIL",
                    f"Login page not accessible (HTTP {response.status_code})"
                )
                self.__class__.auth_session = None
                self.fail(f"Login page returned HTTP {response.status_code}")
                return
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
                self.record_result(
                    "Logger Portal Access",
                    "Data Pipeline Tests",
                    "FAIL",
                    "Authentication failed - credentials rejected (remained on login page)"
                )
                self.__class__.auth_session = None
                self.fail("Authentication failed")
                return
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
        except requests.exceptions.Timeout:
            self.record_result(
                "Logger Portal Access",
                "Data Pipeline Tests",
                "FAIL",
                "Connection timeout"
            )
            self.__class__.auth_session = None
            self.fail("Connection timeout")
        except requests.exceptions.RequestException as e:
            self.record_result(
                "Logger Portal Access",
                "Data Pipeline Tests",
                "FAIL",
                f"Connection error: {str(e)}"
            )
            self.__class__.auth_session = None
            self.fail(f"Connection error: {e}")
        except Exception as e:
            self.record_result(
                "Logger Portal Access",
                "Data Pipeline Tests",
                "FAIL",
                f"Authentication error: {str(e)}"
            )
            self.__class__.auth_session = None
            self.fail(f"Authentication error: {e}")
    def test_03_logger_portal_node_selection(self):
        """Test ability to navigate to and select nodes/tables in the logger portal."""
        if not hasattr(self.__class__, 'auth_session') or self.__class__.auth_session is None:
            self.record_result(
                "Logger Portal Navigation",
                "Data Pipeline Tests",
                "SKIP",
                "Authentication session not available"
            )
            self.skipTest("No authenticated session available")
            return
        # Get first enabled site
        enabled_sites = {name: config for name, config in self.site_config.items()
                         if config.get('enabled', False)}
        if not enabled_sites:
            self.record_result(
                "Logger Portal Navigation",
                "Data Pipeline Tests",
                "SKIP",
                "No enabled sites configured"
            )
            self.skipTest("No enabled sites available")
            return
        # Pick a random enabled site
        test_site_name = random.choice(list(enabled_sites.keys()))
        test_site_config = enabled_sites[test_site_name]
        nav_option = test_site_config.get('nav_option')
        if not nav_option:
            self.record_result(
                "Logger Portal Navigation",
                "Data Pipeline Tests",
                "FAIL",
                f"Site {test_site_name} missing nav_option in configuration"
            )
            self.__class__.channel_response = None
            self.fail("Navigation option not configured")
            return
        base_url = os.getenv("BASE_URL", "").rstrip('/')
        channel_url = f'{base_url}/data-channels.aspx?id={nav_option}'
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        try:
            # Navigate to channel page
            response = self.__class__.auth_session.get(channel_url, headers=headers, timeout=15)
            if response.status_code != 200:
                self.record_result(
                    "Logger Portal Navigation",
                    "Data Pipeline Tests",
                    "FAIL",
                    f"Failed to navigate to channel (HTTP {response.status_code})"
                )
                self.__class__.channel_response = None
                self.fail(f"Navigation returned HTTP {response.status_code}")
                return
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
                self.record_result(
                    "Logger Portal Navigation",
                    "Data Pipeline Tests",
                    "FAIL",
                    f"Channel page missing expected elements (score: {validation_score}/2)"
                )
                self.fail("Channel page validation failed")
        except requests.exceptions.Timeout:
            self.record_result(
                "Logger Portal Navigation",
                "Data Pipeline Tests",
                "FAIL",
                "Navigation timeout"
            )
            self.__class__.channel_response = None
            self.fail("Navigation timeout")
        except requests.exceptions.RequestException as e:
            self.record_result(
                "Logger Portal Navigation",
                "Data Pipeline Tests",
                "FAIL",
                f"Navigation error: {str(e)}"
            )
            self.__class__.channel_response = None
            self.fail(f"Navigation error: {e}")
    def test_04_logger_portal_csv_download(self):
        """Test presence of CSV download functionality."""
        if not hasattr(self.__class__, 'channel_response') or self.__class__.channel_response is None:
            self.record_result(
                "Logger Portal Data Extraction",
                "Data Pipeline Tests",
                "SKIP",
                "No channel navigation response available"
            )
            self.skipTest("No channel page available")
            return
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
                self.record_result(
                    "Logger Portal Data Extraction",
                    "Data Pipeline Tests",
                    "FAIL",
                    f"Neither table selection nor CSV button found on channel page"
                )
                self.fail("CSV download functionality not available")
        except Exception as e:
            self.record_result(
                "Logger Portal Data Extraction",
                "Data Pipeline Tests",
                "FAIL",
                f"Error checking download functionality: {str(e)}"
            )
            self.fail(f"Download check failed: {e}")
   
    # ========================================================================
    # DATA STRUCTURE TESTS
    # ========================================================================
    def test_05_weather_data_structure(self):
        """Test weather website scraping capability and data structure."""
        if not self.weather_url:
            self.record_result(
                "Weather Data Structure",
                "Data Pipeline Tests",
                "SKIP",
                "WEATHER_URL not configured - skipping test"
            )
            self.skipTest("WEATHER_URL not configured")
            return
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        try:
            response = requests.get(self.weather_url, headers=headers, timeout=10)
            if response.status_code != 200:
                self.record_result(
                    "Weather Data Structure",
                    "Data Pipeline Tests",
                    "FAIL",
                    f"Could not access weather website (HTTP {response.status_code})"
                )
                self.fail(f"Weather site returned HTTP {response.status_code}")
                return
            # Parse the page
            soup = BeautifulSoup(response.text, 'html.parser')
            table = soup.select_one('table.data')
            if not table:
                self.record_result(
                    "Weather Data Structure",
                    "Data Pipeline Tests",
                    "FAIL",
                    "Table with class 'data' not found in HTML"
                )
                self.fail("Weather table not found")
                return
            # Get data rows from tbody
            data_rows = table.select('tbody tr')
            if len(data_rows) == 0:
                self.record_result(
                    "Weather Data Structure",
                    "Data Pipeline Tests",
                    "FAIL",
                    "No data rows found in weather table"
                )
                self.fail("No data rows in table")
                return
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
                self.record_result(
                    "Weather Data Structure",
                    "Data Pipeline Tests",
                    "FAIL",
                    "No valid data row found in weather table"
                )
                self.fail("No valid data row")
                return
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
                    f"Weather table structure verified: {total_cells} columns, rainfall at position 4 ({position_4_value} mm), pressure data at positions 15 ({position_15_value} hPa) and 21 ({position_21_value} hPa)"
                )
            else:
                issues = []
                if not position_4_valid:
                    issues.append(f"Position 4 (rainfall) invalid (got: '{position_4_value}')")
                if not position_15_valid:
                    issues.append(f"Position 15 invalid (got: '{position_15_value}')")
                if not position_21_valid:
                    issues.append(f"Position 21 invalid (got: '{position_21_value}')")
                self.record_result(
                    "Weather Data Structure",
                    "Data Pipeline Tests",
                    "FAIL",
                    f"Weather table data validation failed: {', '.join(issues)}"
                )
                self.fail("Weather data not in expected positions")
        except requests.exceptions.Timeout:
            self.record_result(
                "Weather Data Structure",
                "Data Pipeline Tests",
                "FAIL",
                "Connection timeout accessing weather website"
            )
            self.fail("Connection timeout")
        except requests.exceptions.RequestException as e:
            self.record_result(
                "Weather Data Structure",
                "Data Pipeline Tests",
                "FAIL",
                f"Connection error: {str(e)}"
            )
            self.fail(f"Connection error: {e}")
        except Exception as e:
            self.record_result(
                "Weather Data Structure",
                "Data Pipeline Tests",
                "FAIL",
                f"Error parsing weather data: {str(e)}"
            )
            self.fail(f"Failed to parse weather data: {e}")
    def test_06_logger_data_structure(self):
        """Test logger CSV data structure."""
        self.__class__._ensure_downloads_exist()
        if not os.path.exists(self.data_downloads_path):
            self.record_result(
                "Logger Data Structure",
                "Data Pipeline Tests",
                "FAIL",
                "data_downloads directory not found"
            )
            self.fail("data_downloads directory not available - pipeline may not have run")
            return
        csv_files = [f for f in os.listdir(self.data_downloads_path) if f.endswith('.csv') and f != 'weather_data.csv' and 'baro' not in f.lower()]
        if not csv_files:
            self.record_result(
                "Logger Data Structure",
                "Data Pipeline Tests",
                "FAIL",
                "No logger CSV files found in data_downloads"
            )
            self.fail("No logger CSV files available - pipeline may not have run")
            return
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
            # Check battery column (prefix match — unit suffix varies)
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
                    f"Logger data has required columns: Date, Time, {level_col}{batt_note}"
                )
            else:
                missing = []
                if not base_present:
                    missing_base = [col for col in base_columns if col not in df.columns]
                    missing.extend(missing_base)
                if not level_present:
                    missing.append(f"Level column (expected one of: {', '.join(level_column_options)})")
                self.record_result(
                    "Logger Data Structure",
                    "Data Pipeline Tests",
                    "FAIL",
                    f"Missing required columns: {', '.join(missing)}"
                )
                self.fail("Required columns not found")
        except Exception as e:
            self.record_result(
                "Logger Data Structure",
                "Data Pipeline Tests",
                "FAIL",
                f"Error reading logger data: {str(e)}"
            )
            self.fail(f"Failed to read logger data: {e}")
    def test_07_battery_voltage_check(self):
        """Check battery voltage for enabled sites.

        Thresholds (ERP3 / ERP4 use Starlink ~13V; all other sites use logger ~3.5V):
          WARNING : voltage < warn_threshold  OR  last reading > 7 days ago
          FAIL    : voltage < fail_threshold  OR  last reading > 28 days ago
          PASS    : all checks satisfied

        The battery_results list carries a 'voltage_history' key (list of
        {date, voltage} dicts) for any site whose status is WARNING or FAIL,
        so the HTML renderer can draw a trend graph.
        """
        self.__class__._ensure_downloads_exist()
        if not os.path.exists(self.data_downloads_path):
            self.record_result(
                "Network Power",
                "Hardware Tests",
                "FAIL",
                "data_downloads directory not found"
            )
            self.fail("data_downloads directory not available")
            return

        enabled_sites = {
            name: config for name, config in self.site_config.items()
            if config.get('enabled', False)
        }
        if not enabled_sites:
            self.record_result(
                "Network Power",
                "Hardware Tests",
                "SKIP",
                "No enabled sites configured"
            )
            self.skipTest("No enabled sites available")
            return

        # Per-site threshold config.
        # ERP3 / ERP4 are Starlink-powered; everything else is logger battery.
        STARLINK_SITES = {'ERP3', 'ERP4'}
        THRESHOLDS = {
            'starlink': {'warn': 13.2, 'fail': 13.05},
            'logger':   {'warn': 3.6,  'fail': 3.5},
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

                # ── check data recency (independent of voltage) ─────────────
                if df['_datetime'].notna().any():
                    latest_dt   = df['_datetime'].max()
                    days_since  = (now - latest_dt).days
                else:
                    days_since  = None   # unknown

                recency_status = 'PASS'
                if days_since is None or days_since > STALE_FAIL_DAYS:
                    recency_status = 'FAIL'
                elif days_since > STALE_WARN_DAYS:
                    recency_status = 'WARNING'

                # ── no valid battery readings at all ────────────────────────
                if battery_valid.empty:
                    # Still report recency-based status if we can
                    effective_status = recency_status if recency_status == 'FAIL' else 'WARNING'
                    detail_msg = 'No valid battery data'
                    if days_since is not None:
                        detail_msg += f' (last datapoint {days_since}d ago)'
                    battery_results.append({
                        'site': site_id,
                        'status': effective_status,
                        'details': detail_msg,
                        'threshold_warn': thresh_warn,
                        'threshold_fail': thresh_fail,
                    })
                    if effective_status == 'FAIL':
                        fail_count += 1
                    else:
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

                # ── voltage-based status ─────────────────────────────────────
                if current_voltage < thresh_fail:
                    voltage_status = 'FAIL'
                elif current_voltage < thresh_warn:
                    voltage_status = 'WARNING'
                else:
                    voltage_status = 'PASS'

                # ── combine voltage + recency into worst-case status ─────────
                priority = {'FAIL': 2, 'WARNING': 1, 'PASS': 0}
                effective_status = max(
                    [voltage_status, recency_status],
                    key=lambda s: priority[s]
                )

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
                self.record_result(
                    "Site Configuration",
                    "Data Pipeline Tests",
                    "FAIL",
                    "No enabled sites found in SITES_CONFIG",
                    config_data
                )
                self.fail("No enabled sites configured")
            else:
                self.record_result(
                    "Site Configuration",
                    "Data Pipeline Tests",
                    "PASS",
                    f"Enabled: {len(enabled_sites)} sites; Disabled: {len(disabled_sites)} sites",
                    config_data
                )
        except Exception as e:
            self.record_result(
                "Site Configuration",
                "Data Pipeline Tests",
                "FAIL",
                f"Configuration error: {str(e)}"
            )
            self.fail(f"Site configuration test failed: {e}")
   
    # ========================================================================
    # CALCULATION TESTS
    # ========================================================================
    def test_09_depth_calculation_verification(self):
        """Validates result accuracy of depth calculations.

        Imports calculate_adjusted_depth() directly from dataValidation so the
        test always exercises the same code path as the pipeline — not a local
        copy that could drift out of sync.

        Samples 3 rows from validatedDepthData.csv, feeds each row as a
        single-row DataFrame to calculate_adjusted_depth(), then cross-checks
        the returned Depth(m)adjusted value against what the pipeline already
        recorded, within a 2 cm tolerance.
        """
        self.__class__._ensure_validated_exists()
        if not os.path.exists(self.validated_output_file):
            self.record_result(
                "Depth Adjustment Verification",
                "Data Pipeline Tests",
                "SKIP",
                "No validated data found — pipeline has not run and diagnostic fallback failed"
            )
            self.skipTest("Validated output file not available")
            return
        try:
            # Lazy import — isolated to this test so a broken dataValidation
            # module does not crash the entire test runner at startup.
            from dataValidation import calculate_adjusted_depth

            validated_df = load_csv_safely(self.validated_output_file)
            if validated_df is None:
                self.skipTest("Could not load validated output file")
                return

            # Filter for rows that have both raw and adjusted depth values
            valid_mask = (
                validated_df['Depth(m)raw'].notna() &
                validated_df['Depth(m)adjusted'].notna() &
                (pd.to_numeric(validated_df['Depth(m)raw'], errors='coerce') > 0) &
                (pd.to_numeric(validated_df['Depth(m)adjusted'], errors='coerce') > 0)
            )
            filtered_df = validated_df[valid_mask].copy()

            if len(filtered_df) == 0:
                self.record_result(
                    "Depth Adjustment Verification",
                    "Data Pipeline Tests",
                    "SKIP",
                    "No valid depth data in validated output"
                )
                self.skipTest("No valid depth data available")
                return

            num_samples = min(3, len(filtered_df))
            test_samples = filtered_df.sample(n=num_samples, random_state=42)

            calculation_results = []
            all_passed = True
            tolerance = 0.02  # 2 cm

            for idx, row in test_samples.iterrows():
                site = row['Sample Point']
                date = row['Date Time (dd/mm/yyyy hh24:mi:ss)']
                expected_adjusted = float(row['Depth(m)adjusted'])

                # Build a single-row DataFrame matching the schema that
                # calculate_adjusted_depth() expects, using the same column
                # names produced by dataValidation.process_site_file().
                row_df = pd.DataFrame([{
                    'Depth(m)raw':                                  pd.to_numeric(row['Depth(m)raw'], errors='coerce'),
                    'BomBaro':                                       pd.to_numeric(row.get('BomBaro'), errors='coerce'),
                    'Barometric Pressure(RAW)[Main Buffer] (hPa)':  pd.to_numeric(row.get('Barometric Pressure(RAW)[Main Buffer] (hPa)'), errors='coerce'),
                    'Pressure(RAW)[Main Buffer] (PSI)':             pd.to_numeric(row.get('Pressure(RAW)[Main Buffer] (PSI)'), errors='coerce'),
                    'OTHER - Comments - Text':                       '',
                }])

                result_df = calculate_adjusted_depth(row_df)
                calculated_adjusted = float(result_df['Depth(m)adjusted'].iloc[0])

                # If calculate_adjusted_depth returned NaN (e.g. no BOM data),
                # fall back to raw depth — matching what the pipeline itself does.
                if pd.isna(calculated_adjusted):
                    calculated_adjusted = float(row_df['Depth(m)raw'].iloc[0])

                # Identify whether adjustment was applied
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
                })

            self.record_result(
                "Depth Adjustment Verification",
                "Data Pipeline Tests",
                "PASS" if all_passed else "FAIL",
                f"Tested {len(calculation_results)} calculations against dataValidation.calculate_adjusted_depth()",
                {'calculations': calculation_results}
            )
            if not all_passed:
                self.fail("One or more calculated depths did not match expected values within tolerance")

        except ImportError:
            self.record_result(
                "Depth Adjustment Verification",
                "Data Pipeline Tests",
                "FAIL",
                "Could not import calculate_adjusted_depth from dataValidation.py — ensure it is on the Python path"
            )
            self.fail("dataValidation import failed")
        except Exception as e:
            self.record_result(
                "Depth Adjustment Verification",
                "Data Pipeline Tests",
                "FAIL",
                f"Error during calculation verification: {str(e)}"
            )
            self.fail(f"Calculation verification failed: {e}")
   
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
        self.__class__._ensure_downloads_exist()
        STALE_WARN_DAYS = 7
        STALE_FAIL_DAYS = 28

        if not os.path.exists(self.data_downloads_path):
            self.record_result(
                "Network Availability",
                "Hardware Tests",
                "FAIL",
                "data_downloads directory not found"
            )
            self.fail("data_downloads directory not available")
            return

        enabled_sites = {
            name: config for name, config in self.site_config.items()
            if config.get('enabled', False)
        }

        if not enabled_sites:
            self.record_result(
                "Network Availability",
                "Hardware Tests",
                "SKIP",
                "No enabled sites configured"
            )
            self.skipTest("No enabled sites available")
            return

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
            f"Checked {len(enabled_sites)} enabled sites: "
            f"{pass_count} current, "
            f"{warning_count} stale (>7 days), "
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
          - Percentage change is mathematically unstable when the previous
            value is at or near zero. Rows where the previous depth reading
            is <= MIN_DEPTH_FOR_PCT (0.05 m) are excluded from the % change
            check to prevent false flags.

        Depth validity:
          - Depth(m)adjusted > 3 m → WARNING  ('unusually deep, verify sensor')
          - Depth(m)adjusted > 5 m → FAIL     ('data invalid, field calibration required')
          These are appended to depth_anomalies with a 'validity' key so
          the HTML renderer can distinguish them from % change anomalies.
        """
        MIN_DEPTH_FOR_PCT = 0.05   # metres — below this, % change is meaningless
        DEPTH_WARN_M      = 3.0
        DEPTH_FAIL_M      = 5.0

        self.__class__._ensure_validated_exists()
        if not os.path.exists(self.validated_output_file):
            self.record_result(
                "Monthly Statistical Anomalies",
                "Hardware Tests",
                "SKIP",
                "Statistical analysis requires validated data files"
            )
            self.skipTest("Validated output file not available")
            return
        try:
            df = load_csv_safely(self.validated_output_file)
            if df is None:
                self.skipTest("Could not load validated output file")
                return

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

                # ── pressure % change anomalies ─────────────────────────────
                if 'BomBaro' in site_data.columns:
                    site_data['pressure_pct_change'] = site_data['BomBaro'].pct_change() * 100
                    pressure_mask = abs(site_data['pressure_pct_change']) > 2
                    for idx in site_data[pressure_mask].index:
                        if pd.notna(site_data.loc[idx, 'BomBaro']):
                            pressure_anomalies.append({
                                'site':       site,
                                'date':       str(site_data.loc[idx, 'Date Time (dd/mm/yyyy hh24:mi:ss)']),
                                'value':      round(site_data.loc[idx, 'BomBaro'], 1),
                                'change_pct': round(site_data.loc[idx, 'pressure_pct_change'], 1)
                            })

            # ── deduplicate validity flags (keep worst per site) ────────────
            # A site appearing in both pct_change and validity rows is fine —
            # keep all; the HTML renderer shows them in separate rows.

            # ── stats and sort ───────────────────────────────────────────────
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

            details_parts = [
                f"Depth % change anomalies: {depth_stats['count']} (>15%)",
                f"Pressure anomalies: {pressure_stats['count']} (>2%)",
            ]
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
            # Status already recorded via record_result above — do not call
            # self.fail() here as it would trigger the except handler and
            # generate a duplicate FAIL entry in the report.

        except Exception as e:
            self.record_result(
                "Monthly Statistical Anomalies",
                "Hardware Tests",
                "FAIL",
                f"Error during anomaly detection: {str(e)}"
            )
            self.fail(f"Anomaly detection failed: {e}")
   
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
        # Start HTML
        html = HTML_HEADER.format(
            timestamp=timestamp,
            computer_name=COMPUTER_NAME,
            username=USERNAME
        )
        # Summary section
        html += f"""
        <div class="summary">
            <h2>Test Summary</h2>
            <table>
                <tr><th>Total Tests</th><td>{total_tests}</td></tr>
                <tr><th>Passed</th><td class="pass">{passed}</td></tr>
                <tr><th>Failed</th><td class="fail">{failed}</td></tr>
                <tr><th>Warnings</th><td class="warning">{warnings}</td></tr>
                <tr><th>Skipped</th><td class="skip">{skipped}</td></tr>
            </table>
        </div>
        """
        # Group by category
        categories = {}
        for result in cls.results:
            cat = result['category']
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(result)
        # Section definitions — order within each section matches desired report order
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
            html += f'<h2>{section["heading"]}</h2>'
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
        # Save report
        output_dir = os.path.dirname(cls.report_filename)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(cls.report_filename, 'w', encoding='utf-8') as f:
            f.write(html)
        print(f"\n{'='*60}")
        print(f"TEST REPORT GENERATED: {cls.report_filename}")
        print(f"{'='*60}")
        print(f"Total: {total_tests} | Passed: {passed} | Failed: {failed} | Warnings: {warnings} | Skipped: {skipped}")
        print(f"{'='*60}\n")
    @classmethod
    def _generate_test_data_html(cls, test):
        """Generate HTML for test-specific data tables."""
        html = ""
        test_name = test['test_name']
        data = test['data']
        if test_name == 'Network Availability':
            if data and 'recency_results' in data and data['recency_results']:
                html += '<table>'
                html += '<tr><th>Site</th><th>Date</th><th>Depth Reading</th><th>Unit</th><th>Last Datapoint</th><th>Status</th></tr>'
                for res in sorted(data['recency_results'], key=lambda x: x['site']):
                    status_class = res['status'].lower()
                    html += f"""<tr>
                        <td>{res['site']}</td>
                        <td>{res['date']}</td>
                        <td>{res['value']}</td>
                        <td>{res['unit']}</td>
                        <td>{res['days_since']}</td>
                        <td class="{status_class}">{res['status']}</td>
                    </tr>"""
                html += '</table>'
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
                    html += f'<li>Node/Tree Reference: {"✓" if data["node_or_tree_present"] else "✗"}</li>'
                if 'data_table' in data:
                    html += f'<li>Data Table: {"✓" if data["data_table"] else "✗"}</li>'
                html += '</ul>'
        elif test_name == 'Network Power':
            if 'battery_results' in data:
                html += '<h4>Battery Voltage Results:</h4>'
                html += '<table class="calculation-table">'
                html += (
                    '<tr><th>Site</th>'
                    '<th>Last Voltage (V)</th><th>Last Reading</th>'
                    '<th>Status</th></tr>'
                )
                for res in data['battery_results']:
                    status_class = res['status'].lower()
                    details      = res.get('details', '')
                    voltage_cell = res.get('current_voltage', 'N/A') if details == '' else details
                    last_reading = res.get('last_reading', '—')
                    html += f"""<tr>
                        <td>{res['site']}</td>
                        <td>{voltage_cell}</td>
                        <td>{last_reading}</td>
                        <td class="{status_class}">{res['status']}</td>
                    </tr>"""
                html += '</table>'

                # ── conditional voltage trend graphs (WARNING / FAIL only) ──
                sites_needing_graph = [
                    r for r in data['battery_results']
                    if r.get('status') in ('WARNING', 'FAIL')
                    and r.get('voltage_history')
                ]
                if sites_needing_graph:
                    html += '<h4>Voltage Trend (28-day) — Sites Requiring Attention:</h4>'
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
                        PAD_L, PAD_R = 60, 20
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
                                f'font-size="10" fill="#555">{tick_v:.2f}</text>'
                            )

                        x_labels = ''
                        step = max(1, n // 8)
                        for i in range(0, n, step):
                            lx = px_x(i)
                            x_labels += (
                                f'<text x="{lx:.1f}" y="{H - PAD_B + 14}" '
                                f'text-anchor="middle" font-size="9" fill="#555">{labels[i]}</text>'
                            )

                        thresh_svg = ''
                        if tw_val and isinstance(tw_val, (int, float)) and v_min <= tw_val <= v_max:
                            ty = threshold_y(tw_val)
                            thresh_svg += (
                                f'<line x1="{PAD_L}" y1="{ty:.1f}" x2="{W-PAD_R}" y2="{ty:.1f}" '
                                f'stroke="#ffc107" stroke-width="1.5" stroke-dasharray="5,3"/>'
                                f'<text x="{W-PAD_R+2}" y="{ty+4:.1f}" font-size="9" fill="#ffc107">warn</text>'
                            )
                        if tf_val and isinstance(tf_val, (int, float)) and v_min <= tf_val <= v_max:
                            ty = threshold_y(tf_val)
                            thresh_svg += (
                                f'<line x1="{PAD_L}" y1="{ty:.1f}" x2="{W-PAD_R}" y2="{ty:.1f}" '
                                f'stroke="#dc3545" stroke-width="1.5" stroke-dasharray="5,3"/>'
                                f'<text x="{W-PAD_R+2}" y="{ty+4:.1f}" font-size="9" fill="#dc3545">fail</text>'
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
                                          stroke="#667eea" stroke-width="2" stroke-linejoin="round"/>
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
                    html += f'<li>✓ {site}</li>'
                html += '</ul>'
            if data.get('disabled'):
                html += f'<h4>Disabled Sites ({len(data["disabled"])}):</h4><ul>'
                for site in data['disabled']:
                    html += f'<li>✓ {site}</li>'
                html += '</ul>'
        elif test_name == 'Depth Adjustment Verification':
            if 'calculations' in data:
                html += '<h4>Calculation Results:</h4>'
                html += '<table class="calculation-table">'
                html += '<tr><th>Test</th><th>Site</th><th>Raw (m)</th><th>BOM</th><th>Logger</th>'
                html += '<th>Adj?</th><th>Calc</th><th>Expected</th><th>Diff</th><th>✓</th></tr>'
                for calc in data['calculations']:
                    result   = '<span class="pass">✓</span>' if calc['passed'] else '<span class="fail">✗</span>'
                    adj_text = 'Yes' if calc['adjustment_applied'] else 'No'
                    html += f"""<tr>
                        <td>{calc['test_num']}</td>
                        <td>{calc['site']}</td>
                        <td>{calc['depth_raw']}</td>
                        <td>{calc['bom_baro']}</td>
                        <td>{calc['logger_baro']}</td>
                        <td>{adj_text}</td>
                        <td>{calc['calculated']}</td>
                        <td>{calc['expected']}</td>
                        <td>{calc['difference']}</td>
                        <td>{result}</td>
                    </tr>"""
                html += '</table>'
        elif test_name == 'Monthly Statistical Anomalies':
            pct_anomalies      = [a for a in data.get('depth_anomalies', []) if a.get('validity') == 'pct_change']
            validity_anomalies = [a for a in data.get('depth_anomalies', []) if a.get('validity') in ('fail', 'warning')]

            if pct_anomalies:
                html += '<h4>Depth % Change Anomalies:</h4><div class="anomaly">'
                html += f"<p>Max change: {data['depth_stats']['max_change_pct']}%</p>"
                html += (
                    '<table><tr><th>Site</th><th>Date</th>'
                    '<th>Previous (m)</th><th>New (m)</th>'
                    '<th>Change %</th><th>Rain Event (Y/N)</th></tr>'
                )
                for a in pct_anomalies:
                    rain_status = 'Yes' if a.get('rain_event', False) else 'No'
                    html += (
                        f"<tr>"
                        f"<td>{a['site']}</td>"
                        f"<td>{a['date']}</td>"
                        f"<td>{a.get('prev_value', 'N/A')}</td>"
                        f"<td>{a['value']}</td>"
                        f"<td>{a['change_pct']}%</td>"
                        f"<td>{rain_status}</td>"
                        f"</tr>"
                    )
                html += '</table></div>'

            if validity_anomalies:
                html += '<h4>Invalid Depth Readings:</h4><div class="anomaly">'
                html += '<table><tr><th>Site</th><th>Date</th><th>Depth (m)</th><th>Flag</th><th>Note</th></tr>'
                for a in validity_anomalies:
                    flag_class = 'fail' if a['validity'] == 'fail' else 'warning'
                    flag_label = 'FAIL' if a['validity'] == 'fail' else 'WARNING'
                    html += (
                        f"<tr><td>{a['site']}</td><td>{a['date']}</td><td>{a['value']}</td>"
                        f'<td class="{flag_class}">{flag_label}</td>'
                        f"<td>{a.get('note', '')}</td></tr>"
                    )
                html += '</table></div>'

            if data.get('pressure_anomalies'):
                html += '<h4>Pressure Anomalies:</h4><div class="anomaly">'
                html += f"<p>Max change: {data['pressure_stats']['max_change_pct']}%</p>"
                html += '<table><tr><th>Site</th><th>Date</th><th>Value (hPa)</th><th>Change %</th></tr>'
                for a in data['pressure_anomalies']:
                    html += f"<tr><td>{a['site']}</td><td>{a['date']}</td><td>{a['value']}</td><td>{a['change_pct']}%</td></tr>"
                html += '</table></div>'
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