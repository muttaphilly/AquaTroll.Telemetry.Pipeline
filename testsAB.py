"""
AquaTroll Depth Data Pipeline
A/B tests for the data scraping, validation & processing pipeline.
Monthly verification of script performance through testing of:
- Data connectivity and authentication
- Validation of returned data structures
- Battery health check
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
    "Weather Website Accessibility": (
        "Verifies if weather website accessible "
        "and returns received HTTP response code."
    ),
    "Logger Portal Authentication": (
        "Checks connectivity & provided authentication credentials allow successful "
        "login to the logger portal website."
    ),
    "Logger Portal Node Selection": (
        "Verifies ability to navigate and identify display structure in use (node or tables)"
    ),
    "Logger Portal CSV Download": (
        "Confirms the presence and functionality of CSV download button for data "
        "extraction."
    ),
    "Weather Data Structure": (
        "Verifies weather website accessibility and confirms the presence of pressure data "
        "at the expected positions (9 am at position 15, 3 pm at position 21)."
    ),
    "Logger Data Structure": (
        "Checks that downloaded logger CSV files contain required columns for "
        "Date, Time, and Level measurements."
    ),
    "Battery Voltage Check": (
        "Verifies battery voltage levels for each enabled site, flagging warnings if below "
        "thresholds (Logger: <3.5V, Starlink: <13.1V)."
    ),
    "Site Configuration": (
        "Checks site availability from SITES_CONFIG env variable, "
        "by reading the 'enabled' flag."
    ),
    "Network Availability Check": (
    "Locates last recorded depth reading for each enabled site. "
    "Use as a prompt to investigate whether pool has gone dry or equipment is faulty"
    ),
    "Depth Calculation Verification": (
        "Independently tests barometric pressure adjustments using downloaded files "
        "then cross-checks results against the pipeline's final csv results. "
        "Applies thresholds: depth > 0.3m and baro difference > 5 hPa to prevent "
        "corrections from very shallow/dry pools and sensor noise."
    ),
    "Statistical Anomaly Detection": (
        "Analyses validated data for statistical anomalies, flagging depth changes "
        ">15% and pressure changes >2% for review."
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
        <p>End of Test Report: AquaTroll Depth Data Pipeline</p>
    </div>
</body>
</html>
"""
# ============================================================================
# CALCULATION CONSTANTS
# ============================================================================
class CalculationConstants:
    """
    Constants used in depth calculation and pressure conversion.
    These values match logic used in main pipeline (dataValidation.py).
    """
    # Pressure conversion
    HPA_TO_PSI = 0.0145038
    PSI_TO_HPA = 68.9476
   
    # Depth adjustment (AquaTroll specifications)
    CONVERSION_FACTOR = 0.70307 # Pressure differential to water column height
   
    # Thresholds for adjustment application
    MIN_DEPTH_THRESHOLD = 0.3 # meters - prevents corrections in shallow pools
    MIN_BARO_DIFF_THRESHOLD = 5.0 # hPa - prevents corrections for sensor noise
    MAX_BARO_DIFF_THRESHOLD = 20.0 # hPa - prevents corrections for sensor malfunction
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
def filter_valid_depth_data(df, min_depth=0):
    """
    Filter dataframe for rows with valid depth measurements.
    Args:
        df (pd.DataFrame): Input dataframe
        min_depth (float): Minimum depth threshold
    Returns:
        pd.DataFrame: Filtered dataframe
    """
    return df[
        (df['Depth(m)raw'].notna()) &
        (df['Depth(m)raw'] > min_depth) &
        (df['Depth(m)adjusted'].notna()) &
        (df['Depth(m)adjusted'] > min_depth)
    ]
def should_apply_depth_adjustment(depth_m, bom_baro, logger_baro):
    """
    Determine if depth adjustment should be applied based on thresholds.
    Args:
        depth_m (float): Raw depth in meters
        bom_baro (float): BOM barometric pressure in hPa
        logger_baro (float): Logger barometric pressure in hPa
    Returns:
        bool: True if adjustment should be applied
    """
    if bom_baro is None or logger_baro is None:
        return False
    delta_p_hpa = bom_baro - logger_baro
    return (
        depth_m > CalculationConstants.MIN_DEPTH_THRESHOLD and
        delta_p_hpa > CalculationConstants.MIN_BARO_DIFF_THRESHOLD and
        delta_p_hpa <= CalculationConstants.MAX_BARO_DIFF_THRESHOLD
    )
def calculate_depth_adjustment(depth_m_raw, bom_baro, logger_baro):
    """
    Calculate adjusted depth using AquaTroll formula with thresholds.
    This implements the exact logic from dataValidation.py:
    1. Check if thresholds are met (depth > 0.3m, baro diff > 5 hPa)
    2. If thresholds met, apply adjustment formula
    3. Otherwise, return raw depth unchanged
    Args:
        depth_m_raw (float): Raw depth in meters
        bom_baro (float): BOM barometric pressure in hPa
        logger_baro (float): Logger barometric pressure in hPa
    Returns:
        tuple: (adjusted_depth_m, adjustment_applied)
    """
    # Check thresholds
    if not should_apply_depth_adjustment(depth_m_raw, bom_baro, logger_baro):
        return round(depth_m_raw, 2), False
    # Calculate adjustment
    delta_p_hpa = bom_baro - logger_baro
    delta_p_psi = delta_p_hpa * CalculationConstants.HPA_TO_PSI
    depth_adjustment_m = CalculationConstants.CONVERSION_FACTOR * delta_p_psi
    adjusted_depth = depth_m_raw + depth_adjustment_m
    return round(adjusted_depth, 2), True
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
   
    # ========================================================================
    # CONNECTIVITY TESTS
    # ========================================================================
    def test_01_weather_website_accessibility(self):
        """Test connectivity to weather website."""
        if not self.weather_url:
            self.record_result(
                "Weather Website Accessibility",
                "Connectivity",
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
                        "Weather Website Accessibility",
                        "Connectivity",
                        "PASS",
                        f"Successfully connected (HTTP {response.status_code}, {response_time:.2f}s)"
                    )
                else:
                    self.record_result(
                        "Weather Website Accessibility",
                        "Connectivity",
                        "WARNING",
                        f"Connected but table.data not found (HTTP {response.status_code})"
                    )
            else:
                self.record_result(
                    "Weather Website Accessibility",
                    "Connectivity",
                    "FAIL",
                    f"Weather website returned HTTP {response.status_code}"
                )
                self.fail(f"Weather website returned HTTP {response.status_code}")
        except requests.exceptions.Timeout:
            self.record_result(
                "Weather Website Accessibility",
                "Connectivity",
                "FAIL",
                "Connection timeout after 10 seconds"
            )
            self.fail("Connection timeout")
        except requests.exceptions.RequestException as e:
            self.record_result(
                "Weather Website Accessibility",
                "Connectivity",
                "FAIL",
                f"Connection error: {str(e)}"
            )
            self.fail(f"Connection error: {e}")
        except Exception as e:
            self.record_result(
                "Weather Website Accessibility",
                "Connectivity",
                "FAIL",
                f"Unexpected error: {str(e)}"
            )
            self.fail(f"Unexpected error: {e}")
    def test_02_logger_portal_authentication(self):
        """Test authentication to the logger portal using HTTP."""
        if not self.login_url:
            self.record_result(
                "Logger Portal Authentication",
                "Connectivity",
                "SKIP",
                "LOGIN_URL not configured in environment"
            )
            self.skipTest("LOGIN_URL not configured")
            return
        login_username = os.getenv("LOGIN_USERNAME", "")
        login_password = os.getenv("LOGIN_PASSWORD", "")
        if not login_username or not login_password:
            self.record_result(
                "Logger Portal Authentication",
                "Connectivity",
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
                    "Logger Portal Authentication",
                    "Connectivity",
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
                    "Logger Portal Authentication",
                    "Connectivity",
                    "FAIL",
                    "Authentication failed - credentials rejected (remained on login page)"
                )
                self.__class__.auth_session = None
                self.fail("Authentication failed")
                return
            # Success. Store session for other tests (use class variable!)
            self.__class__.auth_session = session
            self.record_result(
                "Logger Portal Authentication",
                "Connectivity",
                "PASS",
                f"Successfully authenticated to logger portal",
                {
                    'session_cookie': bool(session.cookies),
                    'final_url': login_response.url
                }
            )
        except requests.exceptions.Timeout:
            self.record_result(
                "Logger Portal Authentication",
                "Connectivity",
                "FAIL",
                "Connection timeout"
            )
            self.__class__.auth_session = None
            self.fail("Connection timeout")
        except requests.exceptions.RequestException as e:
            self.record_result(
                "Logger Portal Authentication",
                "Connectivity",
                "FAIL",
                f"Connection error: {str(e)}"
            )
            self.__class__.auth_session = None
            self.fail(f"Connection error: {e}")
        except Exception as e:
            self.record_result(
                "Logger Portal Authentication",
                "Connectivity",
                "FAIL",
                f"Authentication error: {str(e)}"
            )
            self.__class__.auth_session = None
            self.fail(f"Authentication error: {e}")
    def test_03_logger_portal_node_selection(self):
        """Test ability to navigate to and select nodes/tables in the logger portal."""
        if not hasattr(self.__class__, 'auth_session') or self.__class__.auth_session is None:
            self.record_result(
                "Logger Portal Node Selection",
                "Connectivity",
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
                "Logger Portal Node Selection",
                "Connectivity",
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
                "Logger Portal Node Selection",
                "Connectivity",
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
                    "Logger Portal Node Selection",
                    "Connectivity",
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
                    "Logger Portal Node Selection",
                    "Connectivity",
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
                    "Logger Portal Node Selection",
                    "Connectivity",
                    "FAIL",
                    f"Channel page missing expected elements (score: {validation_score}/2)"
                )
                self.fail("Channel page validation failed")
        except requests.exceptions.Timeout:
            self.record_result(
                "Logger Portal Node Selection",
                "Connectivity",
                "FAIL",
                "Navigation timeout"
            )
            self.__class__.channel_response = None
            self.fail("Navigation timeout")
        except requests.exceptions.RequestException as e:
            self.record_result(
                "Logger Portal Node Selection",
                "Connectivity",
                "FAIL",
                f"Navigation error: {str(e)}"
            )
            self.__class__.channel_response = None
            self.fail(f"Navigation error: {e}")
    def test_04_logger_portal_csv_download(self):
        """Test presence of CSV download functionality."""
        if not hasattr(self.__class__, 'channel_response') or self.__class__.channel_response is None:
            self.record_result(
                "Logger Portal CSV Download",
                "Connectivity",
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
                    "Logger Portal CSV Download",
                    "Connectivity",
                    "PASS",
                    f"CSV download capability verified: Table selector present and CSV/export button found"
                )
            elif has_csv_button and not has_channel_selector:
                self.record_result(
                    "Logger Portal CSV Download",
                    "Connectivity",
                    "WARNING",
                    f"CSV button found but no table selection mechanism detected"
                )
            elif has_channel_selector and not has_csv_button:
                self.record_result(
                    "Logger Portal CSV Download",
                    "Connectivity",
                    "WARNING",
                    f"Table selection present but CSV/export button not found"
                )
            else:
                self.record_result(
                    "Logger Portal CSV Download",
                    "Connectivity",
                    "FAIL",
                    f"Neither table selection nor CSV button found on channel page"
                )
                self.fail("CSV download functionality not available")
        except Exception as e:
            self.record_result(
                "Logger Portal CSV Download",
                "Connectivity",
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
                "Data Structure",
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
                    "Data Structure",
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
                    "Data Structure",
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
                    "Data Structure",
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
                    "Data Structure",
                    "FAIL",
                    "No valid data row found in weather table"
                )
                self.fail("No valid data row")
                return
            # Validate structure
            total_cells = len(first_valid_row)
            position_15_valid = False
            position_21_valid = False
            position_15_value = "N/A"
            position_21_value = "N/A"
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
            # Need both pressure columns to be valid
            if position_15_valid and position_21_valid:
                self.record_result(
                    "Weather Data Structure",
                    "Data Structure",
                    "PASS",
                    f"Weather table structure verified: {total_cells} columns, pressure data at positions 15 ({position_15_value} hPa) and 21 ({position_21_value} hPa)"
                )
            else:
                issues = []
                if not position_15_valid:
                    issues.append(f"Position 15 invalid (got: '{position_15_value}')")
                if not position_21_valid:
                    issues.append(f"Position 21 invalid (got: '{position_21_value}')")
                self.record_result(
                    "Weather Data Structure",
                    "Data Structure",
                    "FAIL",
                    f"Weather table pressure data validation failed: {', '.join(issues)}"
                )
                self.fail("Pressure data not in expected positions")
        except requests.exceptions.Timeout:
            self.record_result(
                "Weather Data Structure",
                "Data Structure",
                "FAIL",
                "Connection timeout accessing weather website"
            )
            self.fail("Connection timeout")
        except requests.exceptions.RequestException as e:
            self.record_result(
                "Weather Data Structure",
                "Data Structure",
                "FAIL",
                f"Connection error: {str(e)}"
            )
            self.fail(f"Connection error: {e}")
        except Exception as e:
            self.record_result(
                "Weather Data Structure",
                "Data Structure",
                "FAIL",
                f"Error parsing weather data: {str(e)}"
            )
            self.fail(f"Failed to parse weather data: {e}")
    def test_06_logger_data_structure(self):
        """Test logger CSV data structure."""
        if not os.path.exists(self.data_downloads_path):
            self.record_result(
                "Logger Data Structure",
                "Data Structure",
                "FAIL",
                "data_downloads directory not found"
            )
            self.fail("data_downloads directory not available - pipeline may not have run")
            return
        csv_files = [f for f in os.listdir(self.data_downloads_path) if f.endswith('.csv') and f != 'weather_data.csv' and 'baro' not in f.lower()]
        if not csv_files:
            self.record_result(
                "Logger Data Structure",
                "Data Structure",
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
            # Check base columns
            base_present = all(col in df.columns for col in base_columns)
            # Check at least one level column variant exists
            level_present = any(col in df.columns for col in level_column_options)
            if base_present and level_present:
                # Find which level column is present
                level_col = next((col for col in level_column_options if col in df.columns), None)
                self.record_result(
                    "Logger Data Structure",
                    "Data Structure",
                    "PASS",
                    f"Logger data has required columns: Date, Time, {level_col}"
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
                    "Data Structure",
                    "FAIL",
                    f"Missing required columns: {', '.join(missing)}"
                )
                self.fail("Required columns not found")
        except Exception as e:
            self.record_result(
                "Logger Data Structure",
                "Data Structure",
                "FAIL",
                f"Error reading logger data: {str(e)}"
            )
            self.fail(f"Failed to read logger data: {e}")
    def test_07_battery_voltage_check(self):
        """Check battery voltage for enabled sites."""
        if not os.path.exists(self.data_downloads_path):
            self.record_result(
                "Battery Voltage Check",
                "Battery Voltage",
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
                "Battery Voltage Check",
                "Battery Voltage",
                "SKIP",
                "No enabled sites configured"
            )
            self.skipTest("No enabled sites available")
            return
        battery_results = []
        critical_fail_count = 0 # Missing file or read error
        warning_count = 0 # Missing column, no data, or low voltage
        for site_id, config in enabled_sites.items():
            csv_path = os.path.join(self.data_downloads_path, f"{site_id}.csv")
            if not os.path.exists(csv_path):
                battery_results.append({
                    'site': site_id,
                    'status': 'FAIL',
                    'details': 'CSV file not found'
                })
                critical_fail_count += 1
                continue
            try:
                df = pd.read_csv(csv_path)
                battery_col = None
                for col in df.columns:
                    stripped_col = col.strip()
                    if stripped_col.startswith('Main Battery(MIN)[Main Buffer]'):
                        battery_col = col
                        break
                if battery_col is None:
                    battery_results.append({
                        'site': site_id,
                        'status': 'WARNING',
                        'details': 'Battery voltage column not found'
                    })
                    warning_count += 1
                    continue
                # Extract latest valid battery value
                battery_series = pd.to_numeric(df[battery_col], errors='coerce').dropna()
                if battery_series.empty:
                    battery_results.append({
                        'site': site_id,
                        'status': 'WARNING',
                        'details': 'No valid battery data'
                    })
                    warning_count += 1
                    continue
                current_voltage = battery_series.iloc[-1] # Most recent reading
                pressure_unit = config.get('pressure_unit', 'hpa').lower()
                threshold = 3.5 if pressure_unit == 'hpa' else 13.1
                if current_voltage < threshold:
                    status = 'WARNING'
                    warning_count += 1
                else:
                    status = 'PASS'
                battery_results.append({
                    'site': site_id,
                    'threshold': threshold,
                    'current_voltage': round(current_voltage, 2),
                    'status': status
                })
            except Exception as e:
                battery_results.append({
                    'site': site_id,
                    'status': 'FAIL',
                    'details': f"Error reading file: {str(e)}"
                })
                critical_fail_count += 1
        # Determine overall status and accurate summary
        if critical_fail_count > 0:
            overall_status = "FAIL"
        elif warning_count > 0:
            overall_status = "WARNING"
        else:
            overall_status = "PASS"
        details = (
            f"Checked {len(enabled_sites)} enabled sites: "
            f"{critical_fail_count} critical failures (no available data), "
            f"{warning_count} warnings (low voltage or missing data)"
        )
        self.record_result(
            "Battery Voltage Check",
            "Battery Voltage",
            overall_status,
            details,
            {'battery_results': battery_results}
        )
        if critical_fail_count > 0:
            self.fail(
                f"{critical_fail_count} site(s) had critical failures "
                "(missing CSV or read error)"
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
                    "Configuration",
                    "FAIL",
                    "No enabled sites found in SITES_CONFIG",
                    config_data
                )
                self.fail("No enabled sites configured")
            else:
                self.record_result(
                    "Site Configuration",
                    "Configuration",
                    "PASS",
                    f"Enabled: {len(enabled_sites)} sites; Disabled: {len(disabled_sites)} sites",
                    config_data
                )
        except Exception as e:
            self.record_result(
                "Site Configuration",
                "Configuration",
                "FAIL",
                f"Configuration error: {str(e)}"
            )
            self.fail(f"Site configuration test failed: {e}")
   
    # ========================================================================
    # CALCULATION TESTS
    # ========================================================================
    def test_09_depth_calculation_verification(self):
        """Validates result accuracy of depth calculations."""
        if not os.path.exists(self.validated_output_file):
            self.record_result(
                "Depth Calculation Verification",
                "Calculations",
                "SKIP",
                "No validated data found"
            )
            self.skipTest("Validated output file not available")
            return
        try:
            # Load validated data
            validated_df = load_csv_safely(self.validated_output_file)
            if validated_df is None:
                self.skipTest("Could not load validated output file")
                return
            validated_df = filter_valid_depth_data(validated_df, min_depth=0)
            if len(validated_df) == 0:
                self.record_result(
                    "Depth Calculation Verification",
                    "Calculations",
                    "SKIP",
                    "No valid depth data in validated output"
                )
                self.skipTest("No valid depth data available")
                return
            # Test random samples
            num_samples = min(3, len(validated_df))
            test_samples = validated_df.sample(n=num_samples, random_state=42)
            calculation_results = []
            all_passed = True
            for idx, row in test_samples.iterrows():
                site = row['Sample Point']
                date = row['Date Time (dd/mm/yyyy hh24:mi:ss)']
                depth_m_raw = float(row['Depth(m)raw'])
                depth_m_adjusted_expected = float(row['Depth(m)adjusted'])
                # Get barometric pressures
                bom_baro = None
                logger_baro = None
                if pd.notna(row.get('BomBaro', '')) and row.get('BomBaro', '') != '':
                    bom_baro = float(row['BomBaro'])
                if pd.notna(row.get('Barometric Pressure(RAW)[Main Buffer] (hPa)', '')) and \
                   row.get('Barometric Pressure(RAW)[Main Buffer] (hPa)', '') != '':
                    logger_baro = float(row['Barometric Pressure(RAW)[Main Buffer] (hPa)'])
                # Calculate adjusted depth
                calculated_adjusted_m, adjustment_applied = calculate_depth_adjustment(
                    depth_m_raw, bom_baro, logger_baro
                )
                # Compare with expected
                tolerance = 0.02 # 2cm tolerance
                passed = abs(calculated_adjusted_m - depth_m_adjusted_expected) < tolerance
                calculation_results.append({
                    'test_num': len(calculation_results) + 1,
                    'site': site,
                    'date': date,
                    'depth_raw': round(depth_m_raw, 2),
                    'bom_baro': round(bom_baro, 1) if bom_baro else 'N/A',
                    'logger_baro': round(logger_baro, 1) if logger_baro else 'N/A',
                    'adjustment_applied': adjustment_applied,
                    'calculated': calculated_adjusted_m,
                    'expected': round(depth_m_adjusted_expected, 2),
                    'difference': round(abs(calculated_adjusted_m - depth_m_adjusted_expected), 3),
                    'passed': passed
                })
                if not passed:
                    all_passed = False
            details = (
                f"Tested {len(calculation_results)} calculations. "
                f"Thresholds: depth > {CalculationConstants.MIN_DEPTH_THRESHOLD}m, "
                f"baro diff > {CalculationConstants.MIN_BARO_DIFF_THRESHOLD} hPa"
            )
            self.record_result(
                "Depth Calculation Verification",
                "Calculations",
                "PASS" if all_passed else "FAIL",
                details,
                {'calculations': calculation_results}
            )
            if not all_passed:
                self.fail("Some calculations did not match expected values")
        except Exception as e:
            self.record_result(
                "Depth Calculation Verification",
                "Calculations",
                "FAIL",
                f"Error during calculation verification: {str(e)}"
            )
            self.fail(f"Calculation verification failed: {e}")
   
    # ========================================================================
    # DATA QUALITY TESTS
    # ========================================================================
 
    def test_10_data_recency_check(self):
            """Check the most recent depth data reading for each enabled site."""
            if not os.path.exists(self.data_downloads_path):
                self.record_result(
                    "Network Availability Check",
                    "Data Quality",
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
                    "Network Availability Check",
                    "Data Quality",
                    "SKIP",
                    "No enabled sites configured"
                )
                self.skipTest("No enabled sites available")
                return
           
            recency_results = []
            current_date = datetime.now()
            fail_count = 0 # No CSV or empty CSV
            warning_count = 0 # Data > 28 days old
            pass_count = 0 # Data within 28 days
           
            for site_id, config in enabled_sites.items():
                csv_path = os.path.join(self.data_downloads_path, f"{site_id}.csv")
               
                # Check if CSV exists
                if not os.path.exists(csv_path):
                    recency_results.append({
                        'site': site_id,
                        'date': 'N/A',
                        'value': 'N/A',
                        'unit': 'N/A',
                        'days_since': '>28 days ago',
                        'status': 'FAIL'
                    })
                    fail_count += 1
                    continue
               
                try:
                    df = pd.read_csv(csv_path)
                   
                    # Check if CSV is empty
                    if df.empty:
                        recency_results.append({
                            'site': site_id,
                            'date': 'N/A',
                            'value': 'N/A',
                            'unit': 'N/A',
                            'days_since': '>28 days ago',
                            'status': 'FAIL'
                        })
                        fail_count += 1
                        continue
                   
                    # Detect level column and unit
                    level_column = None
                    unit = None
                   
                    # Check for metres
                    if 'Level in metres (m)' in df.columns:
                        level_column = 'Level in metres (m)'
                        unit = 'm'
                    # Check for feet
                    elif 'Level(RAW)[Main Buffer] (ft)' in df.columns:
                        level_column = 'Level(RAW)[Main Buffer] (ft)'
                        unit = 'ft'
                    # Fallback: any column with 'Level'
                    else:
                        for col in df.columns:
                            if 'Level' in col:
                                level_column = col
                                # Try to infer unit from column name
                                if '(m)' in col:
                                    unit = 'm'
                                elif '(ft)' in col:
                                    unit = 'ft'
                                else:
                                    unit = 'unknown'
                                break
                   
                    if level_column is None:
                        recency_results.append({
                            'site': site_id,
                            'date': 'N/A',
                            'value': 'N/A',
                            'unit': 'N/A',
                            'days_since': 'No level column found',
                            'status': 'FAIL'
                        })
                        fail_count += 1
                        continue
                   
                    # Check for Date column
                    if 'Date' not in df.columns:
                        recency_results.append({
                            'site': site_id,
                            'date': 'N/A',
                            'value': 'N/A',
                            'unit': unit,
                            'days_since': 'No date column found',
                            'status': 'FAIL'
                        })
                        fail_count += 1
                        continue
                   
                    # Parse dates and find most recent with valid depth data
                    df['parsed_date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
                    df[level_column] = pd.to_numeric(df[level_column], errors='coerce')
                   
                    # Filter for valid dates and depths
                    valid_data = df[df['parsed_date'].notna() & df[level_column].notna()]
                   
                    if valid_data.empty:
                        recency_results.append({
                            'site': site_id,
                            'date': 'N/A',
                            'value': 'N/A',
                            'unit': unit,
                            'days_since': '>28 days ago',
                            'status': 'FAIL'
                        })
                        fail_count += 1
                        continue
                   
                    # Get most recent entry
                    most_recent_idx = valid_data['parsed_date'].idxmax()
                    most_recent_date = valid_data.loc[most_recent_idx, 'parsed_date']
                    most_recent_value = valid_data.loc[most_recent_idx, level_column]
                   
                    # Calculate days since
                    days_since = (current_date - most_recent_date).days
                   
                    # Format date for display
                    date_str = most_recent_date.strftime('%d/%m/%Y')
                   
                    # Determine status
                    if days_since <= 28:
                        status = 'PASS'
                        pass_count += 1
                    else:
                        status = 'WARNING'
                        warning_count += 1
                   
                    recency_results.append({
                        'site': site_id,
                        'date': date_str,
                        'value': round(most_recent_value, 2),
                        'unit': unit,
                        'days_since': f"{days_since} days ago",
                        'status': status
                    })
                   
                except Exception as e:
                    recency_results.append({
                        'site': site_id,
                        'date': 'N/A',
                        'value': 'N/A',
                        'unit': 'N/A',
                        'days_since': f'Error: {str(e)}',
                        'status': 'FAIL'
                    })
                    fail_count += 1
           
            # Determine overall test status
            if fail_count > 0:
                overall_status = "FAIL"
            elif warning_count > 0:
                overall_status = "WARNING"
            else:
                overall_status = "PASS"
           
            details = (
                f"Checked {len(enabled_sites)} enabled sites: "
                f"{pass_count} within 28 days, "
                f"{warning_count} stale (>28 days), "
                f"{fail_count} missing/empty data"
            )
           
            self.record_result(
                "Network Availability Check",
                "Data Quality",
                overall_status,
                details,
                {'recency_results': recency_results}
            )
           
            if fail_count > 0:
                self.fail(
                    f"{fail_count} site(s) had no data or missing CSV files"
                )
   
    def test_11_statistical_anomaly_detection(self):
        """Test for statistical anomalies in validated data."""
        if not os.path.exists(self.validated_output_file):
            self.record_result(
                "Statistical Anomaly Detection",
                "Data Quality",
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
            # Convert to numeric
            df['Depth(m)adjusted'] = pd.to_numeric(df['Depth(m)adjusted'], errors='coerce')
            df['BomBaro'] = pd.to_numeric(df['BomBaro'], errors='coerce')
            df = df.dropna(subset=['Sample Point'])
            depth_anomalies = []
            pressure_anomalies = []
            # Analyse by site
            for site in df['Sample Point'].unique():
                site_data = df[df['Sample Point'] == site].copy()
                site_data = site_data.sort_values('Date Time (dd/mm/yyyy hh24:mi:ss)')
                # Depth anomalies (>15% change)
                if 'Depth(m)adjusted' in site_data.columns:
                    site_data['depth_pct_change'] = site_data['Depth(m)adjusted'].pct_change() * 100
                    depth_mask = abs(site_data['depth_pct_change']) > 15
                    for idx in site_data[depth_mask].index:
                        depth_anomalies.append({
                            'site': site,
                            'date': str(site_data.loc[idx, 'Date Time (dd/mm/yyyy hh24:mi:ss)']),
                            'value': round(site_data.loc[idx, 'Depth(m)adjusted'], 1),
                            'change_pct': round(site_data.loc[idx, 'depth_pct_change'], 1)
                        })
                # Pressure anomalies (>2% change)
                if 'BomBaro' in site_data.columns:
                    site_data['pressure_pct_change'] = site_data['BomBaro'].pct_change() * 100
                    pressure_mask = abs(site_data['pressure_pct_change']) > 2
                    for idx in site_data[pressure_mask].index:
                        if pd.notna(site_data.loc[idx, 'BomBaro']):
                            pressure_anomalies.append({
                                'site': site,
                                'date': str(site_data.loc[idx, 'Date Time (dd/mm/yyyy hh24:mi:ss)']),
                                'value': round(site_data.loc[idx, 'BomBaro'], 1),
                                'change_pct': round(site_data.loc[idx, 'pressure_pct_change'], 1)
                            })
            # Calculate stats
            depth_stats = {
                'max_change_pct': round(max([abs(a['change_pct']) for a in depth_anomalies], default=0), 1),
                'count': len(depth_anomalies)
            }
            pressure_stats = {
                'max_change_pct': round(max([abs(a['change_pct']) for a in pressure_anomalies], default=0), 1),
                'count': len(pressure_anomalies)
            }
            # Limit results
            depth_anomalies.sort(key=lambda x: abs(x['change_pct']), reverse=True)
            pressure_anomalies.sort(key=lambda x: abs(x['change_pct']), reverse=True)
            depth_anomalies = depth_anomalies[:10]
            pressure_anomalies = pressure_anomalies[:10]
            status = "WARNING" if (len(depth_anomalies) > 0 or len(pressure_anomalies) > 0) else "PASS"
            self.record_result(
                "Statistical Anomaly Detection",
                "Data Quality",
                status,
                f"Depth anomalies: {depth_stats['count']} (>15%), Pressure anomalies: {pressure_stats['count']} (>2%)",
                {
                    'depth_anomalies': depth_anomalies,
                    'pressure_anomalies': pressure_anomalies,
                    'depth_stats': depth_stats,
                    'pressure_stats': pressure_stats
                }
            )
        except Exception as e:
            self.record_result(
                "Statistical Anomaly Detection",
                "Data Quality",
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
        # Category order
        category_order = [
            'Connectivity',
            'Data Structure',
            'Battery Voltage',
            'Configuration',
            'Calculations',
            'Data Quality'
        ]
        # Generate sections
        for category in category_order:
            if category in categories:
                tests = categories[category]
                html += f'<h2>{category} Tests</h2>'
                for test in tests:
                    status_class = test['status'].lower()
                    test_description = TEST_DESCRIPTIONS.get(test['test_name'], '')
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
        if test_name == 'Network Availability Check':
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
        elif test_name == 'Logger Portal Authentication':
            if data:
                html += '<h4>Authentication Success:</h4><ul>'
                if 'session_cookie' in data:
                    html += f'<li>Session Cookie: {data["session_cookie"]}</li>'
                if 'final_url' in data:
                    html += f'<li>Final URL: {data["final_url"]}</li>'
                html += '</ul>'
        elif test_name == 'Logger Portal Node Selection':
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
        elif test_name == 'Battery Voltage Check':
            if 'battery_results' in data:
                html += '<h4>Battery Voltage Results:</h4>'
                html += '<table class="calculation-table">'
                html += '<tr><th>Site</th><th>Threshold (V)</th><th>Current Voltage (V)</th><th>Status</th></tr>'
                for res in data['battery_results']:
                    status_class = res['status'].lower()
                    details = res.get('details', '')
                    current_voltage = res.get('current_voltage', 'N/A')
                    threshold = res.get('threshold', 'N/A')
                    html += f"""<tr>
                        <td>{res['site']}</td>
                        <td>{threshold}</td>
                        <td>{current_voltage if details == '' else details}</td>
                        <td class="{status_class}">{res['status']}</td>
                    </tr>"""
                html += '</table>'
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
        elif test_name == 'Depth Calculation Verification':
            if 'calculations' in data:
                html += '<h4>Calculation Results:</h4>'
                html += '<table class="calculation-table">'
                html += '<tr><th>Test</th><th>Site</th><th>Raw (m)</th><th>BOM</th><th>Logger</th>'
                html += '<th>Adj?</th><th>Calc</th><th>Expected</th><th>Diff</th><th>✓</th></tr>'
                for calc in data['calculations']:
                    result = '<span class="pass">✓</span>' if calc['passed'] else '<span class="fail">✗</span>'
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
        elif test_name == 'Statistical Anomaly Detection':
            if data.get('depth_anomalies'):
                html += '<h4>Depth Anomalies:</h4><div class="anomaly">'
                html += f"<p>Max change: {data['depth_stats']['max_change_pct']}%</p>"
                html += '<table><tr><th>Site</th><th>Date</th><th>Value</th><th>Change %</th></tr>'
                for a in data['depth_anomalies']:
                    html += f"<tr><td>{a['site']}</td><td>{a['date']}</td><td>{a['value']}</td><td>{a['change_pct']}%</td></tr>"
                html += '</table></div>'
            if data.get('pressure_anomalies'):
                html += '<h4>Pressure Anomalies:</h4><div class="anomaly">'
                html += f"<p>Max change: {data['pressure_stats']['max_change_pct']}%</p>"
                html += '<table><tr><th>Site</th><th>Date</th><th>Value</th><th>Change %</th></tr>'
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
