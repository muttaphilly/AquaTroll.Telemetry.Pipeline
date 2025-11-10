"""
AquaTroll Depth Data Pipeline
A/B tests for the data scraping, validation & processing pipeline.

Monthly verification of script performance through testing of:
- Data connectivity and authentication
- Validation or returned data structures
- Verification of depth calculations
- Site availability checks
- Flagging of any statistical anomalies

Author: Philip Curry
Last Modified: 10-11-2025
"""

# ============================================================================
# IMPORTS
# ============================================================================

# Standard Library
import json
import logging
import os
import sys
import unittest
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

# ID the system and user running the script
COMPUTER_NAME = gethostname()
USERNAME = getuser()

# ============================================================================
# TEST DESCRIPTIONS
# ============================================================================

TEST_DESCRIPTIONS = {
    "Weather Website Accessibility": (
        "Verifies if weather website accessible "
        "and returns the HTTP response code."
    ),
    "Logger Portal Authentication": (
        "Tests connectivity to the logger data portal and validates authentication "
    ),
    "Weather Data Structure": (
        "Validates that scraped weather data contains the expected column structure, "
    ),
    "Logger Data Structure": (
        "Checks that downloaded logger CSV files contain required columns for "
        "Date, Time, and Level measurements."
    ),
    "Site Configuration": (
        "Verifies site availability from SITES_CONFIG env variable, "
        "checking which sites are enabled vs disabled based on the 'enabled' flag."
    ),
    "Depth Calculation Verification": (
        "Indpendently tests barometric pressure adjustments using downloaded files "
        "then cross-checks results against the pipelines final validated csv results. "
        "Applies thresholds: depth > 0.3m and baro difference > 5 hPa to prevent "
        "corrections from very shallow/dry pools and sensor noise."
    ),
    "PSI to hPa Conversion": (
        "Validates PSI to hPa conversion accuracy (factor: 68.9476) by comparing "
        "calculated values against validated output data from any PSI-based logger sites."
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
        <h1>🔬 Environmental Data Pipeline Test Report</h1>
        <p>Generated: {timestamp} on {computer_name} by {username}</p>
    </div>
    <img src="images/gorgeMonitoring.jpg" alt="Gorge Monitoring Banner" class="banner" onerror="this.style.display='none'">
"""

HTML_FOOTER = """
    <div class="footer">
        <p>End of Test Report - AquaTroll Environmental Data Pipeline</p>
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
    These values match those in dataValidation.py for consistency.
    """
    # Pressure conversion
    HPA_TO_PSI = 0.0145038
    PSI_TO_HPA = 68.9476
    
    # Depth adjustment (AquaTroll specifications)
    CONVERSION_FACTOR = 0.70307  # Pressure differential to water column height
    
    # Thresholds for adjustment application (adjiusts for observed localised variations)
    MIN_DEPTH_THRESHOLD = 0.3  # meters - prevents corrections in shallow pools
    MIN_BARO_DIFF_THRESHOLD = 5.0  # hPa - prevents corrections for sensor noise
    
    # Water properties
    WATER_DENSITY = 1000.0  # kg/m³
    REFERENCE_DENSITY = 1000.0  # kg/m³
    
    @classmethod
    def specific_gravity(cls):
        """Calculate specific gravity for fresh water."""
        return cls.WATER_DENSITY / cls.REFERENCE_DENSITY

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
    
    delta_p_hpa = abs(logger_baro - bom_baro)
    
    return (
        depth_m > CalculationConstants.MIN_DEPTH_THRESHOLD and
        delta_p_hpa > CalculationConstants.MIN_BARO_DIFF_THRESHOLD
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
    delta_p_hpa = logger_baro - bom_baro
    delta_p_psi = delta_p_hpa * CalculationConstants.HPA_TO_PSI
    depth_adjustment_m = (
        CalculationConstants.CONVERSION_FACTOR * 
        delta_p_psi / 
        CalculationConstants.specific_gravity()
    )
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
        
        # Suppress logging during tests
        logging.getLogger().setLevel(logging.CRITICAL)
        
        # Load configurations from env
        cls.weather_url = os.getenv("WEATHER_URL", "")
        cls.login_url = os.getenv("LOGIN_URL", "")
        
        # Load logger sites
        cls.site_config = json.loads(os.getenv("SITES_CONFIG", "{}"))
        
        # Define data paths
        cls.data_downloads_path = 'data_downloads'
        cls.transformed_data_path = 'transformed_data'
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
    
    def test_weather_website_accessibility(self):
        """Verify weather website is accessible."""
        if not self.weather_url:
            self.record_result(
                "Weather Website Accessibility",
                "Connectivity",
                "FAIL",
                "WEATHER_URL not configured in environment"
            )
            self.fail("WEATHER_URL not configured")
        
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
                        f"Successfully connected (Status: {response.status_code}, Response time: {response_time:.2f}s, Table found: YES)"
                    )
                    self.assertTrue(True)
                else:
                    self.record_result(
                        "Weather Website Accessibility",
                        "Connectivity",
                        "WARNING",
                        f"Connected but table.data not found (Status: {response.status_code})"
                    )
                    self.assertTrue(True)
            else:
                self.record_result(
                    "Weather Website Accessibility",
                    "Connectivity",
                    "FAIL",
                    f"Connection failed (Status: {response.status_code})"
                )
                self.fail(f"Weather site returned status {response.status_code}")
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
            self.fail(f"Could not connect to weather site: {e}")
        except Exception as e:
            self.record_result(
                "Weather Website Accessibility",
                "Connectivity",
                "FAIL",
                f"Unexpected error: {str(e)}"
            )
            self.fail(f"Unexpected error: {e}")
    
    def test_logger_portal_authentication(self):
        """Verify logger portal authentication endpoint is accessible."""
        if not self.login_url:
            self.record_result(
                "Logger Portal Authentication",
                "Connectivity",
                "SKIP",
                "LOGIN_URL not configured in environment"
            )
            self.skipTest("LOGIN_URL not configured")
            return
        
        # Get credentials from environment
        login_username = os.getenv("LOGIN_USERNAME", "")
        login_password = os.getenv("LOGIN_PASSWORD", "")
        
        if not login_username or not login_password:
            self.record_result(
                "Logger Portal Authentication",
                "Connectivity",
                "FAIL",
                "LOGIN_USERNAME or LOGIN_PASSWORD not configured in environment"
            )
            self.fail("Login credentials not configured")
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        try:
            # Create session to handle cookies and redirects
            session = requests.Session()
            
            # First, verify the login page is accessible
            start_time = datetime.now()
            response = session.get(self.login_url, headers=headers, timeout=10)
            response_time = (datetime.now() - start_time).total_seconds()
            
            if response.status_code != 200:
                self.record_result(
                    "Logger Portal Authentication",
                    "Connectivity",
                    "FAIL",
                    f"Login page not accessible (Status: {response.status_code})"
                )
                self.fail(f"Login page returned status {response.status_code}")
            
            # Parse the login form to get any hidden fields (ASP.NET ViewState, etc.)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Build form data with all hidden fields plus credentials
            form_data = {}
            
            # Get all hidden input fields (ASP.NET requires these)
            for hidden_input in soup.find_all('input', type='hidden'):
                name = hidden_input.get('name')
                value = hidden_input.get('value', '')
                if name:
                    form_data[name] = value
            
            # Add login credentials
            form_data['txtUsername'] = login_username
            form_data['txtPassword'] = login_password
            
            # Also look for submit button value
            submit_button = soup.find('input', type='submit')
            if submit_button and submit_button.get('name'):
                form_data[submit_button.get('name')] = submit_button.get('value', '')
            
            # Submit login form
            auth_start_time = datetime.now()
            login_response = session.post(
                self.login_url,
                data=form_data,
                headers=headers,
                timeout=15,
                allow_redirects=True
            )
            auth_time = (datetime.now() - auth_start_time).total_seconds()
            
            # Check authenticated page reached (i.e. not still on login page)
            final_url = login_response.url
            
            if 'logon.aspx' in final_url.lower():
                # Still on login page = authentication failed
                self.record_result(
                    "Logger Portal Authentication",
                    "Connectivity",
                    "FAIL",
                    f"Authentication failed - credentials rejected (remained on login page)"
                )
                self.fail("Authentication failed - invalid credentials")
            else:
                # Redirected to any page other than login = authentication successful 
                self.record_result(
                    "Logger Portal Authentication",
                    "Connectivity",
                    "PASS",
                    f"Successfully authenticated and redirected to portal (Response time: {response_time:.2f}s, Auth time: {auth_time:.2f}s, Landing page: {final_url})"
                )
                self.assertTrue(True)
            
        except requests.exceptions.Timeout:
            self.record_result(
                "Logger Portal Authentication",
                "Connectivity",
                "FAIL",
                "Connection timeout - portal not responding"
            )
            self.fail("Connection timeout")
        except requests.exceptions.RequestException as e:
            self.record_result(
                "Logger Portal Authentication",
                "Connectivity",
                "FAIL",
                f"Connection error: {str(e)}"
            )
            self.fail(f"Could not connect to logger portal: {e}")
        except Exception as e:
            self.record_result(
                "Logger Portal Authentication",
                "Connectivity",
                "FAIL",
                f"Authentication error: {str(e)}"
            )
            self.fail(f"Portal authentication test failed: {e}")
       
    # ========================================================================
    # DATA STRUCTURE TESTS
    # ========================================================================
    
    def test_weather_data_structure(self):
        """Validate weather data contains expected column structure."""
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
        
        # Required positions matching weatherStation.py scraping logic
        required_positions = {
            'MSL Pressure (9am) [hPa]': 15,  # Morning hPa (hpa_am_index)
            'MSL Pressure (3pm) [hPa]': 21   # Afternoon hPa (hpa_pm_index)
        }
        
        try:
            response = requests.get(self.weather_url, headers=headers, timeout=10)
            
            if response.status_code != 200:
                self.record_result(
                    "Weather Data Structure",
                    "Data Structure",
                    "FAIL",
                    f"Could not retrieve webpage (Status: {response.status_code})",
                    {
                        'all_columns': [],
                        'required_columns': required_positions,
                        'issues': [f"HTTP {response.status_code}"]
                    }
                )
                self.fail(f"Weather site returned status {response.status_code}")
            
            soup = BeautifulSoup(response.text, 'html.parser')
            table = soup.select_one('table.data')
            
            if not table:
                self.record_result(
                    "Weather Data Structure",
                    "Data Structure",
                    "FAIL",
                    "Table with class 'data' not found in HTML",
                    {
                        'all_columns': [],
                        'required_columns': required_positions,
                        'issues': ["Table not found"]
                    }
                )
                self.fail("Could not parse weather table structure")
            
            # Weather site uses colspan/rowspan for group headers
            # Can't parse thead, analyse actual data row structure
            
            data_rows = table.select('tbody tr')
            first_valid_row = None
            headers_list = []
            
            # Find first valid data row (skip summary rows)
            for row in data_rows:
                cells = row.select('th, td')
                if cells and len(cells) > 1:
                    date_cell = cells[0].get_text(strip=True)
                    if date_cell.isdigit():  # Valid day number
                        first_valid_row = cells
                        break
            
            if not first_valid_row:
                self.record_result(
                    "Weather Data Structure",
                    "Data Structure",
                    "FAIL",
                    "Could not find valid data row in table",
                    {
                        'all_columns': [],
                        'required_columns': required_positions,
                        'issues': ["No valid data rows found"]
                    }
                )
                self.fail("Could not find valid data row in table")
            
            # Build column descriptions based on returned data structure
            total_columns = len(first_valid_row)
            
            # Headers for display in report
            # which match the weather sites structure
            column_descriptions = [
                'Date (Day)',
                'Day of Week',
                'Min Temp (°C)',
                'Max Temp (°C)',
                'Rain (mm)',
                'Evaporation (mm)',
                'Sunshine (hours)',
                'Max Gust Direction',
                'Max Gust Speed (km/h)',
                'Max Gust Time',
                'Temp 9am (°C)',
                'Relative Humidity 9am (%)',
                'Cloud 9am (oktas)',
                'Wind Direction 9am',
                'Wind Speed 9am (km/h)',
                'MSL Pressure 9am (hPa)',  # Position 15
                'Temp 3pm (°C)',
                'Relative Humidity 3pm (%)',
                'Cloud 3pm (oktas)',
                'Wind Direction 3pm',
                'Wind Speed 3pm (km/h)',
                'MSL Pressure 3pm (hPa)'   # Position 21
            ]
            
            # Use actual column count, but provide descriptions for display
            if total_columns <= len(column_descriptions):
                headers_list = column_descriptions[:total_columns]
            else:
                # More columns than expected, add generic names
                headers_list = column_descriptions + [f'Column_{i}' for i in range(len(column_descriptions), total_columns)]
            
            # Verify positions 15 and 21 exist and contain pressure data
            position_15_ok = False
            position_21_ok = False
            
            if total_columns > 15:
                # Extract value at position 15 from first valid row
                val_15 = first_valid_row[15].get_text(strip=True)
                try:
                    # Check numeric value in reasonable hPa range (950-1050)
                    float_val = float(val_15)
                    position_15_ok = 950 <= float_val <= 1050
                except (ValueError, TypeError):
                    position_15_ok = False
            
            if total_columns > 21:
                # Extract value at position 21 from first valid row
                val_21 = first_valid_row[21].get_text(strip=True)
                try:
                    # Check numeric value 
                    float_val = float(val_21)
                    position_21_ok = 950 <= float_val <= 1050
                except (ValueError, TypeError):
                    position_21_ok = False
            
            test_passed = position_15_ok and position_21_ok
            
            if test_passed:
                # Show sample values to user
                sample_val_15 = first_valid_row[15].get_text(strip=True)
                sample_val_21 = first_valid_row[21].get_text(strip=True)
                
                self.record_result(
                    "Weather Data Structure",
                    "Data Structure",
                    "PASS",
                    f"Found {total_columns} columns. Required pressure columns at positions 15 and 21. Sample values: [15]='{sample_val_15}' hPa, [21]='{sample_val_21}' hPa",
                    {
                        'all_columns': headers_list,
                        'required_columns': required_positions
                    }
                )
            else:
                missing = []
                if not position_15_ok:
                    if total_columns > 15:
                        val = first_valid_row[15].get_text(strip=True)
                        missing.append(f"Pressure data at position 15 (found: '{val}', not valid hPa)")
                    else:
                        missing.append(f"Position 15 does not exist (only {total_columns} columns)")
                
                if not position_21_ok:
                    if total_columns > 21:
                        val = first_valid_row[21].get_text(strip=True)
                        missing.append(f"Pressure data at position 21 (found: '{val}', not valid hPa)")
                    else:
                        missing.append(f"Position 21 does not exist (only {total_columns} columns)")
                
                self.record_result(
                    "Weather Data Structure",
                    "Data Structure",
                    "FAIL",
                    f"Missing required columns: {', '.join(missing)}",
                    {
                        'all_columns': headers_list,
                        'required_columns': required_positions,
                        'issues': missing
                    }
                )
            
            self.assertTrue(test_passed, f"Weather table structure validation failed: {missing if not test_passed else ''}")
            
        except requests.exceptions.RequestException as e:
            self.record_result(
                "Weather Data Structure",
                "Data Structure",
                "FAIL",
                f"Connection error: {str(e)}",
                {
                    'all_columns': [],
                    'required_columns': required_positions,
                    'issues': [str(e)]
                }
            )
            self.fail(f"Could not connect to weather site: {e}")
        except Exception as e:
            self.record_result(
                "Weather Data Structure",
                "Data Structure",
                "FAIL",
                f"Parsing error: {str(e)}",
                {
                    'all_columns': [],
                    'required_columns': required_positions,
                    'issues': [str(e)]
                }
            )
            self.fail(f"Could not parse weather table structure: {e}")
    
    def test_logger_data_structure(self):
        """Verify logger CSV files have required column structure."""
        if not os.path.exists(self.data_downloads_path):
            self.record_result(
                "Logger Data Structure",
                "Data Structure",
                "SKIP",
                "data_downloads directory not found - skipping test"
            )
            self.skipTest("data_downloads directory not available")
            return
        
        csv_files = [f for f in os.listdir(self.data_downloads_path) if f.endswith('.csv')]
        
        if not csv_files:
            self.record_result(
                "Logger Data Structure",
                "Data Structure",
                "SKIP",
                "No CSV files found in data_downloads - skipping test"
            )
            self.skipTest("No CSV files available for testing")
            return
        
        test_file = os.path.join(self.data_downloads_path, csv_files[0])
        
        try:
            df = pd.read_csv(test_file)
            required_columns = ['Date', 'Time', 'Level(RAW)[Main Buffer] (ft)']
            
            columns_present = all(col in df.columns for col in required_columns)
            
            if columns_present:
                self.record_result(
                    "Logger Data Structure",
                    "Data Structure",
                    "PASS",
                    f"Expected columns verified: {', '.join(required_columns)}"
                )
                self.assertTrue(True)
            else:
                missing = [col for col in required_columns if col not in df.columns]
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
                f"Error reading CSV: {str(e)}"
            )
            self.fail(f"Failed to read logger data: {e}")
    
    # ========================================================================
    # CONFIGURATION TESTS
    # ========================================================================
    
    def test_site_configuration(self):
        """Validate site configuration from SITES_CONFIG."""
        try:
            enabled_sites = []
            disabled_sites = []
            
            # Parse enabled flag from SITES_CONFIG
            for site_name, config in self.site_config.items():
                is_enabled = config.get('enabled', False)
                if is_enabled:
                    enabled_sites.append(site_name)
                else:
                    disabled_sites.append(site_name)
            
            config_data = {
                'enabled': sorted(enabled_sites),
                'disabled': sorted(disabled_sites),
                'missing': []
            }
            
            if len(enabled_sites) == 0:
                self.record_result(
                    "Site Configuration",
                    "Logger Availability",
                    "FAIL",
                    "No enabled sites found in SITES_CONFIG",
                    config_data
                )
                self.fail("No enabled sites configured")
            else:
                self.record_result(
                    "Site Configuration",
                    "Logger Availability",
                    "PASS",
                    f"Found {len(enabled_sites)} enabled sites, {len(disabled_sites)} disabled sites from SITES_CONFIG",
                    config_data
                )
                self.assertTrue(True)
            
        except Exception as e:
            self.record_result(
                "Site Configuration",
                "Logger Availability",
                "FAIL",
                f"Configuration error: {str(e)}"
            )
            self.fail(f"Site configuration test failed: {e}")
    
    # ========================================================================
    # CALCULATION TESTS
    # ========================================================================
    
    def test_depth_calculation_verification(self):
        """
        Verify depth calculation accuracy using scraped data.
        
        Tests dataValidation.py formula:
        - Site-specific depth conversions
        - Any threshold applications (depth > 0.3m, baro diff > 5 hPa)
        - AquaTroll pressure-to-depth conversion
        - Cross-check against the outputted validated data
        """
        # Check required files exist
        if not os.path.exists(self.data_downloads_path):
            self.record_result(
                "Depth Calculation Verification",
                "Calculations",
                "SKIP",
                "data_downloads directory not found - skipping test"
            )
            self.skipTest("data_downloads directory not available")
            return
        
        if not os.path.exists(self.validated_output_file):
            self.record_result(
                "Depth Calculation Verification",
                "Calculations",
                "SKIP",
                "validatedDepthData.csv not found - skipping test"
            )
            self.skipTest("Validated output file not available")
            return
        
        try:
            # Load and filter validated data
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
                    "No valid depth data found in validated output - skipping test"
                )
                self.skipTest("No valid depth data available")
                return
            
            # Select random test samples from downloaded data
            num_samples = min(3, len(validated_df))
            test_samples = validated_df.sample(n=num_samples, random_state=42)
            
            calculation_results = []
            all_passed = True
            
            for idx, row in test_samples.iterrows():
                site = row['Sample Point']
                date = row['Date Time (dd/mm/yyyy hh24:mi:ss)']
                
                # Extract values
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
                
                # Calculate adjusted depth using helper function
                calculated_adjusted_m, adjustment_applied = calculate_depth_adjustment(
                    depth_m_raw, bom_baro, logger_baro
                )
                
                # Compare with expected
                tolerance = 0.02  # 2cm tolerance
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
            
            if len(calculation_results) == 0:
                self.record_result(
                    "Depth Calculation Verification",
                    "Calculations",
                    "SKIP",
                    "Could not extract valid test data - skipping test"
                )
                self.skipTest("No valid test data available")
                return
            
            details = (
                f"Tested {len(calculation_results)} calculations using dataValidation.py formula with thresholds: "
                f"depth > {CalculationConstants.MIN_DEPTH_THRESHOLD}m, "
                f"baro diff > {CalculationConstants.MIN_BARO_DIFF_THRESHOLD} hPa. "
                f"Formula: adjusted = raw + (CONVERSION_FACTOR * delta_p_psi / SG)"
            )
            
            self.record_result(
                "Depth Calculation Verification",
                "Calculations",
                "PASS" if all_passed else "FAIL",
                details,
                {'calculations': calculation_results}
            )
            
            self.assertTrue(all_passed, "Some calculations did not match expected values")
            
        except Exception as e:
            self.record_result(
                "Depth Calculation Verification",
                "Calculations",
                "FAIL",
                f"Error during calculation verification: {str(e)}"
            )
            self.fail(f"Calculation verification failed: {e}")
    
    def test_psi_to_hpa_conversion(self):
        """
        Validate PSI to hPa conversion accuracy.
        
        Compares calculated values (PSI × 68.9476) against validated
        output data from PSI-based logger sites.
        """
        # Check required files
        if not os.path.exists(self.validated_output_file):
            self.record_result(
                "PSI to hPa Conversion",
                "Calculations",
                "SKIP",
                "validatedDepthData.csv not found - skipping test"
            )
            self.skipTest("Validated output file not available")
            return
        
        try:
            # Load validated output
            validated_df = load_csv_safely(self.validated_output_file)
            if validated_df is None:
                self.skipTest("Could not load validated output file")
                return
            
            # Filter for rows with PSI data
            psi_col = 'Pressure(RAW)[Main Buffer] (PSI) - Original'
            hpa_col = 'Barometric Pressure(RAW)[Main Buffer] (hPa)'
            
            validated_df = validated_df[
                (validated_df[psi_col].notna()) &
                (validated_df[psi_col] > 0) &
                (validated_df[hpa_col].notna()) &
                (validated_df[hpa_col] > 0)
            ]
            
            if len(validated_df) == 0:
                self.record_result(
                    "PSI to hPa Conversion",
                    "Calculations",
                    "SKIP",
                    "No PSI data found in validated output - skipping test"
                )
                self.skipTest("No PSI data available for testing")
                return
            
            # Select random test samples
            num_samples = min(3, len(validated_df))
            test_samples = validated_df.sample(n=num_samples, random_state=42)
            
            conversion_results = []
            all_passed = True
            
            for idx, row in test_samples.iterrows():
                site = row['Sample Point']
                date = row['Date Time (dd/mm/yyyy hh24:mi:ss)']
                psi_value = float(row[psi_col])
                expected_hpa = float(row[hpa_col])
                
                # Calculate using conversion factor
                calculated_hpa = psi_value * CalculationConstants.PSI_TO_HPA
                
                # Check with tolerance
                tolerance = 1.0  # 1 hPa tolerance for rounding
                passed = abs(calculated_hpa - expected_hpa) < tolerance
                
                conversion_results.append({
                    'test_num': len(conversion_results) + 1,
                    'site': site,
                    'date': date,
                    'psi': round(psi_value, 3),
                    'calculated_hpa': round(calculated_hpa, 2),
                    'expected_hpa': round(expected_hpa, 2),
                    'difference': round(abs(calculated_hpa - expected_hpa), 2),
                    'factor': CalculationConstants.PSI_TO_HPA,
                    'passed': passed
                })
                
                if not passed:
                    all_passed = False
            
            status = "PASS" if all_passed else "FAIL"
            
            self.record_result(
                "PSI to hPa Conversion",
                "Calculations",
                status,
                f"Tested {len(conversion_results)} PSI to hPa conversions using factor: {CalculationConstants.PSI_TO_HPA}",
                {'conversions': conversion_results}
            )
            
            self.assertTrue(all_passed, "PSI to hPa conversions did not match expected values")
            
        except Exception as e:
            self.record_result(
                "PSI to hPa Conversion",
                "Calculations",
                "FAIL",
                f"Error during PSI conversion test: {str(e)}"
            )
            self.fail(f"PSI conversion test failed: {e}")
    
    # ========================================================================
    # DATA QUALITY TESTS
    # ========================================================================
    
    def test_statistical_anomaly_detection(self):
        """
        Detect statistical anomalies in validated data.
        
        Flags:
        - Depth changes > 15%
        - Pressure changes > 2%
        """
        if not os.path.exists(self.validated_output_file):
            self.record_result(
                "Statistical Anomaly Detection",
                "Data Quality",
                "SKIP",
                "validatedDepthData.csv not found - skipping test"
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
            
            # Analyse sites separately
            for site in df['Sample Point'].unique():
                site_data = df[df['Sample Point'] == site].copy()
                site_data = site_data.sort_values('Date Time (dd/mm/yyyy hh24:mi:ss)')
                
                # Depth anomalies
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
                
                # Pressure anomalies
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
            
            # Calculate statistics
            depth_stats = {
                'max_change_pct': round(max([abs(a['change_pct']) for a in depth_anomalies], default=0), 1),
                'count': len(depth_anomalies)
            }
            
            pressure_stats = {
                'max_change_pct': round(max([abs(a['change_pct']) for a in pressure_anomalies], default=0), 1),
                'count': len(pressure_anomalies)
            }
            
            # Sort and limit
            depth_anomalies.sort(key=lambda x: abs(x['change_pct']), reverse=True)
            pressure_anomalies.sort(key=lambda x: abs(x['change_pct']), reverse=True)
            depth_anomalies = depth_anomalies[:10]
            pressure_anomalies = pressure_anomalies[:10]
            
            status = "WARNING" if (len(depth_anomalies) > 0 or len(pressure_anomalies) > 0) else "PASS"
            
            self.record_result(
                "Statistical Anomaly Detection",
                "Data Quality",
                status,
                f"Found {depth_stats['count']} depth anomalies (>15%), {pressure_stats['count']} pressure anomalies (>2%)",
                {
                    'depth_anomalies': depth_anomalies,
                    'pressure_anomalies': pressure_anomalies,
                    'depth_stats': depth_stats,
                    'pressure_stats': pressure_stats
                }
            )
            
            self.assertTrue(True)  # Anomalies are warnings brosif, not failures
            
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
        """
        Generate the HTML a/b test report.
        """
        timestamp = cls.test_start_time.strftime("%d/%m/%Y %H:%M:%S")
        
        # Calculate summary stats
        total_tests = len(cls.results)
        passed = sum(1 for r in cls.results if r['status'] == 'PASS')
        failed = sum(1 for r in cls.results if r['status'] == 'FAIL')
        warnings = sum(1 for r in cls.results if r['status'] == 'WARNING')
        skipped = sum(1 for r in cls.results if r['status'] == 'SKIP')
        
        # Start building HTML
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
                <tr>
                    <td><strong>Total Tests:</strong></td><td>{total_tests}</td>
                    <td><strong>Duration:</strong></td>
                    <td>{(datetime.now() - cls.test_start_time).total_seconds():.2f}s</td>
                </tr>
                <tr>
                    <td><span class="pass">Passed:</span></td><td>{passed}</td>
                    <td><span class="fail">Failed:</span></td><td>{failed}</td>
                </tr>
                <tr>
                    <td><span class="warning">Warnings:</span></td><td>{warnings}</td>
                    <td class="skip">Skipped:</td><td>{skipped}</td>
                </tr>
            </table>
        </div>
        """
        
        # Group results by category
        categories = {}
        for result in cls.results:
            cat = result['category']
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(result)
        
        # Set category order
        category_order = [
            'Connectivity',
            'Logger Availability',
            'Data Structure',
            'Data Quality',
            'Calculations'
        ]
        
        # Generate sections for each category in set order
        for category in category_order:
            if category in categories:
                tests = categories[category]
                html += f'<div class="test-section"><h2>{category} Tests</h2>'
                
                for test in tests:
                    status_class = test['status'].lower()
                    test_description = TEST_DESCRIPTIONS.get(test['test_name'], '')
                    
                    html += f"""
                    <h3>{test['test_name']} - <span class="{status_class}">{test['status']}</span></h3>
                    """
                    
                    if test_description:
                        html += f'<div class="test-description">{test_description}</div>'
                    
                    html += f'<p>{test["details"]}</p>'
                    
                    # Add specific data based on test type
                    if test['data']:
                        html += cls._generate_test_data_html(test)
                
                html += '</div>'
        
        # Add footer
        html += HTML_FOOTER
        
        # Ensure output directory exists
        output_dir = os.path.dirname(cls.report_filename)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Save report
        with open(cls.report_filename, 'w', encoding='utf-8') as f:
            f.write(html)
        
        print(f"\n{'='*60}")
        print(f"TEST REPORT GENERATED: {cls.report_filename}")
        print(f"{'='*60}")
        print(f"Total: {total_tests} | Passed: {passed} | Failed: {failed} | Warnings: {warnings} | Skipped: {skipped}")
        print(f"{'='*60}\n")
    
    @classmethod
    def _generate_test_data_html(cls, test):
        """
        Generate HTML for test-specific data tables.
        
        Args:
            test (dict): Test result dictionary
            
        Returns:
            str: HTML string for test data
        """
        html = ""
        test_name = test['test_name']
        data = test['data']
        
        if test_name == 'Weather Data Structure':
            if data and 'all_columns' in data:
                html += '<h4>All Columns Found (Array Order):</h4>'
                html += '<div class="column-list"><pre>'
                for i, col in enumerate(data['all_columns']):
                    html += f"[{i}]: {col}\n"
                html += '</pre></div>'
                
                if 'required_columns' in data:
                    html += '<p><strong>Required columns verified at positions:</strong></p>'
                    html += '<ul>'
                    for col, pos in data['required_columns'].items():
                        html += f'<li>{col}: Position {pos}</li>'
                    html += '</ul>'
                
                if 'issues' in data and data['issues']:
                    html += '<h4>Issues Found:</h4>'
                    html += '<ul>'
                    for issue in data['issues']:
                        html += f'<li>{issue}</li>'
                    html += '</ul>'
        
        elif test_name == 'Site Configuration':
            if data.get('enabled'):
                html += f'<h4>Enabled Sites ({len(data["enabled"])}):</h4>'
                html += '<ul>'
                for site in data['enabled']:
                    html += f'<li>✓ {site}</li>'
                html += '</ul>'
            
            if data.get('disabled'):
                html += f'<h4>Disabled Sites ({len(data["disabled"])}):</h4>'
                html += '<ul>'
                for site in data['disabled']:
                    html += f'<li>✓ {site}</li>'
                html += '</ul>'
            
            if data.get('missing'):
                html += f'<h4>Missing Sites:</h4>'
                html += '<ul>'
                for site in data['missing']:
                    html += f'<li>✗ {site}</li>'
                html += '</ul>'
        
        elif test_name == 'Depth Calculation Verification':
            html += '<h4>Calculation Results:</h4>'
            html += '<table class="calculation-table">'
            html += '<tr><th>Test</th><th>Site</th><th>Date</th><th>Depth Raw (m)</th>'
            html += '<th>BOM Baro</th><th>Logger Baro</th><th>Adjustment</th>'
            html += '<th>Calculated</th><th>Expected</th><th>Diff</th><th>Result</th></tr>'
            
            for calc in data['calculations']:
                pass_fail = '<span class="pass">✓</span>' if calc['passed'] else '<span class="fail">✗</span>'
                adjustment_text = 'Yes' if calc['adjustment_applied'] else 'No'
                html += f"""<tr>
                    <td>{calc['test_num']}</td>
                    <td>{calc['site']}</td>
                    <td>{calc['date']}</td>
                    <td>{calc['depth_raw']}</td>
                    <td>{calc['bom_baro']}</td>
                    <td>{calc['logger_baro']}</td>
                    <td>{adjustment_text}</td>
                    <td>{calc['calculated']}</td>
                    <td>{calc['expected']}</td>
                    <td>{calc['difference']}</td>
                    <td>{pass_fail}</td>
                </tr>"""
            html += '</table>'
        
        elif test_name == 'PSI to hPa Conversion':
            html += '<h4>PSI to hPa Conversion Results:</h4>'
            html += '<table class="calculation-table">'
            html += '<tr><th>Test</th><th>Site</th><th>Date</th><th>PSI</th>'
            html += '<th>Calculated hPa</th><th>Expected hPa</th><th>Difference</th><th>Factor</th><th>Result</th></tr>'
            
            for conv in data['conversions']:
                pass_fail = '<span class="pass">✓</span>' if conv['passed'] else '<span class="fail">✗</span>'
                html += f"""<tr>
                    <td>{conv['test_num']}</td>
                    <td>{conv['site']}</td>
                    <td>{conv['date']}</td>
                    <td>{conv['psi']}</td>
                    <td>{conv['calculated_hpa']}</td>
                    <td>{conv['expected_hpa']}</td>
                    <td>{conv['difference']}</td>
                    <td>{conv['factor']}</td>
                    <td>{pass_fail}</td>
                </tr>"""
            html += '</table>'
        
        elif test_name == 'Statistical Anomaly Detection':
            if data['depth_anomalies']:
                html += '<h4>Depth Anomalies Detected:</h4>'
                html += '<div class="anomaly">'
                html += f"<p>Maximum change: {data['depth_stats']['max_change_pct']}%</p>"
                html += '<table>'
                html += '<tr><th>Site</th><th>Date</th><th>Value</th><th>Change %</th></tr>'
                for anomaly in data['depth_anomalies']:
                    html += f"""<tr>
                        <td>{anomaly['site']}</td>
                        <td>{anomaly['date']}</td>
                        <td>{anomaly['value']}</td>
                        <td>{anomaly['change_pct']}%</td>
                    </tr>"""
                html += '</table>'
                html += '</div>'
            
            if data['pressure_anomalies']:
                html += '<h4>Pressure Anomalies Detected:</h4>'
                html += '<div class="anomaly">'
                html += f"<p>Maximum change: {data['pressure_stats']['max_change_pct']}%</p>"
                html += '<table>'
                html += '<tr><th>Site</th><th>Date</th><th>Value</th><th>Change %</th></tr>'
                for anomaly in data['pressure_anomalies']:
                    html += f"""<tr>
                        <td>{anomaly['site']}</td>
                        <td>{anomaly['date']}</td>
                        <td>{anomaly['value']}</td>
                        <td>{anomaly['change_pct']}%</td>
                    </tr>"""
                html += '</table>'
                html += '</div>'
        
        return html

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def run_tests(output_path='transformed_data/abTestsReport.html'):
    """
    Execute tests, generate html report.
    
    Args:
        output_path (str): Path where the HTML report should be saved. Defaults to 'transformed_data/abTestsReport.html'.
    
    Returns:
        bool: True if all tests passed, False otherwise
    """
    # Set the report filename before running tests
    TestEnvironmentalPipeline.report_filename = output_path
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestEnvironmentalPipeline)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    exit(0 if success else 1)