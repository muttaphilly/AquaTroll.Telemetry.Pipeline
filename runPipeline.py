import os
from dotenv import load_dotenv
from playwright.sync_api import sync_playwright
import loggerScraper
import dataValidation
import autoEmail
import testsAB

# ------------ The captain. Scrape -> Validate -> Email. ------------

if __name__ == "__main__":
    load_dotenv()  # Load variables from .env file
    project_root = os.path.dirname(os.path.abspath(__file__))
    input_folder = os.path.join(project_root, 'data_downloads')
    output_folder = os.path.join(project_root, 'transformed_data')
    output_file = os.path.join(output_folder, 'validatedDepthData.csv')
    pbo_pools_file = os.path.join(output_folder, 'SWLVLGenericTemplate_greaterPBOPools.csv')
    log_file = os.path.join(output_folder, 'dataValidation.log')
    html_report_path = os.path.join(output_folder, 'abTestsReport.html')

# ----------------------------------------------------------------------
    # Step 1: Run scrapers
# ----------------------------------------------------------------------
    with sync_playwright() as playwright:
        loggerScraper.run(playwright, input_folder)

# ----------------------------------------------------------------------
    # Step 2: Data validation and verification
# ----------------------------------------------------------------------
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    dataValidation.consolidate_csv_files(input_folder, output_file, log_file)
    testsAB.run_tests(output_path=html_report_path)

# ----------------------------------------------------------------------
    # Step 3: Send emails
# ----------------------------------------------------------------------    
    # Validation and Verification email
    if os.path.exists(output_file):
            recipients = autoEmail.send_validated_data_email(
                output_file,
                html_report_path=html_report_path if os.path.exists(html_report_path) else None
            )
            if recipients:
                print(f"\nMonthly pools data sent to: {', '.join(recipients)}")
            else:
                print("Failed to send verification data to enviro team")
    else:
        print("Failed to send verification data to enviro team (output_file not found)")
                
    # Database email
    if os.path.exists(pbo_pools_file):
        recipients = autoEmail.send_database_email(pbo_pools_file)
        if recipients:
            print(f"\nValidated and verified data sent to environmental database: {', '.join(recipients)}")
        else:
            print("Failed to send enviro database email")
    else:
        print("Failed to send enviro database email (pbo_pools_file not found)")
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------