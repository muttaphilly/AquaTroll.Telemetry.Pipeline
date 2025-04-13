import os
from dotenv import load_dotenv
from playwright.sync_api import sync_playwright
import loggerScraper
import dataValidation
import autoEmail

# ------------ The captain. Scrape -> Validate -> Email. ------------

if __name__ == "__main__":
    load_dotenv()  # Load variables from .env file
    # Assumes you're running from the project root
    project_root = os.path.dirname(os.path.abspath(__file__))
    input_folder = os.path.join(project_root, 'data_downloads')
    output_folder = os.path.join(project_root, 'transformed_data')
    output_file = os.path.join(output_folder, 'validatedDepthData.csv')
    pbo_pools_file = os.path.join(output_folder, 'greaterPBOPools.csv')
    log_file = os.path.join(output_folder, 'dataValidation.log')

    # Step 1: Run scrapers
    with sync_playwright() as playwright:
        loggerScraper.run(playwright, input_folder)

    # Step 2: Data validation and verification
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    dataValidation.consolidate_csv_files(input_folder, output_file, log_file)

    # Step 3: Send emails
    for file_path, email_function, success_msg, fail_msg in [
        (output_file, autoEmail.send_validated_data_email, "Monthly pools data sent to", "Failed to send verification data to enviro team"),
        (pbo_pools_file, autoEmail.send_pbo_pools_email, "Validated and verified data sent to environmental database", "Failed to send enviro database email")
    ]:
        if os.path.exists(file_path) and (recipients := email_function(file_path)):
            print(f"\n{success_msg}: {', '.join(recipients)}")
        else:
            print(f"{fail_msg}")

# ----------------------------------------------------------------------
