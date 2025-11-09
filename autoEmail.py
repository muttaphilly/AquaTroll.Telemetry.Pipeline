import os
import smtplib
import ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from dotenv import load_dotenv
from datetime import datetime
import logging

# Configure logger (sauce: runPipeline)
log = logging.getLogger(__name__)
load_dotenv()
current_month = datetime.now().strftime("%B %Y")

def send_email_with_attachment(recipient_emails, subject, body, attachment_path):
    """
    Sends an email with a specified attachment using credentials from .env.

    Args:
        recipient_emails (list): A list of recipient email addresses.
        subject (str): The subject line of the email.
        body (str): The plain text body of the email.
        attachment_path (str): The full path to the file to attach.

    Returns:
        bool: True if email was sent successfully, False otherwise.
    """
    # Loads from .env 
    sender_email = os.getenv('EMAIL_SENDER_ADDRESS')
    sender_password = os.getenv('EMAIL_SENDER_PASSWORD')
    smtp_server = os.getenv('EMAIL_SMTP_SERVER')
    smtp_port_str = os.getenv('EMAIL_SMTP_PORT')
    # Check that just worked
    if not all([sender_email, sender_password, smtp_server, smtp_port_str]):
        log.error("Email configuration missing in .env file. Cannot send email.")
        return False
    try:
        smtp_port = int(smtp_port_str)
    except ValueError:
        log.error(f"Invalid EMAIL_SMTP_PORT: {smtp_port_str}. Must be an integer.")
        return False
    # Check we got an email list
    if not recipient_emails or not isinstance(recipient_emails, list):
        log.error("Invalid recipient_emails list provided to send_email_with_attachment.")
        return False # i.e is nothing
    valid_recipients = [email for email in recipient_emails if email and isinstance(email, str)]
    if not valid_recipients:
        log.warning("No valid recipient email addresses found after filtering.")
        return False
    # Check we got csv files 
    if not os.path.exists(attachment_path):
        log.error(f"Attachment file not found: {attachment_path}. Cannot send email.")
        return False

    # Create the email template
    message = MIMEMultipart()
    message['From'] = sender_email
    message['To'] = ", ".join(valid_recipients)
    message['Subject'] = subject
    message.attach(MIMEText(body, 'plain'))
    # Attaching sweet, sweet data 
    try:
        with open(attachment_path, "rb") as attachment:
            part = MIMEApplication(attachment.read(), Name=os.path.basename(attachment_path))
        part['Content-Disposition'] = f'attachment; filename="{os.path.basename(attachment_path)}"'
        message.attach(part)
        log.debug(f"Successfully attached file: {os.path.basename(attachment_path)}")
    except Exception as e:
        log.error(f"Error attaching file {attachment_path}: {e}")
        return False
    # Launching it into interwebs
    context = ssl.create_default_context()
    try:
        server = smtplib.SMTP(smtp_server, smtp_port, timeout=30)
        server.starttls(context=context)
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, valid_recipients, message.as_string())       
    # Return the recipient list (so can print in runPipeline.py)
        return valid_recipients
    except Exception as e:
        log.error(f"Email error: {e}")
        return False
    finally:
        if 'server' in locals() and server:
            server.quit()

# ------- Handle email distribution with comma-seperated string
def parse_email_list(email_string):
    if not email_string:
        return []
    
    # Split by comma, strip whitespace, and filter out empty strings
    return [email.strip() for email in email_string.split(',') if email.strip()]

# ------- Email to environment team (for EMS)
def send_validated_data_email(csv_path, html_report_path=None):
    """
    Send validated depth data email with CSV and HTML A/B tests.
    
    Args:
        csv_path (str): Path to CSV data file (required)
        html_report_path (str): Optional path to HTML test report (abTestsReport.html)
    
    Returns:
        list or bool: List of recipients if successful, False otherwise
    """
    recipient_string = os.getenv('RECIPIENT_VALIDATION')
    recipients = parse_email_list(recipient_string)
    
    if not recipients:
        log.warning("Validation report email list is not defined. Skipping email.")
        return False
    
    subject = os.getenv('VALIDATION_EMAIL_SUBJECT', "Logger Validation Report")
    
    # Build attachment list with bullet points
    attachments_list = [f"* {os.path.basename(csv_path)}"]
    if html_report_path and os.path.exists(html_report_path):
        attachments_list.append(f"* {os.path.basename(html_report_path)}")
    
    # Format body with bullet points
    body = f"Please find attached validated depth data for {current_month}:\n" + "\n".join(attachments_list)
        
    # Attach both files (loads from env)
    sender_email = os.getenv('EMAIL_SENDER_ADDRESS')
    sender_password = os.getenv('EMAIL_SENDER_PASSWORD')
    smtp_server = os.getenv('EMAIL_SMTP_SERVER')
    smtp_port_str = os.getenv('EMAIL_SMTP_PORT')
        
    if not all([sender_email, sender_password, smtp_server, smtp_port_str]):
        log.error("Email configuration missing in .env file. Cannot send email.")
        return False
        
    try:
        smtp_port = int(smtp_port_str)
    except ValueError:
        log.error(f"Invalid EMAIL_SMTP_PORT: {smtp_port_str}. Must be an integer.")
        return False
        
    if not os.path.exists(csv_path):
        log.error(f"CSV file not found: {csv_path}. Cannot send email.")
        return False
        
    # Create email
    message = MIMEMultipart()
    message['From'] = sender_email
    message['To'] = ", ".join(recipients)
    message['Subject'] = subject
    message.attach(MIMEText(body, 'plain'))
        
    # Attach CSV
    try:
        with open(csv_path, "rb") as attachment:
            part = MIMEApplication(attachment.read(), Name=os.path.basename(csv_path))
        part['Content-Disposition'] = f'attachment; filename="{os.path.basename(csv_path)}"'
        message.attach(part)
        log.debug(f"Successfully attached file: {os.path.basename(csv_path)}")
    except Exception as e:
        log.error(f"Error attaching file {csv_path}: {e}")
        return False
        
    # Attach A/B report if present
    if html_report_path and os.path.exists(html_report_path):
        try:
            with open(html_report_path, "rb") as attachment:
                part = MIMEApplication(attachment.read(), Name=os.path.basename(html_report_path))
            part['Content-Disposition'] = f'attachment; filename="{os.path.basename(html_report_path)}"'
            message.attach(part)
            log.debug(f"Successfully attached file: {os.path.basename(html_report_path)}")
        except Exception as e:
            log.error(f"Error attaching file {html_report_path}: {e}")
            # Continue without it

    # Send the email
    context = ssl.create_default_context()
    try:
        server = smtplib.SMTP(smtp_server, smtp_port, timeout=30)
        server.starttls(context=context)
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, recipients, message.as_string())
        return recipients
    except Exception as e:
        log.error(f"Email error: {e}")
        return False
    finally:
        if 'server' in locals() and server:
            server.quit()

# ------- Auto-upload to environmental database
def send_database_email(attachment_path):
    recipient_string = os.getenv('RECIPIENT_DATABASE')
    recipients = parse_email_list(recipient_string)
    
    if not recipients:
        log.warning("Database upload email recipient not found. Skipping email.")
        return False
    
    subject_template = os.getenv('DATABASE_EMAIL_SUBJECT', "{month} Depth Data")
    subject = subject_template.format(month=current_month)
    
    body_template = os.getenv('DATABASE_EMAIL_BODY', "For Upload To Database: {filename}")
    body = body_template.format(filename=os.path.basename(attachment_path))
    
    return send_email_with_attachment(recipients, subject, body, attachment_path)