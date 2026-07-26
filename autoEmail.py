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

def send_email_with_attachments(recipient_emails, subject, body, attachment_paths):
    """
    Sends an email with one or more attachments using credentials from .env.

    Args:
        recipient_emails (list): A list of recipient email addresses.
        subject (str): The subject line of the email.
        body (str): The plain text body of the email.
        attachment_paths (list or str): Single path or list of paths to files to attach.

    Returns:
        list or bool: List of recipients if successful, False otherwise.
    """
    # Convert single path to list for uniform handling
    if isinstance(attachment_paths, str):
        attachment_paths = [attachment_paths]
    
    # Load from .env 
    sender_email = os.getenv('EMAIL_SENDER_ADDRESS')
    sender_password = os.getenv('EMAIL_SENDER_PASSWORD')
    smtp_server = os.getenv('EMAIL_SMTP_SERVER')
    smtp_port_str = os.getenv('EMAIL_SMTP_PORT')
    
    # Validate email configuration
    if not all([sender_email, sender_password, smtp_server, smtp_port_str]):
        log.error("Email configuration missing in .env file. Cannot send email.")
        return False
    
    try:
        smtp_port = int(smtp_port_str)
    except ValueError:
        log.error(f"Invalid EMAIL_SMTP_PORT: {smtp_port_str}. Must be an integer.")
        return False
    
    # Validate recipient list
    if not recipient_emails or not isinstance(recipient_emails, list):
        log.error("Invalid recipient_emails list provided.")
        return False
    
    valid_recipients = [email for email in recipient_emails if email and isinstance(email, str)]
    if not valid_recipients:
        log.warning("No valid recipient email addresses found after filtering.")
        return False
    
    # Validate all attachment files exist
    for path in attachment_paths:
        if not os.path.exists(path):
            log.error(f"Attachment file not found: {path}. Cannot send email.")
            return False

    # Create the email
    message = MIMEMultipart()
    message['From'] = sender_email
    message['To'] = ", ".join(valid_recipients)
    message['Subject'] = subject
    message.attach(MIMEText(body, 'plain'))
    
    # Attach all files
    for attachment_path in attachment_paths:
        try:
            with open(attachment_path, "rb") as attachment:
                part = MIMEApplication(attachment.read(), Name=os.path.basename(attachment_path))
            part['Content-Disposition'] = f'attachment; filename="{os.path.basename(attachment_path)}"'
            message.attach(part)
            log.debug(f"Successfully attached file: {os.path.basename(attachment_path)}")
        except Exception as e:
            log.error(f"Error attaching file {attachment_path}: {e}")
            return False
    
    # Send the email
    context = ssl.create_default_context()
    try:
        server = smtplib.SMTP(smtp_server, smtp_port, timeout=30)
        server.starttls(context=context)
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, valid_recipients, message.as_string())
        return valid_recipients
    except Exception as e:
        log.error(f"Email error: {e}")
        return False
    finally:
        if 'server' in locals() and server:
            server.quit()

def parse_email_list(email_string):
    """Parse comma-separated email string into list."""
    if not email_string:
        return []
    return [email.strip() for email in email_string.split(',') if email.strip()]

def send_validated_data_email(csv_path, html_report_path=None):
    """
    Send validated depth data email with CSV & HTML A/B tests.
    
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
    
    # Build attachment list — prefer PDF if WeasyPrint produced one
    attachments = [csv_path]
    attachments_list = [f"* {os.path.basename(csv_path)}"]

    if html_report_path:
        pdf_report_path = os.path.splitext(html_report_path)[0] + '.pdf'
        if os.path.exists(pdf_report_path):
            attachments.append(pdf_report_path)
            attachments_list.append(f"* {os.path.basename(pdf_report_path)}")
        elif os.path.exists(html_report_path):
            attachments.append(html_report_path)
            attachments_list.append(f"* {os.path.basename(html_report_path)}")

    hardware_contact = os.getenv('HARDWARE_CONTACT', '')
    contact_line = (
        f"Please direct any queries to {hardware_contact}"
        if hardware_contact
        else "Please contact your system administrator for queries."
    )

    body = (
        f"GPO Environment\n\n"
        f"Please find attached validated depth data for {current_month}:\n"
        f"{chr(10).join(attachments_list)}\n\n"
        f"This email is not monitored.\n"
        f"{contact_line}\n\n"
    )
  
    return send_email_with_attachments(recipients, subject, body, attachments)

def send_database_email(attachment_path):
    """
    Send database upload email with CSV attachment.
    
    Args:
        attachment_path (str): Path to CSV file for database
    
    Returns:
        list or bool: List of recipients if successful, False otherwise
    """
    recipient_string = os.getenv('RECIPIENT_DATABASE')
    recipients = parse_email_list(recipient_string)
    
    if not recipients:
        log.warning("Database upload email recipient not found. Skipping email.")
        return False
    
    subject_template = os.getenv('DATABASE_EMAIL_SUBJECT', "{month} Depth Data")
    subject = subject_template.format(month=current_month)

    hardware_contact = os.getenv('HARDWARE_CONTACT', '')
    contact_line = (
        f"Please direct any queries to {hardware_contact}"
        if hardware_contact
        else "Please contact your system administrator for queries."
    )

    body_template = os.getenv(
        'DATABASE_EMAIL_BODY',
        (
            "GPO Environment\n\n"
            "For Upload To SQL Database: {filename}\n\n"
            "This email is not monitored.\n"
            "{contact_line}"
        )
    )

    body = body_template.format(
        filename=os.path.basename(attachment_path),
        contact_line=contact_line
    )
    
    return send_email_with_attachments(recipients, subject, body, attachment_path)