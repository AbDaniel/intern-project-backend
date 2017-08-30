import sys
# Import smtplib for the actual sending function
import smtplib

# Import the email modules we'll need
from email.mime.text import MIMEText

def printToFile(toFile):
    print('result: ' + toFile)
    sys.stdout.flush()
    with open('/tmp/test.log', 'a') as myfile:
        myfile.write('result: ' + toFile + "\n")

    return True