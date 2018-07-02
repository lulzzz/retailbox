# -*- coding: utf-8 -*-
import re

from colorama import init
from termcolor import colored

init() # Termcolor support for win32

#
# Color Message Printing for Terminal
#

def printGreen(msg):
    print(colored(msg, 'green'))

def printYellow(msg):
    print(colored(msg, 'yellow'))

def printMagenta(msg):
    print(colored(msg, 'magenta'))

def printRed(msg):
    print(colored(msg, 'red'))


#
# Display Terminal Messages / Prompts
#

help_message = '''
  🛍️ Machine Learning eCommerce Recommender System

  Usage
    $ retailbox [<options> ...]

'''

def displayHelpMessage():
    print(help_message)

def displayVersion(version):
    print('RetailBox' + version)


'''
Display Customer Information:

customer_id     (int)
items_purchases (list)
country         (string) 
amount_spent    (float)  cost of all items in list

'''

def display_customer_information(customer_id, country, items, amount_spent):
    # Print Customer ID
    print('\n')
    print(colored('> Customer_ID: ' + str(customer_id), 'green'))

    # Print Customer's Country of Origin
    print(colored('• Country: ' + country, 'green'))

    # Print Items the Customer Purchased
    print(colored('• Items Purchased:', 'green'))
    counter = 0
    for i in items:
        i = i.lower()
        if counter < 5:
            print(colored('  - ' + i.title(), 'green'))
            counter += 1
        else:
            print(colored('  and ' + str(len(items) - 5) + ' other items..', 'green'))
            break
    
    # Print Total Amount Spent and Number of Items Purchased
    print(colored('• Number of Items: ' + str(len(items)), 'green'))
    print(colored('• Amount Spent: ' + '{:0.2f}'.format(amount_spent), 'green'))

    return 0
    

def display_performance():
    print(colored("★ Recommender System Methods & Performance", 'yellow'))
    # Baseline, ALS, SGD-MF-LogLoss, SGD-MF-BPR, RNN

