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


