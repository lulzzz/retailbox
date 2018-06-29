# -*- coding: utf-8 -*-
import re

from colorama import init
from termcolor import color

init() # Termcolor support for win32

#
# Color Message Printing for Terminal
#

def printGreen(msg):
    print(color(msg, 'green'))

def printYellow(msg):
    print(color(msg, 'yellow'))

def printMagenta(msg):
    print(color(msg, 'magenta'))

def printRed(msg):
    print(color(msg, 'red'))


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


