# -*- coding: utf-8 -*-
import sys
import click
import recommender

from output import displayHelpMessage, displayVersion
from data import search_customer, list_customers
from . import __version__

retailBoxVersion = __version__




if __name__ == '__main__':
    main()