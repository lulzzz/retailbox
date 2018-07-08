# -*- coding: utf-8 -*-
import sys
import click
import recommender
import pandas as pd
import pickle

from output import displayHelpMessage, displayVersion
from data import search_customer, list_customers

retailBoxVersion = '0.1.0'

@click.command(add_help_option=False)

# Customer ID
@click.option('-c', '--customer', default=1, help='Input customer ID')

# Recommendations
@click.option('-r', '--recommend', default=3, help='Number of customer item recommendations')

# Customer Information
@click.option('-i', '--info', is_flag=True, default=False, help="Display customer information")

# Display Status Updates of Program
@click.option('-sd', '--status', is_flag=True, default=False, help='Display program info')

# Version
@click.option('-v', '--version', is_flag=True, default=False, help='Display installed version')

# Help
@click.option('-h', '--help', is_flag=True, default=False, help='Display help message')

# Search for Customer ID
@click.option('-s', '--search', is_flag=True, default=False, help='Search customer by ID')

# List number of Customers
@click.option('-l', '--list', is_flag=True, default=False, help='List customers')

def main(customer, recommend, customer_information, help, version, search, list):
    if (help):
        displayHelpMessage()
        sys.exit(0)
    else:
        if (version):
            displayVersion(retailBoxVersion)
        else:
            # Create dataframe and customer search table
            df = pd.read_pickle('../data/final/df_final.pkl') 
            table_pickle_file = open('../data/final/df_customer_table.pkl', "rb")
            customer_table = pickle.load(table_pickle_file)
            table_pickle_file.close()

            if (search):
                # Search for a Customer
                customer_search_id = click.prompt('❯ Please enter a Customer ID [0-4339]: ', type=int)
                search_customer(customer_search_id, df, customer_table)
                
            else:
                if (list):
                    # Display list of customers dependent on user input
                    list_length = click.prompt('❯ Enter a number of customers to list', type=int)
                    list_customers(list_length, df, customer_table)

                else:
                    # Train a recommender system using [algo] and display recommendations
                    # recommender(...)
                    print("hello")

if __name__ == '__main__':
    main()