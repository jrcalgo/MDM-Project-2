#!/bin/bash

# make sure .env is set correctly, requirements.txt has been installed, and python is >=3.8

python driver.py --input query_group2.xlsx --output rawSearchResults.json --formatted-output searchResults.json

python ComparisionFunction.py

python ValidatorFunction.py

python Generate_CSV.py

python Generate_final_output.py
