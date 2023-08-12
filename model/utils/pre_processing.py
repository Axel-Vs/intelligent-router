from pathlib import Path
import sys
import os
import urllib.request
import json
import numpy as np
import pandas as pd
import folium
import openrouteservice as ors
import matplotlib.colors as mcolors
import geopandas
import geopy
from ortools.linear_solver import pywraplp
from branca.element import Figure
from geopy.extra.rate_limiter import RateLimiter
from geopy.geocoders import Nominatim
import datetime
from datetime import datetime, timedelta


import coloredlogs
import logging as log
import webbrowser
import pycountry
import math

# # Get the parent directory of the current script (project root)
project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
project_root = project_root / "ParcelDeliveryOptimizer"
# # Add the paths of the relevant directories to sys.path
sys.path.append(str(project_root))


from project_utils import *


# Define the path to the data directory
data_directory = os.path.join(project_root, "data")
complete_data=data_pre_process(data_directory, ';', True)


# Data Filtering
drop_dup = False # If true we leave only one address on the period

data_filtered=data_filter(complete_data,'33000','FCA','All', [3], [2021],
                         ['Siemens Mobility GmbH'], 'Krauss-Maffei-Str.', drop_dup)
data_filtered.reset_index(drop=True, inplace=True)



# Time Standarization
# # Create one date-time column for pick-ups
node_df['Requested Loading'] = node_df['Requested Loading Date'] + ' ' + node_df['Requested Loading Time']
node_df['Requested Loading'] = pd.to_datetime(node_df['Requested Loading'])

node_df['Requested Delivery'] = node_df['Requested Delivery Date'] + ' ' + node_df['Requested Delivery Time']
node_df['Requested Delivery'] = pd.to_datetime(node_df['Requested Delivery'])



# Storing
# # Geocoding --------------------------------------------------------------------------------
df_geocoded_test = geo_coding(node_df, True)
df_geocoded_test.to_csv('C:/Users/z004d5wh/Documents/Master_Thesis/thesis_project/Data/df_dup_geocoded_202103.csv', index=False)