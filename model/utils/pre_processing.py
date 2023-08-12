import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
import logging as log
import urllib.request
import json

import numpy as np
import pandas as pd
import folium
import matplotlib.colors as mcolors
import geopandas
import geopy
from geopy.extra.rate_limiter import RateLimiter
from geopy.geocoders import Nominatim
from branca.element import Figure
from ortools.linear_solver import pywraplp
from openrouteservice import client as ors
import pycountry
import webbrowser
import coloredlogs

from project_utils import data_pre_process

# Configure logging
coloredlogs.install(level='DEBUG')

# Get the parent directory of the current script (project root)
project_root = Path(__file__).resolve().parent.parent.parent
# Add the paths of the relevant directories to sys.path
sys.path.append(str(project_root))

# Define the path to the data directory
data_directory = os.path.join(project_root, 'data')
data_name = 'amazon_test_dataset.csv'

complete_data = data_pre_process(data_directory, data_name, ',', True)
complete_data.reset_index(drop=True, inplace=True)