import urllib.request
import json
import os
import numpy as np
import pandas as pd
import folium
import openrouteservice as ors
import matplotlib.colors as mcolors
import geopandas
import geopy
from ortools.linear_solver import pywraplp
import pandas as pd

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


C:\Users\z004d5wh\Documents\Master_Thesis\thesis_project\GBS\2_data
general_path='C:/Users/z004d5wh/Documents/Master_Thesis/thesis_project/Data'
raw_data_path=os.path.join(general_path, 'raw_data.csv')
complete_data=data_pre_process(raw_data_path, ';', True)



