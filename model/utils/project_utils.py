from pathlib import Path
import sys
# Get the parent directory of the current script (project root)
project_root = Path(__file__).resolve().parent.parent
# Add the paths of the relevant directories to sys.path
sys.path.append(str(project_root))

import datetime
import pandas as pd
import numpy as np
import logging as log
import coloredlogs
import math
import json
import os
import pycountry

from IPython.display import display


################################################################################################################################################
# Logout configuration ----------------------------------------------------------------------------------------------------
################################################################################################################################################
# Create a logger object.
logger = log.getLogger(__name__)
# Create a filehandler object
fh = log.FileHandler('spam.log')
fh.setLevel(log.DEBUG)
# Create a ColoredFormatter to use as formatter for the FileHandler
formatter = coloredlogs.ColoredFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)
# Install the coloredlogs module on the root logger
coloredlogs.install(level='DEBUG')


################################################################################################################################################
# Pre-processing functions ----------------------------------------------------------------------------------------
################################################################################################################################################
def column_input(df, column_values):
    """
    Pre-processes a specific column of a DataFrame.
    Args:
        df (pd.DataFrame): The DataFrame.
        column_values (str): The name of the column to process.
    Returns:
        pd.Series: The pre-processed column.
    """
    # Pre-process string values
    if pd.api.types.is_string_dtype(df[column_values]):
        # General replacements
        df[column_values] = df[column_values].str.replace('†', ' ')
        df[column_values] = df[column_values].str.replace('--', '-')
        df[column_values] = df[column_values].str.replace('ß', 'ss')
        # Capital replacements
        df[column_values] = df[column_values].str.replace('Ä', 'AE')
        df[column_values] = df[column_values].str.replace('Ö', 'OE')
        df[column_values] = df[column_values].str.replace('Ü', 'UE')
        # Minuscule replacements
        df[column_values] = df[column_values].str.replace('ä', 'ae')
        df[column_values] = df[column_values].str.replace('ö', 'oe')
        df[column_values] = df[column_values].str.replace('ü', 'ue')

    return df[column_values]

def data_pre_process(path, dataset_name, sep, complete):
    """
    Performs data pre-processing on the input CSV file.
    Args:
        path (str): Path to the CSV file.
        sep (str): Separator used in the CSV file.
        complete (bool): True if complete data is provided.
    Returns:
        pd.DataFrame: The pre-processed DataFrame.
    """
    df = pd.read_csv( os.path.join(path,dataset_name),  sep=sep, engine='python')
    
    # Standardize values in all columns
    for column_names in df.columns:
        df[column_names] = column_input(df, column_names)
    
    # Change country id for country name on Vendors and Recipient ----
    # Download country catalogue
    countries = {}
    for country in pycountry.countries:
        countries[country.alpha_2] = country.name
    # Obtain all the countries id on the dataset
    if complete == True:
        complete_contries_codes = pd.concat([df['Recipient Country'], df['Vendor Country']], axis=0).unique()
    else:
        complete_contries_codes = df['Recipient Country'].unique()
    complete_countries_names = [countries.get(country_id, 'Unknown code') for country_id in complete_contries_codes]
    # Create data with all the countries id
    complete_countries_names = pd.DataFrame(complete_countries_names, columns=['countries_names'])
    complete_contries_codes = pd.DataFrame(complete_contries_codes, columns=['countries_codes'])
    country_catalogue = pd.concat([complete_contries_codes, complete_countries_names], axis=1)
    # Add the name instead of ids
    if complete == True:
        df = df.merge(country_catalogue, how='left', left_on='Recipient Country', right_on='countries_codes')
        df = df.rename(columns={'countries_names': 'Recipient Country Name'})
        df = df.merge(country_catalogue, how='left', left_on='Vendor Country', right_on='countries_codes')
        df = df.rename(columns={'countries_names': 'Vendor Country Name'})
        df.drop(['Recipient Country', 'Vendor Country', 'countries_codes_x', 'countries_codes_y'], axis=1, inplace=True)
        # Create Addresses
        # Assuming 'df' is your DataFrame
        df['Recipient City'] = df['Recipient City'].astype(str)
        df['Recipient Country Name'] = df['Recipient Country Name'].astype(str)

        # Perform the concatenation after ensuring both columns are strings
        df['Recipient Address'] = df.apply(
                                            lambda row: f"{row['Recipient Street']}, {str(row['Recipient Postcode'])}, {row['Recipient City']}, {row['Recipient Country Name']}", 
                                            axis=1
                                           )
        df['Vendor Address'] = df.apply(
                                        lambda row: f"{row['Vendor Street']}, {str(row['Vendor Postcode'])}, {row['Vendor City']}, {row['Vendor Country Name']}",
                                        axis=1
                                        )

    else:    
        df = df.merge(country_catalogue, how='left', left_on='Recipient Country', right_on='countries_codes')
        df = df.rename(columns={'countries_names': 'Recipient Country Name'})
        df.drop(['Recipient Country', 'countries_codes'], axis=1, inplace=True)
        # Create Addresses
        df['Recipient Address'] = df['Recipient Street'] + ', ' + df['Recipient Postcode'].map(str) + ', ' + \
                                  df['Recipient City'] + ', ' + df['Recipient Country Name']        
    return df

# Filtering functions ----------------------------------------------------------------------------------------
def data_filter(df, fces_stype, incoterms, supp_country, req_month, req_year, 
                Recipient_name, Recipient_address_pattern, drop_dup):
    """
    Filters the input DataFrame based on provided parameters.
    Args:
        df (pd.DataFrame): The DataFrame to filter.
        fces_stype (str): FCES_STYPE value to filter by.
        incoterms (str): Incoterms value to filter by.
        supp_country (str): Vendor country name to filter by.
        req_month (list): List of requested loading months.
        req_year (list): List of requested loading years.
        Recipient_name (list): List of Recipient names to filter by.
        Recipient_address_pattern (str): Pattern to match in Recipient address.
        drop_dup (bool): True to drop duplicate entries.
    Returns:
        pd.DataFrame: The filtered DataFrame.
    """
    df['Requested Loading Month'] = pd.DatetimeIndex(df['Requested Loading Date']).month
    df['Requested Loading Year'] = pd.DatetimeIndex(df['Requested Loading Date']).year
    t = pd.DatetimeIndex(df['Requested Loading Date'])
    df['Requested Loading Date'] = np.datetime_as_string(t, unit='D')

    if supp_country == 'All':        
        new_df = df.loc[
            (df['FCES_STYPE'] == fces_stype) &
            (df['Incoterms®'] == incoterms) &
            (df['Requested Loading Month'].isin(req_month)) &
            (df['Requested Loading Year'].isin(req_year)) &
            (df['Recipient Name'].isin(Recipient_name)) &
            (df['Recipient Street'].str.contains(Recipient_address_pattern, case=False, na=False))
        ]
    else:
        new_df = df.loc[
            (df['FCES_STYPE'] == fces_stype) &
            (df['Incoterms®'] == incoterms) &
            (df['Vendor Country Name'] == supp_country) &
            (df['Requested Loading Month'].isin(req_month)) &
            (df['Requested Loading Year'].isin(req_year)) &
            (df['Recipient Name'].isin(Recipient_name)) &
            (df['Recipient Street'].str.contains(Recipient_address_pattern, case=False, na=False))
        ]

    if drop_dup == True:
        new_df.drop_duplicates(subset=['Vendor Address'], inplace=True)  # keep first value (more places on the testing)

    return new_df

def split_values(euro_palett, current_ldm, current_weight, minimum_ldm, maximum_ldm, maximum_weigth, detail):
    """
    Splits cargo values based on provided criteria.
    Args:
        euro_palett (list): List containing palett dimensions.
        current_ldm (float): Current loading meters.
        current_weight (float): Current total weight.
        minimum_ldm (float): Minimum loading meters.
        maximum_ldm (float): Maximum loading meters.
        maximum_weigth (float): Maximum weight.
        detail (bool): True for detailed split, False for regular split.
    Returns:
        int: Number of splits.
        dict: Dictionary containing split loading meters.
        dict: Dictionary containing split weights.
    """
    split_ldm = {}
    split_weight = {}

    # Split based on detailed criteria
    if detail == True:
        euro_constant = euro_palett[0] * euro_palett[1] / euro_palett[2]  # Palett dimensions

        n_items = current_ldm / euro_constant
        n_max_allow = n_items * maximum_ldm / current_ldm

        n_items_rest = n_items % n_max_allow
        rest_ldm = round(n_items_rest * euro_constant, 2)

        if rest_ldm >= minimum_ldm:
            reps_values = math.floor(n_items / n_max_allow)

            for i in range(1, reps_values + 1):
                split_ldm[i] = round(n_max_allow * euro_constant, 2)
                split_weight[i] = round((current_weight * n_max_allow) / n_items, 2)

            n_items_rest = n_items % n_max_allow
            split_ldm[i + 1] = round(n_items_rest * euro_constant, 2)
            split_weight[i + 1] = round((current_weight * n_items_rest) / n_items, 2)
        else:
            reps_values = math.ceil(n_items / n_max_allow)
            for i in range(1, reps_values + 1):
                equal_split_items = math.ceil(n_items / 4)
                split_ldm[i] = round(equal_split_items * euro_constant, 2)
                split_weight[i] = round((current_weight * equal_split_items) / n_items, 2)
    # Regular split
    else:
        reps_values = math.floor(current_ldm / maximum_ldm)

        for i in range(1, reps_values + 1):
            split_ldm[i] = maximum_ldm
            split_weight[i] = math.ceil(maximum_ldm * current_weight / current_ldm)

        rest = current_ldm % maximum_ldm
        if rest != 0:
            split_ldm[i + 1] = rest
            split_weight[i + 1] = math.ceil(rest * current_weight / current_ldm)

    if max(split_weight.values()) > maximum_weigth:
        raise TypeError("New weight exceeds maximum capactiy:", max(split_weight.values()))

    return len(split_weight), split_ldm, split_weight




################################################################################################################################################
# Import parameters ------------------------------------
################################################################################################################################################
def import_parameters(parameters_path):
    network_params_file_name = 'graph_params.txt'
    model_params_file_name = 'model_params.txt'
    simulation_params_file_name = 'simulation_params.txt'

    network_params_path = os.path.join(parameters_path, network_params_file_name)
    model_params_path = os.path.join(parameters_path, model_params_file_name)
    simulation_params_path = os.path.join(parameters_path, simulation_params_file_name)


    network_params = json.load(open(network_params_path))
    networkmodel_params_params = json.load(open(model_params_path))
    simulation_params = json.load(open(simulation_params_path))
    
    return network_params, networkmodel_params_params, simulation_params


################################################################################################################################################
# Optimizer ------------------------------------
################################################################################################################################################
def SolVal(x):
    if type(x) is not list:
        return 0 if x is None else x.SolutionValue()
    elif type(x) is list:
        return [SolVal(e) for e in x ]

def ObjVal(x):
    return x.Objective().Value()

def info_matrix_definition(coordinates, source_node, shape, client):
    info_mtrx = client.distance_matrix(
                                            locations=coordinates,
                                            profile='driving-hgv',
                                            metrics= ['distance', 'duration'],
                                            units = 'km',
                                            resolve_locations=True,
                                            validate=False
                                            )
    distance_mtrx = info_mtrx['distances']
    time_mtrx = info_mtrx['durations']
    
    if shape == 'horizontal':
        # Setting first row to zero
        zero_row = np.zeros(len(distance_mtrx))               
        
        distance_mtrx[source_node] = list(zero_row)
        time_mtrx[source_node] = list(zero_row)
        
        distance_mtrx = np.asarray(distance_mtrx)
        time_mtrx = np.asarray(time_mtrx)
    elif shape == 'vertical':
        for i in range(len(distance_mtrx)):
            distance_mtrx[i][source_node] = 0
            time_mtrx[i][source_node] = 0
        distance_mtrx = np.asarray(distance_mtrx)
        time_mtrx = np.asarray(time_mtrx)
    return distance_mtrx, time_mtrx

def cargo_vector(vendor_df):
    q = []
    q.append(0)
    for i in vendor_df['Total Gross Weight']:
        q.append(i)
    return q

def loading_vector(vendor_df):
    m = []
    m.append(0)
    for i in vendor_df['Calculated Loading Meters']:
        m.append(i)
    return m

def init_max_num_vehicles(n, q, max_weight):
    max_num_vehicles=int(n/3)
    while (max_num_vehicles-1)*max_weight < sum(q):
        max_num_vehicles += 1
    max_num_vehicles = n - math.ceil(math.sqrt(n/2))
    return max_num_vehicles

## ----------------------------------------------------------------------------------------------------------------------------------------
def periods_generator(simulation_period, simulation_interval, vendor_start_hr, pickup_end_hr):
    start = simulation_period[0]
    end = simulation_period[1]
    delta = datetime.timedelta(days=simulation_interval)

    start = datetime.datetime.strptime( start, '%Y-%m-%d' )
    end = datetime.datetime.strptime( end, '%Y-%m-%d' )
    t = start

    periods = []
    while t <= end :
        if t + delta < end:
            periods.append( [t.replace(hour=vendor_start_hr).strftime("%Y-%m-%d %H:%M:%S"), 
                             (t + delta).replace(hour=pickup_end_hr + simulation_interval).strftime("%Y-%m-%d %H:%M:%S") ] )    
            t = t + (delta + datetime.timedelta(days=1) )
        else:
            periods.append( [t.replace(hour=vendor_start_hr).strftime("%Y-%m-%d %H:%M:%S"), 
                             end.replace(hour=pickup_end_hr + simulation_interval).strftime("%Y-%m-%d %H:%M:%S")] )

            t = t + (delta + datetime.timedelta(days=1) )
    return periods

    
def filter_dates(df, period):
    df_f = df.copy(deep=True)
    
    # init_p = datetime.datetime.strptime(period[0], '%Y-%m-%d %H:%M:%S')
    init_p = datetime.datetime.strptime(period[0], '%Y-%m-%d %H:%M:%S')
    endg_p = datetime.datetime.strptime(period[1], '%Y-%m-%d %H:%M:%S')
    
    df_f['temp_date'] = pd.to_datetime(df_f['Requested Loading'], format='%d.%m.%Y %H:%M')
    df_f=df_f[ (df_f['temp_date'] >= init_p) &  (df_f['temp_date'] <= endg_p) ]


    # if len(df_f) == 0:       
    #     raise ValueError('---------------------------- No service needed on this period. Please try later.')
            
    df_f.drop('temp_date', axis=1, inplace=True)
    df_f.sort_values(by=['Requested Loading'],inplace=True)   
    df_f.reset_index(drop=True, inplace=True)
    df_time = pd.to_datetime(df_f['Requested Loading'])
    
    # Week days -----------------------------------------------------------------
    nur_dates = df_time.dt.date.drop_duplicates()
    nur_dates.reset_index(drop=True, inplace=True)
    week_days = [date_obj.strftime('%A') for date_obj in nur_dates]
    week_days = pd.DataFrame(week_days, columns = ['Week Days'])
    # display(pd.concat([nur_dates, week_days], axis=1))
    # Times ---------------------------------------------------------------------
    # times_values = [dates_values.strftime("%H") for dates_values in df_time ]
    return df_f # complete period window



################################################################################################################################################
# Time Matrix ----------------------------------------------------------------------------
################################################################################################################################################

## ----------------------------------------------------------------------------------------------------------------------------------------
def cr_nodes(vendor_df):    
    return list(np.arange( 0, len(vendor_df)+1, 1))

def discret_time_matrix(time_matrix, discretization_constant):    
    time_distance_matrix = time_matrix/(3600) # convert to hours
    
    non_zero_time_matrix = time_distance_matrix[time_distance_matrix != 0]
    current_discretization_constant = max( math.floor(np.min(non_zero_time_matrix)+1), discretization_constant)
    # current_discretization_constant = min(10, current_discretization_constant)
    time_distance_matrix = np.ceil(time_distance_matrix/current_discretization_constant) 
    return time_distance_matrix.astype(int), current_discretization_constant


def date_index(discretization_constant, e_day, e_hour, min_day, tour_days, Tau_hours, arrival_type, maximum=[]):
    #if 24 % discretization_constant != 0:
    #    raise ValueError('check')

    if len(maximum) != 0:
        max_day = datetime.datetime.strptime(maximum, '%d.%m.%Y %H:%M')
    else:
        max_day = min_day + datetime.timedelta(days = tour_days + 1)
        max_day = max_day
    
    # total_tour_days = np.arange(min_day, min_day+tour_days+1, 1)
    total_tour_days = pd.date_range(start=min_day, end=max_day)
    
    counter_tau = {}
    prev_hour=0
    val_day = 0
    counter_val = 0
    counter_tau[val_day] = []
    for elements in range(len(Tau_hours)):

        current_hour = Tau_hours[elements]
        if current_hour - prev_hour < 0:
            val_day += 1
            counter_val = 1
            counter_tau[val_day] = counter_val
        else:
            counter_val += 1
            counter_tau[val_day] = counter_val
            
        prev_hour = current_hour

    index_tour_day = np.where(np.array(total_tour_days.day) == e_day)[0][0]
    
    indx_to_consider= np.arange(counter_tau[index_tour_day]*index_tour_day, 
                                counter_tau[index_tour_day]*index_tour_day + counter_tau[val_day], 
                                1 )

    indx_to_consider = indx_to_consider[indx_to_consider < len(Tau_hours) ]  # precaution of not exceeding
    indx_to_consider = list(indx_to_consider)

    min_tau_hr = np.array(Tau_hours)[indx_to_consider]
    min_tau_hr = list(min_tau_hr)

    if len(np.where(np.array(min_tau_hr) == e_hour)[0]) != 0:
        general_index = np.where(np.array(min_tau_hr) == e_hour)[0][0]
    elif e_hour < min(min_tau_hr):
        general_index = 0
    elif e_hour > max(min_tau_hr):
        general_index = len(min_tau_hr) + 1
    else: # modify arrivals according to the interval. "Risk-averse": Pushing up earliest arrivals and Pushing down latest ones.
        r = e_hour
        if arrival_type == 'early_values':
            while len(np.where(np.array(min_tau_hr) == r)[0]) == 0:
                r = r + 1   
        elif arrival_type == 'latest_values':
            while len(np.where(np.array(min_tau_hr) == r)[0]) == 0:
                r = r - 1   
        else:
            raise ValueError('Plea specifiy arrival type.')            
            
        general_index = np.where(np.array(min_tau_hr) == r)[0][0]
        
    index_val = counter_tau[index_tour_day]*index_tour_day + general_index 
    return index_val

def consecutive(data, stepsize=1):
    return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)

def information_index(vehicles_vector):    
    vehicles_vector = SolVal(vehicles_vector)
    output_values = np.argwhere( np.array(vehicles_vector) == 1 )
    output_values = [i[0] for i in output_values]
    return output_values

def A_index(A_ex, i, delta):    
    A_fun = []
    for j in range(len(A_ex)):
        if delta == 'delta_out': 
            if i == A_ex[j][0][0]:     
                A_fun.append( [[A_ex[j][0][0], A_ex[j][0][1]], [A_ex[j][1][0], A_ex[j][1][1]]] )   
        elif delta == 'delta_in': 
            if i == A_ex[j][1][0]:    
                A_fun.append( [[A_ex[j][0][0], A_ex[j][0][1]], [A_ex[j][1][0], A_ex[j][1][1]]] )   
    return A_fun

def next_index(route, current_index):
    alpha=0
    for i in range(len(route)):
        if route[i][0] == current_index:
            alpha = i            
            break
    return route[alpha][2], route[alpha][3] 


def inv_date_index(discretization_constant, index_to_check, min_date, Tau_hours):
    # min_day = min_date.day
    
#     len_one_day = math.ceil(24/discretization_constant)
#     val_day = math.floor(len(Tau_hours[:index_to_check])/len_one_day)
    val_day=0
    prev_hour=0
    for hours_passed in Tau_hours[:index_to_check]:
        current_hour = hours_passed
        if current_hour - prev_hour < 0:
            val_day += 1
        prev_hour = current_hour
    
    output_value = Tau_hours[index_to_check]    
    str_hour = str(datetime.timedelta(hours=int(output_value))).rsplit(':', 1)[0]
    
    losdatstr = min_date + datetime.timedelta(days=val_day)
    losdatstr = losdatstr.date()
    str_complete = str(losdatstr)  + ' at ' + str_hour
    
    return losdatstr, str_complete


def diff_networks(big_network, small_network):
    diff_elements = []
    for i in range(len(big_network)):
        elem_to_com = big_network[i]
        b=0
        for j in range(len(small_network)):
            if elem_to_com == small_network[j]:
                b=1
                break
        if b == 0:        
            diff_elements.append(elem_to_com)
    return diff_elements

    
################################################################################################################################################
# Post-processing functions ---------------------------------------------------------
################################################################################################################################################
def information_index(vehicles_vector):
    output_values = np.argwhere( np.array(vehicles_vector) == 1 )
    output_values = [i[0] for i in output_values]
    return output_values

def next_index(route, current_index):
    alpha=0
    for i in range(len(route)):
        if route[i][0] == current_index:
            alpha = i            
            break
    return route[alpha][2], route[alpha][3] 

def inv_date_index(index_to_check, discretization_constant, min_day, Tau_hours, time_network=[]):
    if len(time_network) != len(Tau_hours):
        last_value = Tau_hours[len(Tau_hours)-1]
        extra = len(time_network)-len(Tau_hours)

        extra_list = np.arange(last_value, last_value + extra*discretization_constant, discretization_constant )
        extra_list = list(extra_list)[1:]
        Tau_hours = Tau_hours + extra_list

    len_one_day = math.ceil(24/discretization_constant)
    val_day = math.floor(len(Tau_hours[:index_to_check])/len_one_day)

    output_value = Tau_hours[index_to_check]    
    str_hour = str(datetime.timedelta(hours=int(output_value))).rsplit(':', 1)[0]
    str_complete = 'day ' + str(min_day + val_day) + ' at ' + str_hour

    return str_complete