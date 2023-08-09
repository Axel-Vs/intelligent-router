import datetime
import pandas as pd
import numpy as np
import logging as log
import coloredlogs
import math
import json
import os

from IPython.display import display

# Logout configuration ----------------------------------------------------------------------------------------------------
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

# Import parameters ------------------------------------
def import_parameters(parameters_path):
    network_params_file_name = 'network_params.txt'
    model_params_file_name = 'model_params.txt'
    simulation_params_file_name = 'simulation_params.txt'

    network_params_path = os.path.join(parameters_path, network_params_file_name)
    model_params_path = os.path.join(parameters_path, model_params_file_name)
    simulation_params_path = os.path.join(parameters_path, simulation_params_file_name)


    network_params = json.load(open(network_params_path))
    networkmodel_params_params = json.load(open(model_params_path))
    simulation_params = json.load(open(simulation_params_path))
    
    return network_params, networkmodel_params_params, simulation_params

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

def cargo_vector(supplier_df):
    q = []
    q.append(0)
    for i in supplier_df['Total Gross Weight']:
        q.append(i)
    return q

def loading_vector(supplier_df):
    m = []
    m.append(0)
    for i in supplier_df['Calculated Loading Meters']:
        m.append(i)
    return m

def init_max_num_vehicles(n, q, max_weight):
    max_num_vehicles=int(n/3)
    while (max_num_vehicles-1)*max_weight < sum(q):
        max_num_vehicles += 1
    max_num_vehicles = n - math.ceil(math.sqrt(n/2))
    return max_num_vehicles

## ----------------------------------------------------------------------------------------------------------------------------------------
def periods_generator(simulation_period, simulation_interval, supplier_start_hr, pickup_end_hr):
    start = simulation_period[0]
    end = simulation_period[1]
    delta = datetime.timedelta(days=simulation_interval)

    start = datetime.datetime.strptime( start, '%Y-%m-%d' )
    end = datetime.datetime.strptime( end, '%Y-%m-%d' )
    t = start

    periods = []
    while t <= end :
        if t + delta < end:
            periods.append( [t.replace(hour=supplier_start_hr).strftime("%Y-%m-%d %H:%M:%S"), 
                             (t + delta).replace(hour=pickup_end_hr + simulation_interval).strftime("%Y-%m-%d %H:%M:%S") ] )    
            t = t + (delta + datetime.timedelta(days=1) )
        else:
            periods.append( [t.replace(hour=supplier_start_hr).strftime("%Y-%m-%d %H:%M:%S"), 
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

def filter_dates_old(df, column_name, init_p, endg_p):
    df_f = df.copy(deep=True)
    init_p = datetime.datetime.strptime(init_p, '%Y-%m-%d %H:%M:%S')
    endg_p = datetime.datetime.strptime(endg_p, '%Y-%m-%d %H:%M:%S')
    
    df_f['temp_date'] = pd.to_datetime(df_f[column_name])
    df_f=df_f[ (df_f['temp_date'] >=init_p) &  (df_f['temp_date'] <= endg_p) ]
    df_f.drop('temp_date', axis=1, inplace=True)
    df_f.sort_values(by=[column_name],inplace=True)   
    df_f.reset_index(drop=True, inplace=True)
    
    df_time = pd.to_datetime(df_f[column_name])
    
    # Week days -----------------------------------------------------------------
    nur_dates = df_time.dt.date.drop_duplicates()
    nur_dates.reset_index(drop=True, inplace=True)
    week_days = [date_obj.strftime('%A') for date_obj in nur_dates]
    week_days = pd.DataFrame(week_days, columns = ['Week Days'])
    display(pd.concat([nur_dates, week_days], axis=1))
    # Times ---------------------------------------------------------------------
    times_values = [dates_values.strftime("%H") for dates_values in df_time ]

    if len(times_values) != 0:
        print('\n')
        print('With time range:')
        print(' minimum = ', min(times_values), 'hrs' )
        print(' maximum = ', max(times_values), 'hrs' )
        
        min_time = min(df_f['Requested Loading'])
        max_time = max(df_f['Requested Loading'])
    else:
        min_time, max_time = [], []
        
    return df_f, min_time, max_time

def Time_set_wider_old(supplier_df, earliest_arrival, latest_arrival, discretization_constant ):
    min_date = min(supplier_df['Requested Loading']) 
    max_date = max(supplier_df['Requested Loading']) 

    min_dt = datetime.datetime.strptime(min_date, '%Y-%m-%d %H:%M:%S')
    max_dt = datetime.datetime.strptime(max_date, '%Y-%m-%d %H:%M:%S')

    d1 = min_dt - datetime.timedelta(hours=earliest_arrival) # Min picking time
    d2 = max_dt + datetime.timedelta(hours=latest_arrival) # Max arrival time to the depot

    ext_days = ((d2-d1).days+1)
    ext_hours = ext_days*24

    min_sup_arrival_time = int(d1.strftime("%H") )  # 24 hours before minimum arrival time on supplier
    max_dep_arrival_time = int(d2.strftime("%H") )  # 24 hours after maximum arrival time on supplier

    Tau_hours = np.arange(min_sup_arrival_time, 
                          max_dep_arrival_time + ext_hours + discretization_constant, 
                          discretization_constant)

    while len(Tau_hours[Tau_hours>=25]) != 0:
            next_day_length = len(Tau_hours[Tau_hours>=25])
            big_than = Tau_hours[Tau_hours>=25]
            exceed = big_than[0] -24
            Tau_hours[Tau_hours>=25] = np.arange(exceed, 
                                                 exceed + next_day_length*discretization_constant, 
                                                 discretization_constant)

    Tau_hours = list(Tau_hours)
    Tau = list(np.arange(0, len(Tau_hours), 1))

    return d1, d2, ext_days, Tau_hours, Tau   

## ----------------------------------------------------------------------------------------------------------------------------------------
def cr_nodes(supplier_df):    
    return list(np.arange( 0, len(supplier_df)+1, 1))

# Time Matrix ----------------------------------------------------------------------------
def discret_time_matrix_old(time_matrix, discretization_constant):    
    time_distance_matrix = time_matrix/(3600)
    time_distance_matrix = np.ceil(time_distance_matrix/discretization_constant)   # round up
    #time_distance_matrix = time_distance_matrix/discretization_constant           # round down
    #time_distance_matrix = np.round(time_distance_matrix,0)
    return time_distance_matrix.astype(int)

def discret_time_matrix(time_matrix, discretization_constant):    
    time_distance_matrix = time_matrix/(3600) # convert to hours
    
    non_zero_time_matrix = time_distance_matrix[time_distance_matrix != 0]
    current_discretization_constant = max( math.floor(np.min(non_zero_time_matrix)+1), discretization_constant)
    # current_discretization_constant = min(10, current_discretization_constant)
    time_distance_matrix = np.ceil(time_distance_matrix/current_discretization_constant) 
    return time_distance_matrix.astype(int), current_discretization_constant


def date_index_old(discretization_constant, e_day, e_hour, min_day, tour_days, Tau_hours, arrival_type):
    #if 24 % discretization_constant != 0:
    #    raise ValueError('check')
        
    number_points_per_day = int(24/discretization_constant)
    max_day = min_day + datetime.timedelta(days = tour_days + 1)
    
    # total_tour_days = np.arange(min_day, min_day+tour_days+1, 1)
    total_tour_days = pd.date_range(start=min_day,end=max_day)
    
    index_tour_day = np.where(np.array(total_tour_days.day) == e_day)[0][0]

    indx_to_consider= np.arange(number_points_per_day*index_tour_day, 
                                number_points_per_day*index_tour_day + number_points_per_day, 
                                1 )
    indx_to_consider = indx_to_consider[indx_to_consider < len(Tau_hours) ]
    indx_to_consider = list(indx_to_consider)


    min_tau_hr = np.array(Tau_hours)[indx_to_consider]
    min_tau_hr = list(min_tau_hr)


    if len(np.where(np.array(min_tau_hr) == e_hour)[0]) != 0:
        general_index = np.where(np.array(min_tau_hr) == e_hour)[0][0]
        
    else:
        r = e_hour
        if arrival_type == 'early_values':
            while len(np.where(np.array(min_tau_hr) == r)[0]) == 0:
                r = r + 1   
        elif arrival_type == 'latest_values':
            while len(np.where(np.array(min_tau_hr) == r)[0]) == 0:
                r = r - 1   
        else:
            raise ValueError('Please specifiy arrival type.')
            
            
        general_index = np.where(np.array(min_tau_hr) == r)[0][0]

    index_val = index_tour_day*number_points_per_day + general_index 

    return index_val

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

def inv_date_index_old(discretization_constant, index_to_check, min_date, Tau_hours):
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

    
# Post-processing functions ---------------------------------------------------------
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