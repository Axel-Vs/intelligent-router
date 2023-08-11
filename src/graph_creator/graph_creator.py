import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from calendar import monthrange
import datetime
import copy
import os
import pandas as pd
import math
import warnings
import openrouteservice as ors

from utils.utils import *


warnings.simplefilter(action='ignore', category=FutureWarning)

class Network:
    """Create Network & Time-Expanded Network according to the period of time given.
    Input: Dictionary with all the business parameters / period
    """
    def __init__(self, params): 
        self.discretization_constant = params['discretization_constant']
        self.starting_depot = params['starting_depot']
        self.closing_depot = params['closing_depot']
        self.supplier_start_hr = params['supplier_start_hr']
        self.pickup_end_hr = params['pickup_end_hr']
        self.loading = params['loading']        
        self.earl_arv = params['earl_arv']   # Earliest arrival to supplier (days) 
        self.late_arv = params['late_arv']   # Latest arrival to supplier (days)
        self.max_driving = params['max_driving']
        self.driving_starts = params['driving_starts']
        self.driving_stop = params['driving_stop']
        self.max_weight = params['max_weight']
        self.max_ldms = params['max_ldms']
        self.plot_centered_coordinates = params['plot_centered_coordinates']      
        self.client = ors.Client(key= '5b3ce3597851110001cf6248448674063e0d4ec38216e52a54d951b5') # Specify your personal API key
        # Paul: We should have these parameters already defined so they are known
        self.distance_matrix = None
        self.time_distance_matrix = None
        self.disc_time_distance_matrix = None
        # self.capacity_matrix = None
        self.solution = None

    def export_network_to_csv(self, path = "./python_files/Networks/"):
        """Export Network as 4 files that fully define it.
        """    
        folder_name = datetime.datetime.now()+"/"
        path = path+folder_name
        os.mkdir(path)
        params_export_dict = {
            'discretization_constant':self.discretization_constant,
            'starting_depot':self.starting_depot,
            'closing_depot':self.closing_depot,
            'supplier_start_hr':self.supplier_start_hr,
            'pickup_end_hr':self.pickup_end_hr,
            'loading':self.loading,
            'max_driving':self.max_driving,
            'driving_starts':self.driving_starts,
            'driving_stop':self.driving_stop,
            'max_weight':self.max_weight,
            'max_ldms':self.max_ldms,
            'germany_coordinates':self.germany_coordinates
        }
        with open(path+'params.csv', 'w') as f:
            for key in params_export_dict.keys():
                f.write("%s,%s\n"%(key,params_export_dict[key]))
        self.distance_matrix.write_csv(path+"distance_matrix")
        self.time_distance_matrix.write_csv(path+"time_distance_matrix")
        self.capacity_matrix.write_csv(path+"capacity_matrix")
    
    #Paul:TODO: Implement
    def get_subset_of_network(self, first_day: int, last_day: int):
        """
        Get a subset of a network for the rolling horizon scheduling
        """
        return self.distance_matrix, self.time_distance_matrix, self.capacity_matrix, self.loading_matrix

    def update_wrt_solution(self, x: bool ,y: bool ,first_day: int, last_day: int, update_interval: int):
        """
        Updates the (partial) network w.r.t. the (partial) solution from the model
        input:
            x = Solution variable of model. Boolean vector with: k,i,t,j,t'
            y = tbd
            first_day = First day to consider
            last_day = Last day to consider, just as info, probably not needed?!
            update_interval = Consider first_day up to here.
        Output:
            None
        """
        # For all i, t, j, t' remove all connections but the one we took up until first_day+update_interval
        
        # Filter x for t'< first_day+update_interval
        # Only keep i, t, j, t' in time_expnanded_network that was actually taken (i.e. set to 1 in x)
        # -> For >= first_day+update_interval: Remove all connections to and from i's that have been visited
        # Same for Capacity_matrix, distance_matrix, loading_matrix

        # distance_matrix: i,j
        # capacity_matrix: i
        # disc_time_distance_matrix: i,j
        # time_expanded_network: i, t, j, t'

    def read_data(self,  period, initial_dataset):
        """Read pre-processing data, filter the period given and extracts coordinates and supplier dataset.
        Input: Period and dataset path.
        Output: List with coordinates and suppliers information dataframe
        """
        # initial_dataset = pd.read_csv(path)
        #---!
        # initial_dataset = initial_dataset[3:6]
        #---!


        # consignee_extract
        consignee_values = initial_dataset[['Consignee longitude', 'Consignee latitude']].apply(list, axis=1)

        # supplier_extrac
        if not isinstance(period, list):
            in_p = period.replace(hour=0, minute=0)
            en_p = period.replace(hour=23, minute=59)

            in_p = in_p.strftime('%Y-%m-%d %H:%M:%S')
            en_p = en_p.strftime('%Y-%m-%d %H:%M:%S')
            period = [in_p, en_p]

        filter_geocoded = filter_dates(initial_dataset, period)       
 
        filter_geocoded.reset_index(drop=True, inplace=True)
        df_coord = filter_geocoded[['Supplier longitude', 'Supplier latitude']]
        supplier_coordinates = df_coord.apply(list, axis=1)       # just coordinates 

        suppliers_df = filter_geocoded[['Supplier Name', 'Supplier longitude', 'Supplier latitude', 'Calculated Loading Meters', 'Total Gross Weight',  'Requested Loading', 'Requested Delivery']]
        suppliers_df.index = suppliers_df.index + 1
        suppliers_df = suppliers_df
        if len(suppliers_df) !=0:
            print('Number of suppliers extracted:', len(suppliers_df))   # complete dataset
        
        # consolidation
        complete_coordinates = consignee_values.head(1).append(supplier_coordinates)
        complete_coordinates = list(complete_coordinates)
        
        return complete_coordinates, suppliers_df
        

    def create_network(self, complete_coordinates, suppliers_df):
        """Creates Fix connection network.
        Input: Nodes coordinates, source index, openrouteservice connection, setting to zero parameter.
        Output: Distance and Time Travel Matrix
        """
        self.distance_matrix, self.time_distance_matrix = info_matrix_definition(complete_coordinates, 0, 'horizontal', self.client)
        self.Nodes = cr_nodes(suppliers_df)       
        self.length = len(self.distance_matrix)
        print('Distance & Time Distance Matrix created')

    def discretize(self):
        """Discretize Loading Time, Maximum Driving Hours, Time Distance Matrix.      
        """
        self.disc_time_distance_matrix, self.discretization_constant = discret_time_matrix(self.time_distance_matrix, self.discretization_constant)
        self.max_driving = np.max(  [self.max_driving] + [math.ceil(self.time_distance_matrix[i][0]/3600) for i in range(len(self.distance_matrix))] )
        
        print('Maximum hours allow to drive:', self.max_driving)
        print('New discretization constant allow to drive:', self.discretization_constant)

        self.disc_max_driving = self.max_driving/self.discretization_constant
        self.disc_loading = math.floor(self.loading/self.discretization_constant)

        print('Discretized objects: [Loading Time, Maximum Driving Hours, Time Distance Matrix]')


    def time_definition(discretization_constant, upper_time, lower_time):
            ext_days = ((upper_time-lower_time).days+1)
            ext_hours = ext_days*24
            # print('Number of days considered: ', ext_days)

            min_overall_time = int(lower_time.strftime("%H") )  
            max_dep_arrival_time = int(upper_time.strftime("%H") )   

            Tau_hours = np.arange(min_overall_time, 
                                    max_dep_arrival_time + ext_hours + discretization_constant, 
                                    discretization_constant)

            while len(Tau_hours[Tau_hours>=25]) != 0:
                    next_day_length = len(Tau_hours[Tau_hours>=25])
                    big_than = Tau_hours[Tau_hours>=25]
                    exceed = big_than[0] -24
                    Tau_hours[Tau_hours>=25] = np.arange(exceed, exceed + next_day_length*discretization_constant, discretization_constant)
            Tau_hours = list(Tau_hours)
            Tau = list(np.arange(0, len(Tau_hours), 1))

            if len(Tau) != len(Tau_hours):
                print('Length Tau', len(Tau))
                print('Lenght Tau_hours', len(Tau_hours))
                raise ValueError ('Indexes do not match!')
            return ext_days, Tau_hours, Tau


    def Time_set_wider(discretization_constant, earl_arv, late_arv, suppliers_df, local, init_simulation_date, end_simulation_date): 
        """Creates Time Index and Hour range vector of the given period. The interval [d1, d2] is the time space range that will be considered for arriving to the suppliers.
        Input: 
        discretization_constant: Discretization constant.
        earl_arv: Earliest arrival time to the supplier.
        late_arv: Latest arrival time to the supplier.
        suppliers_df: Suppliers DataFrame.
        Output: 
        d1: min_date for arriving during the period excluding the Consignee.
        d2: max_date for arriving during the period excluding the Consignee.
        ext_days: Complete range of arrivals including the Consignee [d1, d2 + 1].
        Tau_hours: Vector of arrival hours.
        Tau: Index of the Tau_hours vector.
        """
        min_date = min(suppliers_df['Requested Loading']) 
        max_date = max(suppliers_df['Requested Loading']) 
        maximum_consignee = suppliers_df['Requested Delivery']

        min_dt = datetime.datetime.strptime(min_date, '%d.%m.%Y %H:%M')
        max_dt = datetime.datetime.strptime(max_date, '%d.%m.%Y %H:%M')
        # should be 
        d0 = init_simulation_date
        d1 = min_dt - datetime.timedelta(days=earl_arv)
        d2 = max_dt + datetime.timedelta(days=late_arv + 1) # Max arrival time to the depot (48 hours after maximum arrival time on supplier)
        dn = end_simulation_date

        # Local Time Array
        if local == True:
            minimum_dt = d1
            ext_days, Tau_hours, Tau = Network.time_definition(discretization_constant, d2, d1)        
        else:
        # Extend Time Array
            minimum_dt = d0
            ext_days, Tau_hours, Tau = Network.time_definition(discretization_constant, dn, d0)

        return maximum_consignee, minimum_dt, ext_days, Tau_hours, Tau

    def create_time_network(self, suppliers_df, init_simulation_date, end_simulation_date):
        """Creates Time-Expanded Network
        Input: 
        suppliers_df: Suppliers DataFrame.
        Output: 
        T_ex: Time-Expanded Network.
        time_network_index: Index of the Tau_hours vector (expanded).
        """
        # 1) Establish the index boundaries of the Time-Network.
        maximum_consignee, self.min_date, self.tour_days, self.Tau_hours, Tau_index = Network.Time_set_wider(self.discretization_constant, self.earl_arv, self.late_arv, suppliers_df, True, init_simulation_date, end_simulation_date)
        self.min_day = self.min_date.day
        
        non_open_depot_index=np.where((np.array(self.Tau_hours)<self.starting_depot) | (np.array(self.Tau_hours)>self.closing_depot))
        non_open_supp_index=np.where((np.array(self.Tau_hours)<self.supplier_start_hr) | (np.array(self.Tau_hours)>self.pickup_end_hr))

        non_open_depot_index = list(non_open_depot_index[0])
        non_open_supp_index = list(non_open_supp_index[0])

        non_driving_index=np.where((np.array(self.Tau_hours)>self.driving_stop) | (np.array(self.Tau_hours)<self.driving_starts))
        non_driving_index=list(non_driving_index[0])
        # night_size = len(consecutive(non_driving_index)[0])
        night_size = max(list(map(len, consecutive(non_driving_index))))

        days_req = pd.to_datetime(suppliers_df['Requested Loading'], format= '%d.%m.%Y %H:%M').dt.day
        if self.late_arv <= 6:
            e = {}
            l = {}
            for i in range(1,len(days_req)+1):
                current_time = suppliers_df['Requested Loading'][i]

                current_time = datetime.datetime.strptime(current_time, '%d.%m.%Y %H:%M')
                upper_time_window = current_time + datetime.timedelta(hours=self.earl_arv)
                lower_time_window = current_time - datetime.timedelta(hours=self.late_arv)

                if upper_time_window.hour >= self.pickup_end_hr:
                    diff_hours = upper_time_window.hour - self.pickup_end_hr
                    if monthrange(current_time.year, current_time.month)[1] >= current_time.day + 1:
                        upper_time_window = datetime.datetime(current_time.year, current_time.month, current_time.day + 1, self.supplier_start_hr + diff_hours, 0, 0)
                    else:
                        upper_time_window = datetime.datetime(current_time.year, current_time.month + 1, current_time.day + 1 - monthrange(current_time.year, current_time.month)[1], self.supplier_start_hr + diff_hours, 0, 0)

                if lower_time_window.hour < self.supplier_start_hr:
                    diff_hours = self.supplier_start_hr - lower_time_window.hour
                    lower_time_window = datetime.datetime(current_time.year, current_time.month, current_time.day -1, self.pickup_end_hr - diff_hours, 0, 0)

                e[i] = [lower_time_window.day, int(lower_time_window.strftime("%H") )]
                l[i] = [upper_time_window.day, int(upper_time_window.strftime("%H") )]
        elif self.late_arv == 24:
            # 2) Defines the time Windows for each Supplier.
            e = {}
            l = {}
            print(suppliers_df)
            for i in days_req.index:
                current_time = suppliers_df['Requested Loading'][i]
                current_time = datetime.datetime.strptime(current_time, '%Y-%m-%d %H:%M:%S')
            
                lower_date = current_time - datetime.timedelta(hours=self.earl_arv)
                upper_date = current_time + datetime.timedelta(hours=self.late_arv)
                
                lower_date = int(lower_date.strftime("%d"))
                upper_date = int(upper_date.strftime("%d"))
                arrv_time = int(current_time.strftime("%H") )
                
                e[i] = [lower_date, arrv_time]  # one day before allow
                l[i] = [upper_date, arrv_time ] # one day after allow
        else:
            raise ValueError('Arrival time not supported')

        e_inx = {}
        l_inx = {}
        e_inx[0] = 0
        l_inx[0] = max(Tau_index)
        for i in e.keys():
            e_inx[i] = date_index(self.discretization_constant, e[i][0], e[i][1], self.min_date, self.tour_days, self.Tau_hours, 'early_values', maximum_consignee[i])
            l_inx[i] = date_index(self.discretization_constant, l[i][0], l[i][1], self.min_date, self.tour_days, self.Tau_hours, 'latest_values', maximum_consignee[i])
            
        time_windows={}
        for i in self.Nodes:
            time_windows[i] = list(np.arange( e_inx[i], l_inx[i]+1, 1))
        
        # 3) Creates the Expanded-Time Network.
        log.info('Creating Time-Expanded Network...')
        T_ex = []
        maximum_values=[]
        for i in range(self.length):
            for j in range(self.length):        
                if i !=j:
                    if i == 0:
                        for u in time_windows[j]:
                            if (u not in non_open_supp_index) & (u <= max(Tau_index)):                        
                                T_ex.append([ [i, u ], [j, u ] ])
                                #T_ex.append([ [j, u + s_d ], [0, u + disc_time_distance_matrix[i][j]  ] ])
                                
                    elif j == 0:
                            for m in time_windows[i]:                                     
                                t_e = m + self.disc_loading + self.disc_time_distance_matrix[i][j]                                                        
                                # if min(abs(t_e - non_driving_index)) < 1:
                                if (t_e in non_driving_index): 
                                    t_e = t_e + night_size            

                                # if (min(abs(t_e - non_open_depot_index)) >= 1) &  (t_e <= max(Tau_index)):                                                      
                                if (t_e not in non_open_depot_index) & (t_e <= max(Tau_index)):                                    
                                    # if min(abs(m - non_open_supp_index)) >= 1:
                                    if m not in non_open_supp_index:
                                        T_ex.append([ [i, m ], [j, t_e] ] )              
                                elif t_e >= max(Tau_index):
                                    # if min(abs(m - non_open_supp_index)) >= 1:
                                    if m not in non_open_supp_index:
                                        T_ex.append( [ [i, m], [j, max(Tau_index)] ] ) # arrive to the depot at the next day
                                        maximum_values.append(t_e)                        
                    else:
                        for m in time_windows[i]:
                            if self.disc_time_distance_matrix[i][j] == 0:                                
                                t_e = m + (self.disc_loading - math.floor( self.disc_loading/2 )) + self.disc_time_distance_matrix[i][j]        
                            else:
                                t_e = m + self.disc_loading + self.disc_time_distance_matrix[i][j]   #
                                
                            # if min(abs(t_e - non_driving_index)) < 1:
                            if (t_e in non_driving_index):                               
                                t_e = t_e + night_size
                                
                            if self.disc_time_distance_matrix[i][j] <= self.disc_max_driving:  
                                # if (min(abs(t_e - non_open_supp_index)) >= 1) & (t_e <= max(Tau_index)):                                                 
                                if (t_e not in non_open_supp_index) & (t_e <= max(Tau_index)):                                                 
                                    if t_e <= max(time_windows[j]):
                                        if self.disc_time_distance_matrix[i][j] <= self.disc_max_driving:
                                            # if min(abs(m - non_open_supp_index)) >= 1:
                                            if m not in non_open_supp_index:
                                                T_ex.append( [ [i, m ], [j,  int(max( time_windows[i][0], t_e) )] ])
        # Sanity Checks:
        for i in range(len(T_ex)):
            if T_ex[i][1][1] - T_ex[i][0][1] < 0:
                print(i)
                print('Your going backwards')
                if T_ex[i][0][0] != 0:
                    print('Teletransport!')
                    print(i)

        # 4) Add time points above the +1 day given in [d1, d2 + 1] if one day is not enough (given the travel time from supplier i to the Consignee).
        time_index_values = []
        for i in range(len(T_ex)):
            time_index_values.append(T_ex[i][1][1] )
        maximum_values=max(time_index_values)
        minimum_values=min(time_index_values)
        # range()
        time_network_index = np.arange(0, maximum_values + 1, 1)
        
        date_time_overall, date_time_network_min = inv_date_index(self.discretization_constant, minimum_values, self.min_date, self.Tau_hours)
        date_time_overall, date_time_network_max = inv_date_index(self.discretization_constant, maximum_values, self.min_date, self.Tau_hours)
        
        print('length time-expanded:                                 ', len(T_ex))
        print('Minimum arrival time in the time-expanded matrix is on', date_time_network_min )
        print('Maximum arrival time in the time-expanded matrix is on', date_time_network_max )

        return T_ex, Tau_index, time_network_index

    def shift_time(self, index_to_shift, suppliers_df, init_simulation_date, end_simulation_date):
        # shift index to the overall time index matrix
        maximum_consignee, min_date_overall, tour_days, Tau_hours_overall, Tau_index_overall = Network.Time_set_wider(self.discretization_constant, self.earl_arv, self.late_arv, suppliers_df, False, init_simulation_date, end_simulation_date)
    
        current_time = self.min_date
        arrv_day = current_time.day
        arrv_time = int(current_time.strftime("%H") )
        return index_to_shift + date_index(self.discretization_constant, arrv_day, arrv_time, min_date_overall, tour_days, Tau_hours_overall, 'early_values')

    def update_time_expanded_network(self, time_expanded_network_local, suppliers_df, init_simulation_date, end_simulation_date):
        time_expanded_network_general = copy.deepcopy(time_expanded_network_local)
        shift_value = Network.shift_time(self, 0, suppliers_df, init_simulation_date, end_simulation_date) 
        for i in range(len(time_expanded_network_local)):
            time_expanded_network_general[i][0][1] = time_expanded_network_local[i][0][1] + shift_value
            time_expanded_network_general[i][1][1] = time_expanded_network_local[i][1][1] + shift_value

        return time_expanded_network_general