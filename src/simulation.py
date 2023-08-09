from graph_creator import Network
from delivery_model import CVRPTW

import utils
import pandas as pd
import os

main_root = os.getcwd()

parameters_path = os.path.join(main_root, 'src/config')
data_path = os.path.join(main_root,'data/delivery_example.csv') # Warning: File dates format '%d.%m.%yyyy %H:%M'
results_path = os.path.join(main_root,'results/optimization')



print('\n')
print('Starting Simulation -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------')
print('\n')

# Import parameters
network_params, model_params, simulation_params = utils.import_parameters(parameters_path)

# Define the time frame for the simulation
periods = utils.periods_generator(simulation_params["Simulation_periods"], 
                                    simulation_params["planning_horizon"],                                    
                                    network_params['supplier_start_hr'],
                                    network_params['pickup_end_hr'])


for w in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
# for w in [1]:
    print('-------------------------------weight:', w)
    # Data Lecture ---------------------------------------------------------------------------------------
    germany_coordinates = [51.1657,10.4515]
    df_geocoded = pd.read_csv(data_path, sep=';')

    consignee_values = df_geocoded[['Consignee longitude', 'Consignee latitude']].apply(list, axis=1)

    for period in periods:    
        # Network Creation ------------------------------------------------------------------------------------------------------------------------------------------------------------------
        net = Network(network_params)
        complete_coordinates, suppliers_df = net.read_data([period[0], period[1]], df_geocoded)

        if len(suppliers_df) in range(1,15): # check for orders done in that period            
            print('\n Time Frame Definition ---------------------------------------------------------------------------------------------------------------------------------------------------')
            print('Initial Simulation Date:', period[0])
            print('End Simulation Date:    ', period[1])      
            print('Length suppliers:', len(suppliers_df))

            print('\n Create Network ----------------------------------------------------------------------------------------------------------------------------------------------------------')        
            net.create_network(complete_coordinates, suppliers_df)
            net.discretize()
            time_expanded_network, complete_time_index, time_expanded_network_index = net.create_time_network(suppliers_df, period[0], period[1])

            capacity_matrix = utils.cargo_vector(suppliers_df)
            loading_matrix  = utils.loading_vector(suppliers_df)

            print('\n Solving -----------------------------------------------------------------------------------------------------------------------------------------------------------------')
            cvrptw = CVRPTW([period[0], period[1]], net.discretization_constant, time_expanded_network, time_expanded_network_index, net.Tau_hours, net.distance_matrix, 
                            net.disc_time_distance_matrix, capacity_matrix, loading_matrix, max_capacity = net.max_weight, max_ldms = net.max_ldms, max_driving = network_params["max_driving"], 
                            is_gap = model_params['gap'], mip_gap=model_params["gap_value"], maximum_minutes=model_params['max_time'])

            cvrptw.min_date = net.min_date  # we need the min date only for printing the solution        
            
            # Classic
            cvrptw.create_model(w)
            status, connections_solution, vehicles_solution = cvrptw.solve_model() # solutions x:connections_solution and y:vehicles_solution

            cvrptw.print_status(status, connections_solution, vehicles_solution)     
            cvrptw.save_solution(results_path)

            print('\n Finish Iteration -----------------------------------------------------------------------------------------------------------------------------------------------------------------')
            print('')
