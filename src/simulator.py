import sys
from pathlib import Path

# Get the parent directory of the current script (project root)
project_root = Path(__file__).resolve().parent.parent

# Add the paths of the relevant directories to sys.path
sys.path.append(str(project_root / 'src'))

# Import the modules from their respective directories
from graph_creator.graph_creator import Network
from model.delivery_model import CVRPTW


# import sys
# from pathlib import Path

# project_root = Path(__file__).resolve().parent.parent
# print('project_root', project_root)
# sys.path.append(str(project_root))


# # Import required modules from their respective directories
# from graph_creator.graph_creator import Network
# from model import delivery_model
from utils.utils import *
import pandas as pd
import os

# Get the current working directory
main_root = os.getcwd()

# Define paths for configuration files, data, and results
parameters_path = os.path.join(main_root, 'src/config')
data_path = os.path.join(main_root, 'data/delivery_example.csv')  # Warning: File dates format '%d.%m.%yyyy %H:%M'
results_path = os.path.join(main_root, 'results/optimization')

# Print initial message
print('\n')
print('Starting Simulation -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------')
print('\n')

# Import parameters for the simulation from configuration files
network_params, model_params, simulation_params = utils.import_parameters(parameters_path)

# Generate time periods for the simulation
periods = utils.periods_generator(simulation_params["Simulation_periods"],
                                  simulation_params["planning_horizon"],
                                  network_params['supplier_start_hr'],
                                  network_params['pickup_end_hr'])

# Iterate through different weight values
for w in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
    print('-------------------------------weight:', w)

    # Read geocoded data from CSV file
    germany_coordinates = [51.1657, 10.4515]
    df_geocoded = pd.read_csv(data_path, sep=';')
    consignee_values = df_geocoded[['Consignee longitude', 'Consignee latitude']].apply(list, axis=1)

    # Iterate through simulation periods
    for period in periods:
        # Create a network object
        net = Network(network_params)

        # Read data and create network for the given period
        complete_coordinates, suppliers_df = net.read_data([period[0], period[1]], df_geocoded)

        if len(suppliers_df) in range(1, 15):  # Check if orders are done in that period
            print('\n Time Frame Definition ---------------------------------------------------------------------------------------------------------------------------------------------------')
            print('Initial Simulation Date:', period[0])
            print('End Simulation Date:    ', period[1])
            print('Length suppliers:', len(suppliers_df))

            print('\n Create Network ----------------------------------------------------------------------------------------------------------------------------------------------------------')
            # Create and discretize the network
            net.create_network(complete_coordinates, suppliers_df)
            net.discretize()
            time_expanded_network, complete_time_index, time_expanded_network_index = net.create_time_network(suppliers_df, period[0], period[1])

            # Create cargo and loading matrices
            capacity_matrix = utils.cargo_vector(suppliers_df)
            loading_matrix = utils.loading_vector(suppliers_df)

            print('\n Solving -----------------------------------------------------------------------------------------------------------------------------------------------------------------')
            # Create the CVRPTW model and solve it
            cvrptw = CVRPTW([period[0], period[1]], net.discretization_constant, time_expanded_network, time_expanded_network_index, net.Tau_hours, net.distance_matrix,
                            net.disc_time_distance_matrix, capacity_matrix, loading_matrix, max_capacity=net.max_weight, max_ldms=net.max_ldms, max_driving=network_params["max_driving"],
                            is_gap=model_params['gap'], mip_gap=model_params["gap_value"], maximum_minutes=model_params['max_time'])
            cvrptw.min_date = net.min_date  # Set the minimum date for printing the solution

            # Create and solve the model
            cvrptw.create_model(w)
            status, connections_solution, vehicles_solution = cvrptw.solve_model()  # Get solutions

            # Print the solution status and save the solution
            cvrptw.print_status(status, connections_solution, vehicles_solution)
            cvrptw.save_solution(results_path)

            print('\n Finish Iteration -----------------------------------------------------------------------------------------------------------------------------------------------------------------')
            print('')

