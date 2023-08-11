import os
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from utils.utils import *
from ortools.linear_solver import pywraplp
from pyqubo import Array, Constraint, Placeholder


class CVRPTW: 
    def __init__(self, evaluation_period, discretization_constant, time_expanded_network, time_expanded_network_index, Tau_hours, distance_matrix, disc_time_distance_matrix, 
                 capacity_matrix, loading_matrix, max_capacity, max_ldms, max_driving, is_gap, mip_gap, maximum_minutes):
        log.info('Defining MIP model... ')
        # Safe problem structure
        self.evaluation_period = evaluation_period
        self.discretization_constant = discretization_constant
        self.time_expanded_network = time_expanded_network
        self.time_expanded_network_index = time_expanded_network_index
        self.Tau_hours = Tau_hours 
        self.distance_matrix = distance_matrix
        self.disc_time_distance_matrix = disc_time_distance_matrix
        self.capacity_matrix = capacity_matrix
        self.loading_matrix = loading_matrix

        self.des_max_driving = max_driving/discretization_constant

        self.length = len(self.distance_matrix) 
        self.max_num_vehicles = self.length - 1

        log.info('Number of nodes %i' % self.length)
        log.info('Solving for maximum number of %i vehicles...' % self.max_num_vehicles)

        self.max_capacity = max_capacity
        self.max_capacity_kg = [max_capacity*1000] * self.max_num_vehicles
        
        self.max_ldms = max_ldms
        self.max_ldms_vc = [max_ldms] * self.max_num_vehicles

        self.connections_solution = None
        self.vehicles_solution = None

        # Setup some solver configurations
        self.model = pywraplp.Solver('CVRPTW',pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)
        self.model.set_time_limit(maximum_minutes*60*1000)  # miliseconds

        self.is_gap = is_gap
        if is_gap == True:
            log.info('MIP GAP %s and maximum solving minutes %s...' %(mip_gap,maximum_minutes))            
            self.solverParams = pywraplp.MPSolverParameters()
            self.solverParams.SetDoubleParam(self.solverParams.RELATIVE_MIP_GAP, mip_gap) 
        

    def create_model(self, w):
        A_i, A_j, self.nodes = CVRPTW.nodes_range(self.time_expanded_network)
        all_duples, index_out, index_ins, index_zero_ins = CVRPTW.nodes_expanded_points(self.time_expanded_network)

        self._defining_variables()
        self._add_constraint_nodes()
        self._add_constraint_vehicle_routing(A_i, A_j, all_duples, index_out, index_ins, index_zero_ins)
        self._add_obj_function(w)

    def solve_model(self):        
        status = 1
        while status != 0:
            if  self.max_num_vehicles < self.length + 2 :
                if self.is_gap:
                    status = self.model.Solve(self.solverParams)
                else:
                    status = self.model.Solve()
                                
                self.max_num_vehicles += 2

                self.max_capacity_kg = [self.max_capacity*1000] * self.max_num_vehicles
                self.max_ldms_vc = [self.max_ldms] * self.max_num_vehicles
            else:    
                print('No solution found, last num. vehicles considered:', self.max_num_vehicles)
                break

            # self.max_capacity_kg = [self.max_capacity_kg] * self.max_num_vehicles 
            # self.max_ldms = [self.max_ldms] * self.max_num_vehicles

        return status, self.x, self.y

    def nodes_range(time_expanded_network):
        """Static function: Gives out the feasible space of the nodes given the Time-Expanded Network.
        Input: 
        time_expanded_network: Time-Expanded Network.
        Output: 
        A_i: Leaving Nodes to consider.
        A_j: Arriving Nodes to consider.
        all_duples: List of of the duples (Node-i,Time-t) on the Time-Expanded Network. Includes arrival and leaving points.
        index_zero_ins: Extracts the arrival time-network points to the Consignee.
        """
        A_i=[]
        A_j=[]
        for j in range(len(time_expanded_network)):
            A_i.append(time_expanded_network[j][0][0])
            A_j.append(time_expanded_network[j][1][0])
        all_nodes = list(set(A_i+A_j))

        return A_i, A_j, all_nodes

    def nodes_expanded_points(time_expanded_network):
        duples_1 = []
        duples_2 = []
        for j in range(len(time_expanded_network)):
                duples_1.append( time_expanded_network[j][0] )
                duples_2.append( time_expanded_network[j][1] )
        all_duples = duples_1 + duples_2
        all_duples = list(set(map(tuple, all_duples)))

        index_ins = {}
        index_out = {}
        for dups in all_duples:
            index_ins[dups] = []
            index_out[dups] = []
            k = 0
            for elem in time_expanded_network:
                if tuple(elem[1]) == dups:
                    if len(elem[0]) != 0:
                        index_ins[ dups ].append(k )
                elif tuple(elem[0]) == dups:
                    if len(elem[1]) != 0:
                        index_out[ dups ].append(k )            
                k += 1        
        index_zero_ins = []
        for vals in all_duples:
            if vals[0] == 0:        
                for j in index_ins[vals]:
                    index_zero_ins.append( [ [time_expanded_network[j][0][0], time_expanded_network[j][0][1]], [time_expanded_network[j][1][0], time_expanded_network[j][1][1]]] )
        
        return all_duples, index_out, index_ins, index_zero_ins

    def _defining_variables(self):
        log.info('Defining variables...')
        self.x = [[[[[self.model.IntVar(0,1,'') for t in self.time_expanded_network_index] for j in range(0, self.length)] for t in self.time_expanded_network_index] for i in range(0, self.length)] for k in range(self.max_num_vehicles)]         
        self.y = [self.model.IntVar(0,1,'') for k in range(self.max_num_vehicles)] 
    
    def _add_constraint_nodes(self):
        log.info('Adding Nodes Constraints...')        
        for i in self.nodes:
            if i != 0:                
                options_i = A_index(self.time_expanded_network, i, 'delta_out') 
                self.model.Add( sum( self.x[k][ options_i[j][0][0] ][ options_i[j][0][1] ][ options_i[j][1][0] ][ options_i[j][1][1] ] for j in range(len(options_i)) for k in range(self.max_num_vehicles) ) == 1 )

    def _add_constraint_vehicle_routing(self, A_i, A_j, all_duples, index_out, index_ins, index_zero_ins):       
        log.info('Adding Vehicle Routing Constraints...')    
        for k in range(self.max_num_vehicles):
            self.model.Add( sum( self.x[k][ j[0][0] ][ j[0][1] ][ j[1][0] ][ j[1][1] ] for j in index_zero_ins ) == self.y[k] )  # every vehicle has to be used and return to 0
            self.model.Add( sum( self.capacity_matrix[self.time_expanded_network[j][0][0]] * self.x[k][ self.time_expanded_network[j][0][0] ][ self.time_expanded_network[j][0][1] ][ self.time_expanded_network[j][1][0] ][ self.time_expanded_network[j][1][1] ] for j in range(0, len(self.time_expanded_network))) <= self.y[k] * self.max_capacity_kg[k] ) 
            for vals in all_duples:
                if vals[0] != 0:
                    self.model.Add( sum( self.x[k][ self.time_expanded_network[j][0][0] ][ self.time_expanded_network[j][0][1] ][ self.time_expanded_network[j][1][0] ][ self.time_expanded_network[j][1][1] ] for j in index_out[vals]) - sum( self.x[k][ self.time_expanded_network[j][0][0] ][self.time_expanded_network[j][0][1]][self.time_expanded_network[j][1][0]][self.time_expanded_network[j][1][1]] for j in index_ins[vals]) == 0 )                

    def _add_obj_function(self, w):
        log.info('Number of constraints = ' + str( self.model.NumConstraints() ) ) 
        log.info('Solving time-extended network MIP model...')

        number_nodes = len(self.distance_matrix)
        P = 0
        for i in range(number_nodes):
            P += self.distance_matrix[i][0]
        P = P/number_nodes
        # print('P', P)
        # print('w',w)

        self.model.Minimize( w * self.model.Sum( self.x[k][ self.time_expanded_network[i][0][0] ][ self.time_expanded_network[i][0][1] ][ self.time_expanded_network[i][1][0] ][ self.time_expanded_network[i][1][1] ]*self.distance_matrix[ self.time_expanded_network[i][0][0] ][ self.time_expanded_network[i][1][0] ] for i in range(len(self.time_expanded_network)) for k in range(self.max_num_vehicles)  ) +
                    (1 - w) * P *self.model.Sum( self.y[k] for k in range(self.max_num_vehicles))) 

        # self.model.Minimize( self.model.Sum( self.x[k][ self.time_expanded_network[i][0][0] ][ self.time_expanded_network[i][0][1] ][ self.time_expanded_network[i][1][0] ][ self.time_expanded_network[i][1][1] ]*self.distance_matrix[ self.time_expanded_network[i][0][0] ][ self.time_expanded_network[i][1][0] ] for i in range(len(self.time_expanded_network)) for k in range(self.max_num_vehicles)  ) ) 

        # self.model.Minimize( self.model.Sum( self.y[k] for k in range(self.max_num_vehicles)) ) 


    def read_solution(self, solution_path):
        current_solution = np.load(solution_path, allow_pickle=True)
        re_dict = current_solution.tolist()
        return re_dict

    def print_solution(self, connections_matrix, index_solution, discretization_constant, min_date, Tau_hours, distance_matrix, 
                   disc_time_distance_matrix, capacity_matrix, loading_matrix):        
        r = {}
        index={}
        dist={}
        driv={}
        cargo={}
        load={}

        total_dist=0
        total_driv=0
        total_cargo=0
        total_load=0
        vehicle_id=1
        vehicle_count = 1
        for k in index_solution:
            vehicle_id = k
            print('Vehicle %i:' % vehicle_count) 
            r[vehicle_id] = np.argwhere( np.array(connections_matrix[k]) == 1 )

            index[vehicle_id] = []
            dist[vehicle_id] = []
            driv[vehicle_id] = []
            cargo[vehicle_id] = []
            load[vehicle_id] = []
            forw_index=1
            prev_index = 0
            while forw_index != 0:      
                forw_index, forw_time = next_index(r[vehicle_id], prev_index)    
                date_time_overall, forw_time = inv_date_index(discretization_constant, forw_time, min_date, Tau_hours)

                print(prev_index, '->',  forw_index, 'Arrive on', forw_time)        
                index[vehicle_id].append(forw_index)
                dist[vehicle_id].append( distance_matrix[prev_index][forw_index])
                driv[vehicle_id].append( disc_time_distance_matrix[prev_index][forw_index])
                cargo[vehicle_id].append(capacity_matrix[forw_index])
                load[vehicle_id].append( loading_matrix[forw_index])

                prev_index = forw_index

            vehicle_count += 1
            total_dist += sum(dist[vehicle_id])
            total_driv += sum(driv[vehicle_id])
            total_cargo += sum(cargo[vehicle_id]) 
            total_load += sum(load[vehicle_id]) 

            print(' - Distance Vehicle %i:           '% vehicle_id, int(sum(dist[vehicle_id])), 'km')
            print(' - Total Driving Time Vehicle %i: '% vehicle_id, int(sum(driv[vehicle_id])*discretization_constant ), 'hrs')
            print(' - Cargo Vehicle %i:              '% vehicle_id, int(sum(cargo[vehicle_id])), 'kg' )
            print(' - L. Meters Vehicle %i:          '% vehicle_id, round(sum(load[vehicle_id]),2), 'm3' )
            # vehicle_id += 1
            print('')

        print('')
        if total_dist != 0:
            total_dist = int(round(total_dist,0))
            print('Total Distance %i km'%total_dist)
            print('Total Cargo %i kg'%total_cargo)
            print('Total Loading Meters %i m3'%round(total_load,2))

        # Compute distance saved.
        nodes_visited = []
        for i in index.keys():
            nodes_visited.extend(index[i])
        nodes_visited = np.unique(nodes_visited)

        before_dist = 0
        for i in nodes_visited:
            before_dist += distance_matrix[i][0]
        before_dist = int(round(before_dist,0))

        if total_dist == 0:
            print('No more routes to optimize.')
        else:
            print('Trivial distance:', before_dist, 'km')
            print('Distance reduction achieved:', round(( (before_dist - total_dist) /before_dist) *100,2), '% \n \n \n')

    def print_status(self, status, x, y):
        self.connections_solution = SolVal(x)
        self.vehicles_solution = SolVal(y)

        if status != pywraplp.Solver.INFEASIBLE:
            if status != pywraplp.Solver.OPTIMAL:
                logger.warning("Due to time constraint, the closer solution for optimality is given...")

            op_num_vehicles = int(sum(self.vehicles_solution))
            obj_value = round( self.model.Objective().Value(),2)

            log.info('Optimal solution found.')
            log.info('Objective value = ' + str( obj_value ) )
            log.info('Number of nodes = ' + str( len(self.distance_matrix)) )
            log.info('Number of vehicles selected = ' + str( op_num_vehicles ) )
            log.info('Total Distance = ' + str( obj_value - np.sum(self.vehicles_solution) ) )    
            log.info('Total Cargo = ' + str( int(sum( self.capacity_matrix)) ) )
            log.info('Total Loading Meters = ' + str( int(sum( self.loading_matrix )) ) )     
            
            self.secs_taken = round(int(self.model.wall_time())/1000,2)
            log.info('Problem solved in %s seconds' %  self.secs_taken) 
            log.info('Problem solved in %s minutes' %  str((self.secs_taken)/60) ) 
        else:
            logger.warning("The problem is infeasible.")
            print(self.time_expanded_network)

            obj_value = round( self.model.Objective().Value(),2)
            print(obj_value)           


        if status != pywraplp.Solver.INFEASIBLE:
            
            index_solution = information_index(self.y)
            CVRPTW.print_solution(self, self.connections_solution, index_solution, self.discretization_constant, 
                                    self.min_date, self.Tau_hours, self.distance_matrix, 
                                    self.disc_time_distance_matrix, self.capacity_matrix, 
                                    self.loading_matrix)


    def save_solution(self, path):
        # Store solution ----------------------------------------------------------------------------------------
        solution_dict = {}
        # Model
        solution_dict['period'] = self.evaluation_period #Paul:TODO: What exactly is this and where do we need to define it?
        solution_dict['discretization_constant'] = self.discretization_constant
        solution_dict['distance_matrix'] = self.distance_matrix
        solution_dict['disc_time_distance_matrix'] = self.disc_time_distance_matrix        
        # Solution
        solution_dict['time_needed'] = self.secs_taken
        solution_dict['index_solution'] = information_index(self.y)
        solution_dict['connections_matrix'] = self.connections_solution
        # Truck
        solution_dict['capacity_matrix'] = self.capacity_matrix
        solution_dict['loading_matrix'] = self.loading_matrix
        # Time 
        solution_dict['min_date'] = self.min_date 
        solution_dict['Tau_hours'] = self.Tau_hours 
        solution_dict['time_expand_network'] = self.time_expanded_network
        solution_dict['time_expand_network_index'] = self.time_expanded_network_index

        #moment = datetime.datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")        
        if not isinstance(self.evaluation_period, list):
            self.evaluation_period = self.evaluation_period.strftime('%Y-%m-%d')
            self.evaluation_period = [self.evaluation_period, self.evaluation_period]
            t_0 = datetime.datetime.strptime(self.evaluation_period[0], '%Y-%m-%d')
            t_0 = t_0.strftime('%m%d%Y')

            t_1 = datetime.datetime.strptime(self.evaluation_period[1], '%Y-%m-%d')
            t_1 = t_1.strftime('%m%d%Y')
        else:
            t_0 = datetime.datetime.strptime(self.evaluation_period[0], '%Y-%m-%d %H:%M:%S')
            t_0 = t_0.strftime('%m%d%Y')

            t_1 = datetime.datetime.strptime(self.evaluation_period[1], '%Y-%m-%d %H:%M:%S')
            t_1 = t_1.strftime('%m%d%Y')

        # Save
        save_name = os.path.join(path, 'solution' + str(self.discretization_constant) + '_' + str(t_0) + '-' + str(t_1) + '.npy' )        
        print('file saved:,', save_name)
        np.save(save_name, solution_dict)



     # Quantum Annealing:
    def _defining_hamiltonian_variables(self, A_i, A_j):
        log.info('Defining Hamiltonian Variables...')
            
        self.x_array = Array.create('connection', shape=(self.max_num_vehicles, len(np.unique(A_i)), len(self.time_expanded_network_index), len(np.unique(A_j)), \
                                        len(self.time_expanded_network_index)), vartype="BINARY")
        self.y_array = Array.create('vehicle', shape=(self.max_num_vehicles,), vartype="BINARY")

        self.capacity_matrix = np.array(self.capacity_matrix, dtype=np.int)          
        self.q_range = range(0, math.ceil(1 + math.log(self.max_capacity*1000 -
                                                       np.min(self.capacity_matrix[np.nonzero(self.capacity_matrix)]) , 2)))       
                     
        self.loading_matrix = np.array(self.loading_matrix, dtype=np.float32)
        self.m_range = range(0, math.ceil(1 + math.log(self.max_ldms -
                                                       np.min(self.loading_matrix[np.nonzero(self.loading_matrix)]) , 2))) 
        
        self.lambda_q = Array.create('slack_q', shape=(max(self.q_range) + 1, self.max_num_vehicles), vartype="BINARY")
        self.lambda_m = Array.create('slack_m', shape=(max(self.m_range) + 1, self.max_num_vehicles), vartype="BINARY")


    def _add_hamiltonian_min_function(self):
        log.info('Adding Hamiltonian Minimization Function (H0)...')    
        w=0.9
        
        H_0 = w*sum(self.x_array[k][ self.time_expanded_network[i][0][0] ][ int(self.time_expanded_network[i][0][1]) ][ int(self.time_expanded_network[i][1][0]) ][ int(self.time_expanded_network[i][1][1]) ] * \
                    self.distance_matrix[ self.time_expanded_network[i][0][0] ][ self.time_expanded_network[i][1][0] ] for i in range(len(self.time_expanded_network)) for k in range(self.max_num_vehicles)  ) + \
        (1-w)*self.P*sum(self.y_array[k] for k in range(self.max_num_vehicles))
        return H_0


    def _add_hamiltonian_constraints_nodes(self):
        log.info('Adding Hamiltonian Constraints Nodes (H1)...') 
        Nodes_0 = list(set(self.nodes) - {0})
        H_1 = {}
        for i in Nodes_0:
            options_i = A_index(self.time_expanded_network, i, 'delta_out') 
            H_1[i] = Constraint( (sum( self.x_array[k][ int(options_i[j][0][0]) ][ int(options_i[j][0][1]) ][ int(options_i[j][1][0]) ][ int(options_i[j][1][1]) ] \
                                for j in range(len(options_i)) for k in range(self.max_num_vehicles)) - 1 )**2, 'Route_Depo_End_Constraints')
        # Merging the constraint into one sum
        H_1 = sum(H_1[i] for i in Nodes_0)
        return H_1


    def _add_hamiltonian_constraint_vehicle_routing(self, all_duples, index_out, index_ins, index_zero_ins):       
        log.info('Adding Hamiltonian Vehicle Routing Constraints (H2, H3, H4, H5)...')    

        H_2 = {}
        H_3 = {}
        H_4 = {}
        H_5 = {}
        for k in range(self.max_num_vehicles):
            H_2[k] = {}
            for vals in all_duples:
                if vals[0] != 0:
                    H_2[k][vals] = Constraint( (sum( self.x_array[k][ int(self.time_expanded_network[j][0][0]) ][ int(self.time_expanded_network[j][0][1]) ][ int(self.time_expanded_network[j][1][0]) ][ int(self.time_expanded_network[j][1][1]) ] \
                                                for j in index_out[vals]) - \
                                                sum( self.x_array[k][ int(self.time_expanded_network[j][0][0]) ][ int(self.time_expanded_network[j][0][1]) ][ int(self.time_expanded_network[j][1][0]) ][ int(self.time_expanded_network[j][1][1]) ] \
                                                    for j in index_ins[vals]) )**2, 'Equilibrium_Constraints')
                    
            H_3[k] = Constraint( (sum( self.x_array[k][ int(j[0][0]) ][ int(j[0][1]) ][ int(j[1][0]) ][ int(j[1][1]) ] \
                                    for j in index_zero_ins) - self.y_array[k])**2, 'Vehicle_Return_Constraints')  # every vehicle has to be used and return to 0    

            H_4[k] = Constraint( (sum(self.capacity_matrix[ int(self.time_expanded_network[j][0][0]) ] * self.x_array[k][ int(self.time_expanded_network[j][0][0]) ][ int(self.time_expanded_network[j][0][1]) ][ int(self.time_expanded_network[j][1][0]) ][ int(self.time_expanded_network[j][1][1]) ] \
                                           for j in range(0,len(self.time_expanded_network)) ) + \
                         sum(self.lambda_q[l][k]*(2**((l))) for l in self.q_range) - \
                                  self.y_array[k]*self.max_capacity*1000 )**2, 'Capacity_Countraints')
                            
            H_5[k] = Constraint( (sum( self.loading_matrix[ int(self.time_expanded_network[j][0][0]) ] * self.x_array[k][ int(self.time_expanded_network[j][0][0]) ][ int(self.time_expanded_network[j][0][1]) ][ int(self.time_expanded_network[j][1][0]) ][ int(self.time_expanded_network[j][1][1]) ] \
                                           for j in range(0,len(self.time_expanded_network)) ) + \
                         sum(self.lambda_m[l][k]*(2**((l))) for l in self.m_range) - self.y_array[k]*self.max_ldms )**2, 'Loading_Countraints')
                      
        
        # Merging the constraint into sums
        H_2 = sum(H_2[k][vals] for k in range(self.max_num_vehicles) for vals in all_duples if vals[0] != 0 )

        H_3 = sum(H_3[k] for k in range(self.max_num_vehicles))
        H_4 = sum(H_4[k] for k in range(self.max_num_vehicles))
        H_5 = sum(H_5[k] for k in range(self.max_num_vehicles)) 
        return H_2, H_3, H_4, H_5
            

    def create_hamiltonian_model(self):
        log.info('Creating Hamiltonian Model (HF)...')   
        A_i, A_j, self.nodes = CVRPTW.nodes_range(self.time_expanded_network)
        all_duples, index_out, index_ins, index_zero_ins = CVRPTW.nodes_expanded_points(self.time_expanded_network)

        self._defining_hamiltonian_variables(A_i, A_j)
        H_0 = self._add_hamiltonian_min_function()
        H_1 = self._add_hamiltonian_constraints_nodes()
        H_2, H_3, H_4, H_5 = self._add_hamiltonian_constraint_vehicle_routing(all_duples, index_out, index_ins, index_zero_ins)

        # Penalization Terms
        B = Placeholder('B')        
        HF = H_0 + B*(H_1 + H_2 + H_3 + H_4 + H_5)

        log.info('Compiling HF...') 
        HF_model = HF.compile()
        
        log.info('HF Finished...') 
        return HF_model
