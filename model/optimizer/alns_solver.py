"""
Adaptive Large Neighborhood Search (ALNS) solver for VRP.
Uses destroy and repair operators on route-based representations.
"""

import numpy as np
import random
import copy
from .route_solution import RouteSolution


class ALNSSolver:
    """
    Adaptive Large Neighborhood Search for Vehicle Routing Problem.
    
    Much faster than MIP for large instances (50+ vendors).
    Uses route-based representation instead of time-expanded binary tensors.
    """
    
    def __init__(self, vendors_df, distance_matrix, time_matrix, 
                 capacity_matrix, loading_matrix, max_capacity_kg, max_ldms_vc,
                 discretization_constant, min_date, config=None):
        """
        Initialize ALNS solver.
        
        Args:
            vendors_df: DataFrame with vendor information
            distance_matrix: Distance matrix [km]
            time_matrix: Time matrix [seconds]
            capacity_matrix: Cargo weight per vendor [kg]
            loading_matrix: Loading volume per vendor [m³]
            max_capacity_kg: Max weight capacity per vehicle [kg]
            max_ldms_vc: Max volume capacity per vehicle [m³]
            discretization_constant: Time discretization [hours]
            min_date: Minimum simulation date
            config: Dictionary with ALNS parameters
        """
        self.vendors_df = vendors_df
        self.distance_matrix = distance_matrix
        self.time_matrix = time_matrix
        self.capacity_matrix = capacity_matrix
        self.loading_matrix = loading_matrix
        self.max_capacity_kg = max_capacity_kg
        self.max_ldms_vc = max_ldms_vc
        self.discretization_constant = discretization_constant
        self.min_date = min_date
        
        # ALNS parameters
        self.config = config or {}
        self.max_iterations = self.config.get('max_iterations', 1000)
        self.min_removal_size = self.config.get('min_removal_size', 0.1)  # 10% of vendors
        self.max_removal_size = self.config.get('max_removal_size', 0.4)  # 40% of vendors
        self.initial_temperature = self.config.get('initial_temperature', 1000)
        self.cooling_rate = self.config.get('cooling_rate', 0.995)
        
        # Operator weights (adaptive)
        self.destroy_operators = {
            'random': {'weight': 1.0, 'calls': 0, 'improvements': 0},
            'worst_cost': {'weight': 1.0, 'calls': 0, 'improvements': 0},
            'shaw': {'weight': 1.0, 'calls': 0, 'improvements': 0}
        }
        
        self.repair_operators = {
            'greedy': {'weight': 1.0, 'calls': 0, 'improvements': 0},
            'regret2': {'weight': 1.0, 'calls': 0, 'improvements': 0}
        }
        
        self.num_vendors = len(capacity_matrix) - 1  # Exclude depot
    
    def solve(self, verbose=True):
        """
        Run ALNS algorithm.
        
        Args:
            verbose: Print progress information
            
        Returns:
            RouteSolution: Best solution found
        """
        if verbose:
            print(f'\n🔍 Running ALNS metaheuristic solver')
            print(f'   - Max iterations: {self.max_iterations}')
            print(f'   - Vendors: {self.num_vendors}')
        
        # Generate initial solution
        current = self.generate_initial_solution()
        best = current.copy()
        
        if verbose:
            print(f'   - Initial solution: {current.get_num_routes()} routes, {current.evaluate():.0f} km')
        
        temperature = self.initial_temperature
        no_improvement_count = 0
        
        for iteration in range(self.max_iterations):
            # Select operators adaptively
            destroy_op = self.select_operator(self.destroy_operators)
            repair_op = self.select_operator(self.repair_operators)
            
            # Destroy
            removal_size = random.uniform(self.min_removal_size, self.max_removal_size)
            num_remove = max(1, int(self.num_vendors * removal_size))
            destroyed, removed_vendors = self.destroy(current, destroy_op, num_remove)
            
            # Repair
            repaired = self.repair(destroyed, removed_vendors, repair_op)
            
            # Acceptance criterion (simulated annealing)
            current_cost = current.evaluate()
            repaired_cost = repaired.evaluate()
            delta = repaired_cost - current_cost
            
            accept = False
            if delta < 0:
                # Improvement
                accept = True
                self.update_operator_weights(destroy_op, repair_op, reward=3)
            elif random.random() < np.exp(-delta / temperature):
                # Accept worse solution probabilistically
                accept = True
                self.update_operator_weights(destroy_op, repair_op, reward=1)
            
            if accept:
                current = repaired
                no_improvement_count = 0
                
                if repaired_cost < best.evaluate():
                    best = repaired.copy()
                    if verbose and iteration % 100 == 0:
                        print(f'   - Iteration {iteration}: New best {best.evaluate():.0f} km ({best.get_num_routes()} routes)')
            else:
                no_improvement_count += 1
            
            # Cool down temperature
            temperature *= self.cooling_rate
            
            # Early stopping if no improvement for long time
            if no_improvement_count > 200:
                break
        
        if verbose:
            print(f'   - Final solution: {best.get_num_routes()} routes, {best.evaluate():.0f} km')
            print(f'   - Feasible: {best.is_feasible(check_all=False)}')
        
        return best
    
    def generate_initial_solution(self):
        """Generate initial solution using greedy nearest neighbor."""
        routes = []
        unvisited = set(range(1, self.num_vendors + 1))
        
        while unvisited:
            route = [0]  # Start at depot
            current = 0
            route_weight = 0
            route_volume = 0
            vehicle_idx = len(routes)
            
            max_weight = self.max_capacity_kg[vehicle_idx] if vehicle_idx < len(self.max_capacity_kg) else float('inf')
            max_volume = self.max_ldms_vc[vehicle_idx] if vehicle_idx < len(self.max_ldms_vc) else float('inf')
            
            while unvisited:
                # Find nearest feasible vendor
                best_vendor = None
                best_distance = float('inf')
                
                for vendor in unvisited:
                    # Check capacity
                    if route_weight + self.capacity_matrix[vendor] > max_weight:
                        continue
                    if route_volume + self.loading_matrix[vendor] > max_volume:
                        continue
                    
                    # Check distance
                    distance = self.distance_matrix[current][vendor]
                    if distance < best_distance:
                        best_distance = distance
                        best_vendor = vendor
                
                if best_vendor is None:
                    break
                
                route.append(best_vendor)
                unvisited.remove(best_vendor)
                current = best_vendor
                route_weight += self.capacity_matrix[best_vendor]
                route_volume += self.loading_matrix[best_vendor]
            
            route.append(0)  # Return to depot
            routes.append(route)
        
        return RouteSolution(
            routes=routes,
            vendors_df=self.vendors_df,
            distance_matrix=self.distance_matrix,
            time_matrix=self.time_matrix,
            capacity_matrix=self.capacity_matrix,
            loading_matrix=self.loading_matrix,
            max_capacity_kg=self.max_capacity_kg,
            max_ldms_vc=self.max_ldms_vc,
            discretization_constant=self.discretization_constant,
            min_date=self.min_date
        )
    
    def destroy(self, solution, operator, num_remove):
        """
        Destroy operators: remove vendors from solution.
        
        Returns:
            RouteSolution: Partial solution
            list: Removed vendors
        """
        destroyed = solution.copy()
        removed = []
        
        if operator == 'random':
            removed = self.destroy_random(destroyed, num_remove)
        elif operator == 'worst_cost':
            removed = self.destroy_worst_cost(destroyed, num_remove)
        elif operator == 'shaw':
            removed = self.destroy_shaw(destroyed, num_remove)
        
        return destroyed, removed
    
    def destroy_random(self, solution, num_remove):
        """Random removal destroy operator."""
        all_vendors = []
        for route in solution.routes:
            all_vendors.extend([v for v in route if v != 0])
        
        removed = random.sample(all_vendors, min(num_remove, len(all_vendors)))
        
        # Remove from routes
        for route in solution.routes:
            route[:] = [v for v in route if v not in removed or v == 0]
        
        # Remove empty routes
        solution.routes = [r for r in solution.routes if len(r) > 2]
        solution.invalidate_cache()
        
        return removed
    
    def destroy_worst_cost(self, solution, num_remove):
        """Remove vendors with highest cost contribution."""
        vendor_costs = []
        
        for route in solution.routes:
            for i, vendor in enumerate(route):
                if vendor == 0:
                    continue
                
                # Calculate cost of removing this vendor
                prev_node = route[i-1] if i > 0 else 0
                next_node = route[i+1] if i < len(route)-1 else 0
                
                current_cost = self.distance_matrix[prev_node][vendor] + self.distance_matrix[vendor][next_node]
                direct_cost = self.distance_matrix[prev_node][next_node]
                savings = current_cost - direct_cost
                
                vendor_costs.append((vendor, savings))
        
        # Sort by worst savings (highest cost)
        vendor_costs.sort(key=lambda x: x[1], reverse=True)
        removed = [v for v, _ in vendor_costs[:num_remove]]
        
        # Remove from routes
        for route in solution.routes:
            route[:] = [v for v in route if v not in removed or v == 0]
        
        solution.routes = [r for r in solution.routes if len(r) > 2]
        solution.invalidate_cache()
        
        return removed
    
    def destroy_shaw(self, solution, num_remove):
        """Shaw removal: remove similar vendors (by distance)."""
        # Pick random seed vendor
        all_vendors = []
        for route in solution.routes:
            all_vendors.extend([v for v in route if v != 0])
        
        if not all_vendors:
            return []
        
        seed = random.choice(all_vendors)
        
        # Calculate relatedness (inverse distance)
        relatedness = [(v, 1.0 / (self.distance_matrix[seed][v] + 1)) for v in all_vendors if v != seed]
        relatedness.sort(key=lambda x: x[1], reverse=True)
        
        removed = [seed] + [v for v, _ in relatedness[:num_remove-1]]
        
        # Remove from routes
        for route in solution.routes:
            route[:] = [v for v in route if v not in removed or v == 0]
        
        solution.routes = [r for r in solution.routes if len(r) > 2]
        solution.invalidate_cache()
        
        return removed
    
    def repair(self, solution, removed_vendors, operator):
        """
        Repair operators: reinsert removed vendors.
        
        Returns:
            RouteSolution: Repaired solution
        """
        if operator == 'greedy':
            return self.repair_greedy(solution, removed_vendors)
        elif operator == 'regret2':
            return self.repair_regret2(solution, removed_vendors)
        
        return solution
    
    def repair_greedy(self, solution, removed_vendors):
        """Greedy insertion: insert each vendor at best position."""
        for vendor in removed_vendors:
            best_cost = float('inf')
            best_route_idx = -1
            best_position = -1
            
            # Try inserting in existing routes
            for route_idx, route in enumerate(solution.routes):
                for pos in range(1, len(route)):
                    # Check capacity
                    route_weight, route_volume = solution.get_route_capacity_usage(route_idx)
                    vehicle_idx = route_idx
                    max_weight = self.max_capacity_kg[vehicle_idx] if vehicle_idx < len(self.max_capacity_kg) else float('inf')
                    max_volume = self.max_ldms_vc[vehicle_idx] if vehicle_idx < len(self.max_ldms_vc) else float('inf')
                    
                    if route_weight + self.capacity_matrix[vendor] > max_weight:
                        continue
                    if route_volume + self.loading_matrix[vendor] > max_volume:
                        continue
                    
                    # Calculate insertion cost
                    prev_node = route[pos-1]
                    next_node = route[pos]
                    
                    current_cost = self.distance_matrix[prev_node][next_node]
                    new_cost = self.distance_matrix[prev_node][vendor] + self.distance_matrix[vendor][next_node]
                    insertion_cost = new_cost - current_cost
                    
                    if insertion_cost < best_cost:
                        best_cost = insertion_cost
                        best_route_idx = route_idx
                        best_position = pos
            
            # Insert at best position or create new route
            if best_route_idx >= 0:
                solution.routes[best_route_idx].insert(best_position, vendor)
            else:
                # Create new route
                solution.routes.append([0, vendor, 0])
        
        solution.invalidate_cache()
        return solution
    
    def repair_regret2(self, solution, removed_vendors):
        """Regret-2 insertion: prioritize vendors with large regret."""
        uninserted = set(removed_vendors)
        
        while uninserted:
            best_regret = -float('inf')
            best_vendor = None
            best_route_idx = -1
            best_position = -1
            
            for vendor in uninserted:
                # Find best and second-best insertion positions
                costs = []
                
                for route_idx, route in enumerate(solution.routes):
                    for pos in range(1, len(route)):
                        # Check capacity
                        route_weight, route_volume = solution.get_route_capacity_usage(route_idx)
                        vehicle_idx = route_idx
                        max_weight = self.max_capacity_kg[vehicle_idx] if vehicle_idx < len(self.max_capacity_kg) else float('inf')
                        max_volume = self.max_ldms_vc[vehicle_idx] if vehicle_idx < len(self.max_ldms_vc) else float('inf')
                        
                        if route_weight + self.capacity_matrix[vendor] > max_weight:
                            continue
                        if route_volume + self.loading_matrix[vendor] > max_volume:
                            continue
                        
                        # Calculate insertion cost
                        prev_node = route[pos-1]
                        next_node = route[pos]
                        
                        current_cost = self.distance_matrix[prev_node][next_node]
                        new_cost = self.distance_matrix[prev_node][vendor] + self.distance_matrix[vendor][next_node]
                        insertion_cost = new_cost - current_cost
                        
                        costs.append((insertion_cost, route_idx, pos))
                
                if len(costs) >= 2:
                    costs.sort(key=lambda x: x[0])
                    regret = costs[1][0] - costs[0][0]  # Second best - best
                    
                    if regret > best_regret:
                        best_regret = regret
                        best_vendor = vendor
                        best_route_idx = costs[0][1]
                        best_position = costs[0][2]
                elif len(costs) == 1:
                    if best_vendor is None:
                        best_vendor = vendor
                        best_route_idx = costs[0][1]
                        best_position = costs[0][2]
            
            if best_vendor is None:
                # Create new route for remaining vendors
                for vendor in uninserted:
                    solution.routes.append([0, vendor, 0])
                break
            
            solution.routes[best_route_idx].insert(best_position, best_vendor)
            uninserted.remove(best_vendor)
        
        solution.invalidate_cache()
        return solution
    
    def select_operator(self, operators):
        """Select operator based on adaptive weights."""
        total_weight = sum(op['weight'] for op in operators.values())
        rand = random.uniform(0, total_weight)
        
        cumulative = 0
        for name, op in operators.items():
            cumulative += op['weight']
            if rand <= cumulative:
                op['calls'] += 1
                return name
        
        return list(operators.keys())[0]
    
    def update_operator_weights(self, destroy_op, repair_op, reward):
        """Update operator weights based on performance."""
        self.destroy_operators[destroy_op]['improvements'] += reward
        self.repair_operators[repair_op]['improvements'] += reward
        
        # Adaptive weight update every 100 iterations
        for operators in [self.destroy_operators, self.repair_operators]:
            for op in operators.values():
                if op['calls'] > 0:
                    op['weight'] = 0.8 * op['weight'] + 0.2 * (op['improvements'] / op['calls'])
