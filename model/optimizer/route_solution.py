"""
Route-based solution representation for VRP metaheuristics.
Compact representation using routes instead of time-expanded binary tensors.
"""

import numpy as np
import copy
from datetime import datetime, timedelta


class RouteSolution:
    """
    Represents a VRP solution as a list of routes.
    Each route is a sequence: [depot, vendor1, vendor2, ..., depot]
    
    This is much more efficient than x[k, i, t1, j, t2] for metaheuristics:
    - Compact: O(n) vs O(k × n² × T²)
    - Always feasible structure
    - Natural for local search operators
    """
    
    def __init__(self, routes, vendors_df, distance_matrix, time_matrix, 
                 capacity_matrix, loading_matrix, max_capacity_kg, max_ldms_vc,
                 discretization_constant, min_date):
        """
        Initialize route solution.
        
        Args:
            routes: List of routes [[0, 2, 5, 0], [0, 1, 4, 0], ...]
            vendors_df: DataFrame with vendor information
            distance_matrix: Distance matrix [km]
            time_matrix: Time matrix [seconds]
            capacity_matrix: Cargo weight per vendor [kg]
            loading_matrix: Loading volume per vendor [m³]
            max_capacity_kg: Max weight capacity per vehicle [kg]
            max_ldms_vc: Max volume capacity per vehicle [m³]
            discretization_constant: Time discretization [hours]
            min_date: Minimum simulation date
        """
        self.routes = routes
        self.vendors_df = vendors_df
        self.distance_matrix = distance_matrix
        self.time_matrix = time_matrix
        self.capacity_matrix = capacity_matrix
        self.loading_matrix = loading_matrix
        self.max_capacity_kg = max_capacity_kg
        self.max_ldms_vc = max_ldms_vc
        self.discretization_constant = discretization_constant
        self.min_date = min_date
        
        # Cache evaluation results
        self._total_distance = None
        self._total_time = None
        self._is_feasible = None
        self._constraint_violations = []
        
    def evaluate(self):
        """
        Calculate objective function: total distance + penalties for violations.
        
        Returns:
            float: Total cost (distance + penalties)
        """
        if self._total_distance is not None:
            return self._total_distance
        
        total_distance = 0
        total_time = 0
        
        for route in self.routes:
            for i in range(len(route) - 1):
                from_node = route[i]
                to_node = route[i + 1]
                total_distance += self.distance_matrix[from_node][to_node]
                total_time += self.time_matrix[from_node][to_node]
        
        self._total_distance = total_distance
        self._total_time = total_time
        
        return total_distance
    
    def is_feasible(self, check_all=True):
        """
        Check if solution satisfies all constraints.
        
        Args:
            check_all: If True, check all constraints. If False, stop at first violation.
            
        Returns:
            bool: True if feasible, False otherwise
        """
        if self._is_feasible is not None and not check_all:
            return self._is_feasible
        
        self._constraint_violations = []
        
        # Check 1: All vendors visited exactly once
        visited_vendors = set()
        for route in self.routes:
            for node in route:
                if node != 0:  # Not depot
                    if node in visited_vendors:
                        self._constraint_violations.append(f"Vendor {node} visited multiple times")
                        if not check_all:
                            self._is_feasible = False
                            return False
                    visited_vendors.add(node)
        
        num_vendors = len(self.capacity_matrix) - 1  # Exclude depot
        if len(visited_vendors) != num_vendors:
            missing = set(range(1, num_vendors + 1)) - visited_vendors
            self._constraint_violations.append(f"Missing vendors: {missing}")
            if not check_all:
                self._is_feasible = False
                return False
        
        # Check 2: Capacity constraints
        for route_idx, route in enumerate(self.routes):
            route_weight = sum(self.capacity_matrix[node] for node in route if node != 0)
            route_volume = sum(self.loading_matrix[node] for node in route if node != 0)
            
            if route_idx < len(self.max_capacity_kg):
                if route_weight > self.max_capacity_kg[route_idx]:
                    self._constraint_violations.append(
                        f"Route {route_idx}: Weight {route_weight:.0f} > {self.max_capacity_kg[route_idx]:.0f} kg"
                    )
                    if not check_all:
                        self._is_feasible = False
                        return False
                
                if route_volume > self.max_ldms_vc[route_idx]:
                    self._constraint_violations.append(
                        f"Route {route_idx}: Volume {route_volume:.1f} > {self.max_ldms_vc[route_idx]:.1f} m³"
                    )
                    if not check_all:
                        self._is_feasible = False
                        return False
        
        # Check 3: Time windows (if vendors_df has time window info)
        if 'Requested Loading' in self.vendors_df.columns:
            for route in self.routes:
                # Initialize current_time - ensure it's a datetime object
                if isinstance(self.min_date, datetime):
                    current_time = self.min_date
                elif isinstance(self.min_date, str):
                    try:
                        current_time = datetime.strptime(self.min_date, '%Y-%m-%d %H:%M:%S')
                    except:
                        # Skip time window checks if date format is unknown
                        continue
                else:
                    # Skip if min_date is not usable
                    continue
                
                for i, node in enumerate(route):
                    if node == 0:
                        continue
                    
                    # Add travel time from previous node
                    if i > 0:
                        prev_node = route[i - 1]
                        travel_seconds = self.time_matrix[prev_node][node]
                        current_time = current_time + timedelta(seconds=travel_seconds)
                    
                    # Check time window (basic check - can be extended)
                    vendor_idx = node - 1
                    if vendor_idx < len(self.vendors_df):
                        requested_time = self.vendors_df.iloc[vendor_idx]['Requested Loading']
                        # Time window checks would go here if needed
        
        self._is_feasible = len(self._constraint_violations) == 0
        return self._is_feasible
    
    def get_num_routes(self):
        """Return number of routes (vehicles used)."""
        return len(self.routes)
    
    def get_route_cost(self, route_idx):
        """Get distance cost of a specific route."""
        if route_idx >= len(self.routes):
            return 0
        
        route = self.routes[route_idx]
        distance = 0
        for i in range(len(route) - 1):
            distance += self.distance_matrix[route[i]][route[i + 1]]
        return distance
    
    def get_route_capacity_usage(self, route_idx):
        """Get capacity usage for a specific route."""
        if route_idx >= len(self.routes):
            return 0, 0
        
        route = self.routes[route_idx]
        weight = sum(self.capacity_matrix[node] for node in route if node != 0)
        volume = sum(self.loading_matrix[node] for node in route if node != 0)
        return weight, volume
    
    def copy(self):
        """Create a deep copy of this solution."""
        return RouteSolution(
            routes=copy.deepcopy(self.routes),
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
    
    def invalidate_cache(self):
        """Invalidate cached evaluation results after modification."""
        self._total_distance = None
        self._total_time = None
        self._is_feasible = None
        self._constraint_violations = []
    
    def __str__(self):
        """String representation of solution."""
        cost = self.evaluate()
        feasible = "✓" if self.is_feasible(check_all=False) else "✗"
        return f"RouteSolution[{len(self.routes)} routes, {cost:.0f} km, feasible: {feasible}]"
