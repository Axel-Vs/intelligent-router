"""
Flask API Backend for Parcel Delivery Optimizer
Provides REST endpoints for route optimization
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.graph_creator.graph_creator import Graph
from model.optimizer.delivery_model import DeliveryOptimizer
from model.optimizer.alns_solver import ALNSSolver
import json

app = Flask(__name__, static_folder='web', static_url_path='')
CORS(app)

# Create necessary directories
os.makedirs('uploads', exist_ok=True)
os.makedirs('results/optimization', exist_ok=True)


@app.route('/')
def index():
    """Serve the main HTML interface"""
    return send_from_directory('web', 'index.html')


@app.route('/api/upload-csv', methods=['POST'])
def upload_csv():
    """Handle CSV file upload and return vendor data"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Save uploaded file
        filepath = os.path.join('uploads', f'vendors_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
        file.save(filepath)
        
        # Read and validate CSV
        df = pd.read_csv(filepath)
        
        # Return vendor data as JSON
        # The CSV will be processed and geocoded during the optimization phase
        vendors = []
        for idx, row in df.iterrows():
            vendors.append({
                'id': idx + 1,
                'name': str(row.get('vendor Name', row.get('Vendor Name', f'Vendor {idx + 1}'))),
                'city': str(row.get('Vendor City', 'N/A')),
                'latitude': float(row.get('vendor_latitude', 0)),
                'longitude': float(row.get('vendor_longitude', 0)),
                'recipient_latitude': float(row.get('recipient_latitude', 0)),
                'recipient_longitude': float(row.get('recipient_longitude', 0)),
                'weight': float(row.get('Total Gross Weight', row.get('Vendor Gross Weight', 0))),
                'volume': float(row.get('Calculated Loading Meters', row.get('Vendor Loading Meters', row.get('Vendor Dimensions in m3', 0)))),
                'delivery_date': str(row.get('Requested Delivery', row.get('Requested Delivery Date', '')))
            })
        
        return jsonify({
            'success': True,
            'vendors': vendors,
            'count': len(vendors),
            'filepath': filepath
        })
    
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in upload-csv: {error_details}")
        return jsonify({'error': str(e), 'details': error_details}), 500


@app.route('/api/optimize', methods=['POST'])
def optimize_routes():
    """Run route optimization with provided parameters"""
    try:
        data = request.json
        vendors_data = data.get('vendors', [])
        params = data.get('parameters', {})
        csv_filepath = data.get('csv_filepath', None)
        
        if not vendors_data:
            return jsonify({'error': 'No vendor data provided'}), 400
        
        # Get other parameters
        use_metaheuristic = params.get('use_metaheuristic', True)
        max_vehicles = len(vendors_data)  # Always use all vendors as max vehicles
        
        # Read the original CSV file to preserve all columns for geocoding
        if csv_filepath and os.path.exists(csv_filepath):
            print(f"Using CSV file: {csv_filepath}")
            df_raw = pd.read_csv(csv_filepath)
            
            # Calculate period from RAW dates spanning earliest loading to latest delivery
            period = None
            min_date = None
            max_date = None
            
            # Get earliest loading date
            if 'Requested Loading Date' in df_raw.columns:
                loading_dates = pd.to_datetime(df_raw['Requested Loading Date'], errors='coerce').dropna()
                if len(loading_dates) > 0:
                    min_date = loading_dates.min()
                    max_date = loading_dates.max()
            
            # Get latest delivery date and expand range
            if 'Requested Delivery Date' in df_raw.columns:
                delivery_dates = pd.to_datetime(df_raw['Requested Delivery Date'], errors='coerce').dropna()
                if len(delivery_dates) > 0:
                    if min_date is None:
                        min_date = delivery_dates.min()
                    else:
                        min_date = min(min_date, delivery_dates.min())
                    
                    if max_date is None:
                        max_date = delivery_dates.max()
                    else:
                        max_date = max(max_date, delivery_dates.max())
            
            if min_date is not None and max_date is not None:
                period = [min_date, max_date]
                print(f"📅 Tour period: {min_date} to {max_date}")
            
            # Prepare dataframe similar to simulator.py preprocessing
            df = df_raw.copy()
            
            # Map vendor name
            if 'Vendor Name' in df.columns:
                df['vendor Name'] = df['Vendor Name'].astype(str)
            
            # Map weights and loading
            if 'Vendor Gross Weight' in df.columns:
                df['Total Gross Weight'] = df['Vendor Gross Weight']
            if 'Vendor Loading Meters' in df.columns:
                df['Calculated Loading Meters'] = df['Vendor Loading Meters']
            elif 'Vendor Dimensions in m3' in df.columns:
                df['Calculated Loading Meters'] = df['Vendor Dimensions in m3']
            
            # Map dates
            if 'Requested Loading Date' in df.columns:
                df['Requested Loading'] = pd.to_datetime(df['Requested Loading Date'], errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S')
            else:
                df['Requested Loading'] = ''
            
            if 'Requested Delivery Date' in df.columns:
                df['Requested Delivery'] = pd.to_datetime(df['Requested Delivery Date'], errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S')
            else:
                df['Requested Delivery'] = df['Requested Loading']
            
            # Track failed geocoding attempts (will be returned to frontend)
            failed_geocodes = []
            
            # Geocode addresses if coordinates not present
            if 'vendor_latitude' not in df.columns or 'recipient_latitude' not in df.columns:
                print("📍 Geocoding addresses...")
                
                # City coordinate fallbacks for common cities (used when geocoding fails)
                city_coords = {
                    'San Francisco': (37.7749, -122.4194),
                    'Chicago': (41.8781, -87.6298),
                    'Houston': (29.7604, -95.3698),
                    'Los Angeles': (34.0522, -118.2437),
                    'New York': (40.7128, -74.0060),
                    'Miami': (25.7617, -80.1918),
                    'Mexico City': (19.4326, -99.1332),
                    'Guadalajara': (20.6597, -103.3496),
                    'Monterrey': (25.6866, -100.3161),
                    'Vancouver': (49.2827, -123.1207),
                    'Toronto': (43.6532, -79.3832),
                    'Montreal': (45.5017, -73.5673),
                    'Seattle': (47.6062, -122.3321)
                }
                
                from model.utils.geocoder import Geocoder
                geocode_cache = os.path.join('data', 'geocode_cache.csv')
                g = Geocoder(cache_path=geocode_cache, user_agent='parcel_web_geocoder', min_delay_seconds=1)
                
                # Determine which columns need geocoding
                need_vendor_geocoding = 'vendor_latitude' not in df.columns
                need_recipient_geocoding = 'recipient_latitude' not in df.columns
                
                # Geocode all addresses
                for idx, row in df.iterrows():
                    if need_vendor_geocoding:
                        # Try city-based fallback first for speed
                        vendor_city = str(row.get('Vendor City', '')).strip()
                        if vendor_city in city_coords:
                            df.at[idx, 'vendor_latitude'] = city_coords[vendor_city][0]
                            df.at[idx, 'vendor_longitude'] = city_coords[vendor_city][1]
                        else:
                            vendor_addr = f"{row.get('Vendor Street', '')}, {row.get('Vendor City', '')}, {row.get('Vendor Postcode', '')}, {row.get('Vendor Country Name', '')}"
                            vendor_coords = g.geocode_address(vendor_addr.strip())
                            if vendor_coords and vendor_coords[0] is not None and vendor_coords[1] is not None:
                                df.at[idx, 'vendor_latitude'] = float(vendor_coords[0])
                                df.at[idx, 'vendor_longitude'] = float(vendor_coords[1])
                            else:
                                df.at[idx, 'vendor_latitude'] = 0.0
                                df.at[idx, 'vendor_longitude'] = 0.0
                                failed_geocodes.append({
                                    'type': 'vendor',
                                    'address': vendor_addr.strip(),
                                    'row': int(idx),
                                    'vendor_name': str(row.get('Vendor Name', 'Unknown'))
                                })
                    
                    if need_recipient_geocoding:
                        # Try city-based fallback first
                        recipient_city = str(row.get('Recipient City', '')).strip()
                        if recipient_city in city_coords:
                            df.at[idx, 'recipient_latitude'] = city_coords[recipient_city][0]
                            df.at[idx, 'recipient_longitude'] = city_coords[recipient_city][1]
                        else:
                            # Geocode recipient address
                            recipient_addr = f"{row.get('Recipient Street', '')}, {row.get('Recipient City', '')}, {row.get('Recipient Postcode', '')}, {row.get('Recipient Country Name', '')}"
                            recipient_coords = g.geocode_address(recipient_addr.strip())
                            if recipient_coords and recipient_coords[0] is not None and recipient_coords[1] is not None:
                                df.at[idx, 'recipient_latitude'] = float(recipient_coords[0])
                                df.at[idx, 'recipient_longitude'] = float(recipient_coords[1])
                            else:
                                # Default to Seattle if geocoding fails
                                df.at[idx, 'recipient_latitude'] = 47.6062
                                df.at[idx, 'recipient_longitude'] = -122.3321
                                failed_geocodes.append({
                                    'type': 'recipient',
                                    'address': recipient_addr.strip(),
                                    'row': int(idx),
                                    'vendor_name': str(row.get('Vendor Name', 'Unknown'))
                                })
                
                print(f"✅ Geocoded {len(df)} addresses")
                
                # Log failed geocoding attempts
                if failed_geocodes:
                    print(f"\n⚠️  Failed to geocode {len(failed_geocodes)} address(es):")
                    for fail in failed_geocodes:
                        print(f"   • {fail['type'].capitalize()}: {fail['address']}")
                        print(f"     Vendor: {fail['vendor_name']} (Row {fail['row']})")
                    
                    # Save to log file
                    failed_log_path = os.path.join('data', 'failed_geocodes.csv')
                    pd.DataFrame(failed_geocodes).to_csv(failed_log_path, index=False)
                    print(f"   📄 Details saved to: {failed_log_path}")
                else:
                    print("✅ All addresses geocoded successfully!")
            
            # Save processed CSV for read_data
            temp_csv = os.path.join('uploads', f'processed_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
            df.to_csv(temp_csv, index=False)
        else:
            # Fallback: create DataFrame from JSON vendors_data (no geocoding needed)
            failed_geocodes = []
            df = pd.DataFrame(vendors_data)
            df['vendor Name'] = df['name']
            df['Vendor City'] = df['city']
            df['vendor_latitude'] = df['latitude']
            df['vendor_longitude'] = df['longitude']
            df['recipient_latitude'] = df['recipient_latitude']
            df['recipient_longitude'] = df['recipient_longitude']
            df['Total Gross Weight'] = df['weight']
            df['Calculated Loading Meters'] = df['volume']
            df['Requested Delivery'] = df['delivery_date']
            df['Requested Loading'] = df['delivery_date']
            
            temp_csv = os.path.join('uploads', f'temp_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
            df.to_csv(temp_csv, index=False)
        
        # Get depot coordinates from the first recipient row (they should all be the same depot)
        depot_lat = df['recipient_latitude'].iloc[0] if 'recipient_latitude' in df.columns else 47.6062
        depot_lon = df['recipient_longitude'].iloc[0] if 'recipient_longitude' in df.columns else -122.3321
        
        # Create network parameters
        network_params = {
            'discretization_constant': 4,
            'starting_depot': params.get('starting_depot', 8),
            'closing_depot': params.get('closing_depot', 18),
            'vendor_start_hr': params.get('vendor_start_hr', 6),
            'pickup_end_hr': params.get('pickup_end_hr', 14),
            'loading': params.get('loading', 2),
            'earl_arv': params.get('earl_arv', 24),
            'late_arv': params.get('late_arv', 24),
            'max_driving': params.get('max_driving', 15),
            'driving_starts': params.get('driving_starts', 6),
            'driving_stop': params.get('driving_stop', 21),
            'max_weight': params.get('max_weight', 30),
            'max_ldms': params.get('max_ldms', 70),
            'plot_centered_coordinates': [depot_lat, depot_lon],
            'max_feasible_distance': 3000,
            'time_window_sampling_threshold': 20,
            'time_window_sample_size': 20
        }
        
        # Create graph
        print(f"Creating graph with {len(vendors_data)} vendors...")
        net = Graph(network_params)
        
        # Use period calculated earlier from raw CSV, or fallback to current time
        if period is None:
            print("⚠️ No period calculated from CSV, using current time as fallback")
            period = [pd.Timestamp.now(), pd.Timestamp.now()]
        
        # Convert period to strings for processing
        period_str = [period[0].strftime('%Y-%m-%d %H:%M:%S'), period[1].strftime('%Y-%m-%d %H:%M:%S')]
        
        # Force all date columns to be strings before passing to read_data
        for col in ['Requested Delivery', 'Requested Loading']:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: 
                    x.strftime('%Y-%m-%d %H:%M:%S') if hasattr(x, 'strftime') 
                    else str(x) if x and not pd.isna(x) 
                    else datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                )
        
        # Read data into graph
        print(f"Reading vendor data for period {period_str[0]} to {period_str[1]}...")
        try:
            complete_coordinates, vendors_df = net.read_data(period_str, df)
        except Exception as e:
            return jsonify({'error': f'Failed to read vendor data: {str(e)}'}), 500
        
        print(f"Successfully loaded {len(vendors_df)} vendors")
        
        # Create network
        net.create_network(complete_coordinates, vendors_df)
        
        # Discretize the network (creates disc_time_distance_matrix)
        net.discretize()
        
        # Extract matrices and parameters
        capacity_matrix = vendors_df['Total Gross Weight'].to_numpy()
        loading_matrix = vendors_df['Calculated Loading Meters'].to_numpy()
        
        # Prepare network data based on solver type (following simulator.py pattern)
        if use_metaheuristic:
            # Metaheuristic doesn't need time-expanded network
            print("🚀 Using ALNS metaheuristic solver (fast mode)")
            time_expanded_network = []
            time_expanded_network_index = []
            net.Tau_hours = [0]
            net.min_date = period[0]
        else:
            # MIP solver requires time-expanded network
            print("🎯 Using exact MIP solver")
            time_expanded_network, complete_time_index, time_expanded_network_index = net.create_time_network(
                vendors_df, period_str[0], period_str[1]
            )
        
        # Create optimizer (same for both solver types)
        optimizer = DeliveryOptimizer(
            evaluation_period=period,
            discretization_constant=net.discretization_constant,
            time_expanded_network=time_expanded_network,
            time_expanded_network_index=time_expanded_network_index,
            Tau_hours=net.Tau_hours,
            distance_matrix=net.distance_matrix,
            time_distance_matrix=net.time_distance_matrix,
            disc_time_distance_matrix=net.disc_time_distance_matrix,
            capacity_matrix=capacity_matrix,
            loading_matrix=loading_matrix,
            max_capacity=network_params['max_weight'],
            max_ldms=network_params['max_ldms'],
            max_driving=network_params['max_driving'],
            is_gap=not use_metaheuristic,
            mip_gap=0.05,
            maximum_minutes=360 if not use_metaheuristic else 60,
            vendors_df=vendors_df
        )
        optimizer.min_date = net.min_date
        
        # Run optimization with selected solver
        print(f"Starting optimization...")
        start_time = datetime.now()
        
        if use_metaheuristic:
            status, x, y = optimizer.solve_with_metaheuristic(
                w=0.5,
                max_iterations=params.get('iterations', 2000),
                verbose=True
            )
            solver_type = "ALNS Metaheuristic"
        else:
            optimizer.create_model(w=0.5)
            status, x, y = optimizer.solve_model()
            solver_type = "CBC MIP"
        
        solving_time = (datetime.now() - start_time).total_seconds()
        
        # Check if solution was found
        if status != 0:
            return jsonify({'error': f'Optimization failed with status {status} (0=optimal, 1=feasible, 2=infeasible)'}), 500
        
        # Generate map
        map_filename = f'routes_{datetime.now().strftime("%Y%m%d_%H%M%S")}.html'
        map_path = os.path.join('results/optimization', map_filename)
        os.makedirs('results/optimization', exist_ok=True)
        
        # Plot routes using the optimizer's built-in method
        _, route_stats = optimizer.plot_routes(x, y, show_plot=False, save_path=map_path)
        
        # Extract statistics from route_stats returned by plot_routes
        num_vehicles_used = len(route_stats)
        total_distance = sum(stats['total_distance'] for stats in route_stats.values())
        total_cargo = sum(stats['total_cargo'] for stats in route_stats.values())
        total_loading = sum(stats['total_loading'] for stats in route_stats.values())
        
        print(f"Solution found: {num_vehicles_used} vehicles used")
        print(f"Total distance: {total_distance:.0f} km")
        print(f"Total cargo: {total_cargo:.0f} kg")
        print(f"Total loading: {total_loading:.1f} m³")
        
        # Create detailed route summaries from statistics
        route_summaries = []
        for vehicle_id in sorted(route_stats.keys()):
            stats = route_stats[vehicle_id]
            route_summaries.append({
                'route_id': int(vehicle_id + 1),
                'num_vendors': int(stats['num_vendors']),
                'distance': float(round(stats['total_distance'], 1)),
                'cargo': float(round(stats['total_cargo'], 0)),
                'loading': float(round(stats['total_loading'], 2))
            })
        
        return jsonify({
            'success': True,
            'map_url': f'/results/optimization/{map_filename}',
            'statistics': {
                'total_distance': float(round(total_distance, 1)),
                'total_cargo': float(round(total_cargo, 0)),
                'total_loading': float(round(total_loading, 2)),
                'num_routes': int(num_vehicles_used),
                'num_vendors': int(len(vendors_data)),
                'solving_time': float(round(solving_time, 2)),
                'solver_type': str(solver_type)
            },
            'routes': route_summaries,
            'failed_geocodes': failed_geocodes
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/results/optimization/<path:filename>')
def serve_map(filename):
    """Serve generated map files"""
    return send_from_directory('results/optimization', filename)


@app.route('/api/health', methods=['GET'])
def health_check():
    """API health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat()
    })


if __name__ == '__main__':
    print("🚀 Starting Parcel Delivery Optimizer Server...")
    print("📍 Open http://localhost:8080 in your browser")
    app.run(debug=True, host='0.0.0.0', port=8080)
