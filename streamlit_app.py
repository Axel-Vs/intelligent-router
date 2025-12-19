import streamlit as st
import pandas as pd
import os
import sys
import json
from datetime import datetime
import tempfile
import numpy as np
import requests

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from model.graph_creator.graph_creator import Graph
from model.optimizer.delivery_model import DeliveryOptimizer
from model.optimizer.route_edit import insert_stop_best_position, remove_stop
from model.utils.pre_processing import *
from model.utils.project_utils import *
from model.utils.run_storage import list_runs, load_run, save_run, generate_run_id

st.set_page_config(
    page_title="🚚 Parcel Delivery Optimizer",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(120deg, #2196F3, #4CAF50);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(120deg, #2196F3, #4CAF50);
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
    }
    .stButton>button:hover {
        background: linear-gradient(120deg, #1976D2, #388E3C);
        transform: scale(1.02);
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🚚 Parcel Delivery Optimizer</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Optimize delivery routes with AI-powered algorithms</p>', unsafe_allow_html=True)

# Sidebar Configuration
st.sidebar.title("⚙️ Configuration")

# Data Source
data_source = st.sidebar.radio(
    "📂 Data Source",
    ["Use Example Dataset", "Upload Custom CSV"],
    help="Choose to use the example Amazon dataset or upload your own"
)

if data_source == "Upload Custom CSV":
    uploaded_file = st.sidebar.file_uploader(
        "Upload Vendor CSV",
        type=['csv'],
        help="CSV should include vendor coordinates, cargo details, and delivery dates"
    )
else:
    uploaded_file = None

# Solver Selection
st.sidebar.markdown("---")
st.sidebar.subheader("🧮 Solver Settings")

solver_type = st.sidebar.selectbox(
    "Optimization Solver",
    ["Auto (Recommended)", "ALNS Metaheuristic", "CBC MIP Exact"],
    help="Auto selects the best solver based on problem size"
)

use_metaheuristic = solver_type in ["Auto (Recommended)", "ALNS Metaheuristic"]

# Advanced Parameters
with st.sidebar.expander("🔧 Advanced Parameters"):
    if use_metaheuristic or solver_type == "Auto (Recommended)":
        st.markdown("**ALNS Parameters**")
        alns_iterations = st.number_input("Iterations", 500, 5000, 2000, 100)
        alns_temp = st.number_input("Temperature", 500, 3000, 1500, 100)
        alns_cooling = st.slider("Cooling Rate", 0.95, 0.999, 0.997, 0.001)
    
    if solver_type in ["CBC MIP Exact", "Auto (Recommended)"]:
        st.markdown("**MIP Parameters**")
        solver_time_limit = st.number_input("Time Limit (seconds)", 60, 3600, 900, 60)
        mip_gap = st.slider("MIP Gap Tolerance", 0.01, 0.5, 0.1, 0.01)

# Date Range
st.sidebar.markdown("---")
st.sidebar.subheader("📅 Simulation Period")

use_custom_dates = st.sidebar.checkbox("Custom Date Range", value=False)

if use_custom_dates:
    start_date = st.sidebar.date_input("Start Date", datetime(2023, 8, 15))
    end_date = st.sidebar.date_input("End Date", datetime(2023, 10, 11))
    period = [start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")]
else:
    period = None  # Will use dataset date range

# Main Content
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    optimize_button = st.button("🚀 Optimize Routes", use_container_width=True)

if optimize_button or st.session_state.get('optimization_complete', False):
    with st.spinner("🔄 Loading data and preparing optimization..."):
        try:
            # Load data
            if uploaded_file is not None:
                vendors_df = pd.read_csv(uploaded_file)
                st.success(f"✅ Loaded {len(vendors_df)} vendors from uploaded file")
            else:
                data_path = 'data/amazon_test_dataset.csv'
                vendors_df = pd.read_csv(data_path)
                st.success(f"✅ Loaded {len(vendors_df)} vendors from example dataset")
            
            # Set period if not custom
            if period is None:
                period = get_dates_period(vendors_df, 'Requested Delivery')
            
            st.info(f"📅 Optimization period: {period[0]} to {period[1]}")
            
            # Show vendor summary
            with st.expander("📊 View Vendor Data", expanded=False):
                st.dataframe(vendors_df.head(10), use_container_width=True)
                st.caption(f"Showing first 10 of {len(vendors_df)} vendors")
            
        except Exception as e:
            st.error(f"❌ Error loading data: {str(e)}")
            st.stop()
    
    # Optimization
    if len(vendors_df) >= 1:
        with st.spinner(f"🧮 Optimizing routes for {len(vendors_df)} vendors..."):
            try:
                progress_bar = st.progress(0, text="Initializing optimization...")
                
                # Extract vendor information
                progress_bar.progress(10, text="Extracting vendor information...")
                vendors_coordinates, depot_coordinates, vendors_demand_pickup, vendors_dates, vendors_info = extract_info(vendors_df)
                
                # Create network
                progress_bar.progress(30, text="Creating transportation network...")
                net = Graph()
                net.load_parameters('model/config/network_params.txt')
                net.create_network(depot_coordinates, vendors_coordinates, vendors_demand_pickup, vendors_dates, vendors_info)
                
                # Determine solver
                num_vendors = len(vendors_df)
                auto_use_metaheuristic = (solver_type == "Auto (Recommended)" and num_vendors >= 20)
                final_use_metaheuristic = use_metaheuristic or auto_use_metaheuristic
                
                if final_use_metaheuristic:
                    progress_bar.progress(40, text=f"Using ALNS metaheuristic for {num_vendors} vendors...")
                    st.info("🎯 Using ALNS Metaheuristic solver (optimized for large problems)")
                else:
                    progress_bar.progress(40, text="Creating time-expanded network...")
                    time_expanded_network, complete_time_index, time_expanded_network_index = net.create_time_network(period, net.Tau_hours, 0, 0)
                    st.info("🎯 Using CBC MIP solver (exact solution)")
                
                # Initialize optimizer
                progress_bar.progress(60, text="Initializing optimizer...")
                model = DeliveryOptimizer()
                model.load_parameters('model/config/model_params.txt')
                model.generate_model(net, time_expanded_network_index if not final_use_metaheuristic else [])
                
                # Solve
                progress_bar.progress(70, text="Solving optimization problem...")
                start_time = datetime.now()
                
                if final_use_metaheuristic:
                    solution, solving_time = model.solve_with_metaheuristic(
                        iterations=alns_iterations if 'alns_iterations' in locals() else 2000,
                        start_temperature=alns_temp if 'alns_temp' in locals() else 1500,
                        cooling_rate=alns_cooling if 'alns_cooling' in locals() else 0.997
                    )
                else:
                    solution = model.solve_MIP()
                    solving_time = (datetime.now() - start_time).total_seconds()
                
                progress_bar.progress(90, text="Generating visualization...")
                
                # Create map
                if final_use_metaheuristic:
                    solver_name = "ALNS Metaheuristic"
                    map_filename = f"routes_{period[0]}_metaheuristic.html"
                else:
                    solver_name = "CBC MIP"
                    map_filename = f"routes_{period[0]}.html"
                
                save_path = os.path.join('results', 'optimization', map_filename)
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                
                model.visualize_routes(
                    solution,
                    net.depot_coordinates,
                    net.vendors_coordinates,
                    net.vendors_demand_pickup,
                    net.vendors_info,
                    solving_time,
                    solver_name,
                    save_path=save_path,
                    show_plot=False
                )
                
                progress_bar.progress(100, text="✅ Optimization complete!")
                
                # Store results in session state
                st.session_state['optimization_complete'] = True
                st.session_state['map_path'] = save_path
                st.session_state['solution'] = solution
                st.session_state['solving_time'] = solving_time
                st.session_state['num_vendors'] = num_vendors
                st.session_state['solver_name'] = solver_name

                # Persist matrices/routes for local edits
                try:
                    # Routes: convert dict -> list for convenience
                    if isinstance(solution, dict):
                        st.session_state['last_routes'] = list(solution.values())
                    else:
                        st.session_state['last_routes'] = solution

                    # Distance matrix if available from network
                    if hasattr(net, 'distance_matrix') and net.distance_matrix is not None:
                        st.session_state['last_distance_matrix'] = net.distance_matrix

                    # Capacity/volume from dataset (best-effort; depot=0)
                    if 'Total Gross Weight' in vendors_df.columns:
                        st.session_state['last_capacity_matrix'] = [0] + vendors_df['Total Gross Weight'].fillna(0).tolist()
                    elif 'Vendor Gross Weight' in vendors_df.columns:
                        st.session_state['last_capacity_matrix'] = [0] + vendors_df['Vendor Gross Weight'].fillna(0).tolist()

                    vol_col = None
                    for candidate in ['Calculated Loading Meters', 'Vendor Loading Meters', 'Vendor Dimensions in m3']:
                        if candidate in vendors_df.columns:
                            vol_col = candidate
                            break
                    if vol_col:
                        st.session_state['last_loading_matrix'] = [0] + vendors_df[vol_col].fillna(0).tolist()
                except Exception:
                    pass
                
            except Exception as e:
                st.error(f"❌ Optimization failed: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
                st.stop()
    else:
        st.warning("⚠️ No vendors to optimize. Please check your data.")
        st.stop()

# Display Results
if st.session_state.get('optimization_complete', False):
    st.success("🎉 Optimization Complete!")
    
    # Metrics
    st.markdown("### 📊 Solution Summary")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🚚 Vehicles Used", len(st.session_state['solution']))
    
    with col2:
        st.metric("📦 Vendors Served", st.session_state['num_vendors'])
    
    with col3:
        st.metric("⏱️ Solving Time", f"{st.session_state['solving_time']:.2f}s")
    
    with col4:
        st.metric("🧮 Solver", st.session_state['solver_name'])
    
    # Display Map
    st.markdown("---")
    st.markdown("### 🗺️ Interactive Route Map")
    
    # Read and display HTML map
    with open(st.session_state['map_path'], 'r', encoding='utf-8') as f:
        map_html = f.read()
    
    st.components.v1.html(map_html, height=800, scrolling=True)
    
    # Download Options
    st.markdown("---")
    st.markdown("### 📥 Download Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        with open(st.session_state['map_path'], 'r', encoding='utf-8') as f:
            st.download_button(
                label="📄 Download HTML Map",
                data=f.read(),
                file_name=os.path.basename(st.session_state['map_path']),
                mime="text/html"
            )
    
    with col2:
        # Create solution summary as CSV
        solution_data = []
        for vehicle_id, route in st.session_state['solution'].items():
            solution_data.append({
                'Vehicle': vehicle_id,
                'Route': ' → '.join([str(v) for v in route])
            })
        solution_df = pd.DataFrame(solution_data)
        
        st.download_button(
            label="📊 Download Solution CSV",
            data=solution_df.to_csv(index=False),
            file_name=f"solution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

    st.markdown("---")
    st.markdown("### ✏️ Adjust Routes Locally (No Full Re-run)")
    st.caption("Use this to add/remove stops without re-optimizing everything. Provide current routes and matrices; only the affected route changes.")

    default_routes = (
        st.session_state.get('last_routes')
        or (list(st.session_state['solution'].values()) if st.session_state.get('solution') else [[0, 1, 0]])
    )

    default_distance = st.session_state.get('last_distance_matrix')
    default_capacity = st.session_state.get('last_capacity_matrix')
    default_loading = st.session_state.get('last_loading_matrix')
    default_frozen = None
    routes_text = st.text_area(
        "Current routes (JSON array of routes)",
        value=json.dumps(default_routes, indent=2),
        height=150,
    )

    colA, colB = st.columns(2)
    with colA:
        distance_text = st.text_area(
            "Distance matrix (2D JSON)",
            height=120,
            value=json.dumps(default_distance, indent=2) if default_distance is not None else "",
            placeholder="[[0, 5, ...], [5, 0, ...], ...]",
        )
        capacity_text = st.text_area(
            "Capacity (weight) per node (JSON array)",
            height=80,
            value=json.dumps(default_capacity, indent=2) if default_capacity is not None else "",
            placeholder="[0, 1200, 800, ...]",
        )
    with colB:
        loading_text = st.text_area(
            "Volume per node (JSON array)",
            height=80,
            value=json.dumps(default_loading, indent=2) if default_loading is not None else "",
            placeholder="[0, 3.4, 2.1, ...]",
        )
        frozen_text = st.text_area(
            "Frozen prefix per route (JSON array, optional)",
            height=80,
            value=json.dumps(default_frozen, indent=2) if default_frozen is not None else "",
            placeholder="[0, 0, 0]",
        )

    add_col, rem_col = st.columns(2)
    with add_col:
        new_stop = st.number_input("Stop to add (node index)", min_value=1, step=1, value=1)
        allow_new_route = st.checkbox("Allow opening a new route", value=True)
        add_btn = st.button("➕ Add Stop")
    with rem_col:
        remove_stop_id = st.number_input("Stop to remove (node index)", min_value=1, step=1, value=1)
        rem_btn = st.button("➖ Remove Stop")

    def _parse_json(label, text, required=True):
        if not text.strip():
            if required:
                st.error(f"{label} is required")
            return None
        try:
            return json.loads(text)
        except Exception as e:
            st.error(f"{label} parse error: {e}")
            return None

    if add_btn or rem_btn:
        routes = _parse_json("Routes", routes_text)
        distance_matrix = _parse_json("Distance matrix", distance_text)
        capacity_matrix = _parse_json("Capacity (weight)", capacity_text)
        loading_matrix = _parse_json("Volume", loading_text)
        frozen_prefix = _parse_json("Frozen prefix", frozen_text, required=False)

        if routes is None or distance_matrix is None or capacity_matrix is None or loading_matrix is None:
            st.stop()

        try:
            max_capacity_kg = max(capacity_matrix) if len(capacity_matrix) > 0 else 0
            max_ldms_vc = max(loading_matrix) if len(loading_matrix) > 0 else 0
        except Exception:
            st.error("Invalid capacity or volume arrays")
            st.stop()

        if add_btn:
            result = insert_stop_best_position(
                routes=routes,
                new_stop=int(new_stop),
                distance_matrix=distance_matrix,
                capacity_matrix=capacity_matrix,
                loading_matrix=loading_matrix,
                max_capacity_kg=float(max_capacity_kg),
                max_ldms_vc=float(max_ldms_vc),
                frozen_prefix=frozen_prefix,
                allow_new_route=allow_new_route,
            )
            if result.get("success"):
                st.success(f"Stop inserted in route {result.get('route_index')} at position {result.get('position')}. Δdistance={result.get('delta_distance')}")
                st.json(result.get("routes"))
                # Update session state
                st.session_state['last_routes'] = result.get("routes")
            else:
                st.error(result.get("message", "Insertion failed"))

        if rem_btn:
            result = remove_stop(routes=routes, stop=int(remove_stop_id))
            if result.get("success"):
                st.success(f"Stop removed from route {result.get('route_index')}")
                st.json(result.get("routes"))
                st.session_state['last_routes'] = result.get("routes")
            else:
                st.error(result.get("message", "Stop not found"))

    st.markdown("---")
    st.markdown("### 🗂️ Runs: Load and Save")
    try:
        runs = list_runs()
        run_options = {f"{r.get('name','(unnamed)')} ({r.get('run_id')})": r.get('run_id') for r in runs}
        selected_label = st.selectbox("Select a saved run", options=list(run_options.keys())) if run_options else None
        if selected_label:
            if st.button("📥 Load Selected Run"):
                data = load_run(run_options[selected_label])
                if data.get('success'):
                    state = data['state']
                    st.session_state['last_routes'] = state.get('routes')
                    st.session_state['last_distance_matrix'] = state.get('distance_matrix')
                    st.session_state['last_capacity_matrix'] = state.get('capacity_matrix')
                    st.session_state['last_loading_matrix'] = state.get('loading_matrix')
                    st.success(f"Loaded run {data['metadata'].get('name')} ({data['run_id']})")
                else:
                    st.error(data.get('error','Failed to load run'))

            # Re-run from dataset using Flask endpoint
            selected_run_id = run_options[selected_label]
            colR1, colR2 = st.columns(2)
            with colR1:
                if st.button("🔁 Re-run Selected Run (Flask)"):
                    try:
                        params = {}
                        api_url = os.environ.get('PARCEL_API_URL', 'http://localhost:8080')
                        resp = requests.post(f"{api_url}/api/runs/rerun", json={
                            'run_id': selected_run_id,
                            'parameters': params,
                        }, timeout=60)
                        if resp.status_code == 200:
                            payload = resp.json()
                            new_run_id = payload.get('new_run_id') or (payload.get('result') or {}).get('run_id')
                            if new_run_id:
                                st.success(f"Re-run completed. New run id: {new_run_id}")
                            else:
                                st.success("Re-run completed.")
                        else:
                            st.error(f"Re-run failed: {resp.text}")
                    except Exception as e:
                        st.error(f"Re-run error: {e}")
            with colR2:
                if st.button("📄 Download Input with Status (CSV)"):
                    try:
                        api_url = os.environ.get('PARCEL_API_URL', 'http://localhost:8080')
                        resp = requests.get(f"{api_url}/api/runs/download-input/{selected_run_id}", timeout=60)
                        if resp.status_code == 200:
                            st.download_button(
                                label="⬇️ Save CSV",
                                data=resp.content,
                                file_name=f"input_with_status_{selected_run_id}.csv",
                                mime="text/csv",
                            )
                        else:
                            st.error(f"Download failed: {resp.text}")
                    except Exception as e:
                        st.error(f"Download error: {e}")

        new_run_name = st.text_input("Save current state as new run (name)", value=f"Run {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        if st.button("💾 Save Run"):
            current_state = {
                'routes': st.session_state.get('last_routes'),
                'distance_matrix': st.session_state.get('last_distance_matrix'),
                'capacity_matrix': st.session_state.get('last_capacity_matrix'),
                'loading_matrix': st.session_state.get('last_loading_matrix'),
                'frozen_prefix': None,
            }
            run_id = generate_run_id('run')
            res = save_run(run_id, new_run_name, current_state, {'created_at': datetime.now().isoformat()})
            if res.get('success'):
                st.success(f"Saved new run: {new_run_name} ({run_id})")
            else:
                st.error("Failed to save run")
    except Exception as e:
        st.error(f"Run management error: {e}")
    
    # Reset button
    if st.button("🔄 Start New Optimization"):
        st.session_state['optimization_complete'] = False
        st.rerun()

else:
    # Instructions when no optimization has been run
    st.markdown("---")
    st.markdown("""
    ### 🚀 Getting Started
    
    1. **Choose your data source** in the sidebar (example dataset or upload custom CSV)
    2. **Select optimization solver** (Auto, ALNS, or CBC MIP)
    3. **Configure parameters** (optional, expand Advanced Parameters)
    4. **Set date range** (optional, use Custom Date Range)
    5. **Click "Optimize Routes"** to start optimization
    
    ### 📋 CSV Format Required
    
    Your CSV should include these columns:
    - `vendor_latitude`, `vendor_longitude` - Vendor location
    - `recipient_latitude`, `recipient_longitude` - Delivery destination  
    - `vendor Name`, `Vendor City`, `Vendor Postcode` - Vendor info
    - `Total Shipment Weight (kg)` - Cargo weight
    - `Total Volume (cbm)` - Cargo volume
    - `Requested Delivery` - Delivery date/time
    
    ### 🎯 Solver Selection Guide
    
    - **Auto (Recommended)**: Smart selection based on problem size
    - **ALNS Metaheuristic**: Fast, handles 50+ vendors efficiently
    - **CBC MIP Exact**: Optimal solutions for smaller problems (<20 vendors)
    """)
    
    # Example data preview
    st.markdown("---")
    st.markdown("### 📊 Example Dataset Preview")
    try:
        example_df = pd.read_csv('data/amazon_test_dataset.csv')
        st.dataframe(example_df.head(5), use_container_width=True)
        st.caption(f"Example dataset contains {len(example_df)} vendors")
    except:
        pass

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p>🚚 Parcel Delivery Optimizer | Built with Streamlit</p>
    <p>Powered by OR-Tools CBC & ALNS Metaheuristic</p>
</div>
""", unsafe_allow_html=True)
