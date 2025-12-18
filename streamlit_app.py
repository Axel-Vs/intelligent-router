import streamlit as st
import pandas as pd
import os
import sys
from datetime import datetime
import tempfile
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from model.graph_creator.graph_creator import Graph
from model.optimizer.delivery_model import DeliveryOptimizer
from model.utils.pre_processing import *
from model.utils.project_utils import *

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
