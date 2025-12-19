# Parcel Delivery Solver - AI Agent Instructions

## Architecture Overview

This is a **vehicle routing problem (VRP) optimization platform** with three entry points:
- `streamlit_app.py` - Streamlit web UI for interactive optimization
- `app.py` - Flask REST API backend serving `web/` frontend
- `example/simulator.py` - CLI script for batch optimization

### Core Components

**Time-Expanded Network Model:**
- `model/graph_creator/graph_creator.py` creates time-discretized graphs from vendor data
- Time is discretized into periods (default 4-hour blocks via `discretization_constant`)
- Nodes represent (location, time) tuples; arcs connect feasible transitions
- Arc pruning parameters: `max_feasible_distance`, `time_window_sampling_threshold`

**Dual Solver Architecture:**
- `model/optimizer/delivery_model.py` - CBC MIP exact solver (OR-Tools)
  - Creates binary decision variables for time-expanded network
  - Handles <20 vendors efficiently, uses `mip_gap` tolerance (default 0.1)
- `model/optimizer/alns_solver.py` - ALNS metaheuristic (fast for 20+ vendors)
  - Route-based representation (not binary tensors)
  - Destroy/repair operators: random, worst_cost, shaw, greedy, regret2
  - Adaptive operator weights updated during search
- **Auto-switching:** System automatically selects ALNS for ≥20 vendors

**Routing Providers (Priority Order):**
1. **OSRM** (default, free, unlimited) - `http://router.project-osrm.org`
2. Google Maps (if `google_maps_api_key` in `network_params.txt`)
3. OpenRouteService (ORS) - fallback with API key
4. Haversine formula - straight-line distance fallback

## Key Workflows

### Running Optimization

**CLI (Recommended for development):**
```bash
# Activate environment first
source parcel_env/bin/activate  # or parcel_env\Scripts\activate on Windows

# Run simulator
python example/simulator.py

# Configuration files read from:
# - model/config/network_params.txt (depot hours, vehicle capacity)
# - model/config/model_params.txt (solver parameters, MIP gap)
# - model/config/simulation_params.txt (geocoding, visualization)
```

**Streamlit UI:**
```bash
streamlit run streamlit_app.py
```

**Flask API:**
```bash
python app.py  # Serves on http://localhost:5000
```

### Geocoding System

- `model/utils/geocoder.py` provides persistent caching in `data/geocode_cache.csv`
- **Never delete the cache** - geocoding is rate-limited and slow
- Normalizes postcodes (e.g., "98101.0" → "98101") before geocoding
- Priority: ORS API > Nominatim (respects `min_delay_seconds` rate limits)
- Coordinates in `vendor_latitude`, `vendor_longitude`, `recipient_latitude`, `recipient_longitude`

### CSV Data Structure

Expected columns (case-sensitive variants handled):
- `Vendor Name` / `vendor Name`
- `Vendor Gross Weight` / `Total Gross Weight` [kg]
- `Vendor Loading Meters` / `Calculated Loading Meters` / `Vendor Dimensions in m3` [m³]
- `Requested Loading Date` / `Requested Loading` [datetime]
- `Requested Delivery Date` / `Requested Delivery` [datetime]
- Address components: `Vendor Street`, `Vendor City`, `Vendor Postcode`, etc.

Date formats: Parse with `pd.to_datetime(errors='coerce')`, output as `'%Y-%m-%d %H:%M:%S'`

## Configuration Parameters

**Network Parameters** (`model/config/network_params.txt`):
```json
{
  "discretization_constant": 4,        // Time block size in hours
  "starting_depot": 8,                 // Depot opening hour
  "closing_depot": 18,                 // Depot closing hour
  "max_driving": 9,                    // Max driving hours per day
  "max_weight": 24,                    // Vehicle capacity [tons]
  "max_ldms": 13.6,                    // Vehicle volume [m³]
  "google_maps_api_key": null          // Optional: use Google Maps
}
```

**Auto-Scaling Behavior:**
- <20 vendors → CBC MIP (exact solution, may take minutes)
- ≥20 vendors → ALNS metaheuristic (heuristic, seconds to minutes)
- ALNS config: `max_iterations=2000`, `cooling_rate=0.997`, `initial_temperature=1500`

## Project-Specific Conventions

### Path Management
Always use absolute paths from project root. All modules add parent to `sys.path`:
```python
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))
```

### Solution Format
- `connections_solution` - List of route tuples: `[(vendor_id, time_period), ...]`
- `vehicles_solution` - Vehicle assignment per arc
- Routes saved to `uploads/processed_YYYYMMDD_HHMMSS.csv`
- Interactive maps generated with `folium` (multi-layer tiles, Excel-style route filters)

### Capacity Handling
- Weight: kg (convert to tons: `/1000`)
- Volume: "loading meters" (m³) - NOT linear meters
- Constraints enforced per route: `sum(cargo) <= max_capacity_kg[vehicle]`

### Visualization
- Maps use OSRM for real road routing (polylines via `folium.PolyLine`)
- Transparent route filter panels with Select All/Deselect All
- Tooltips show: cargo weight/volume, distance, duration, capacity utilization
- Same-location vendors get circular markers

## Common Pitfalls

1. **Geocoding Rate Limits:** Always use cached geocoder, never call APIs directly
2. **Date Format Mismatches:** Graph creator expects exact `'%Y-%m-%d %H:%M:%S'` format
3. **Column Name Variants:** Use defensive checks for both `Vendor Name` and `vendor Name`
4. **MIP Timeout:** Large instances (>20 vendors) may timeout - use ALNS instead
5. **Distance Matrix Size:** OSRM/Google have batch limits - requests chunked automatically
6. **Depot Index:** Depot is always `index 0` in distance/capacity matrices

## Testing & Dependencies

**Virtual Environment:**
```bash
python -m venv parcel_env
source parcel_env/bin/activate
pip install -r streamlit-requirements.txt  # For Streamlit
# OR install from pyproject.toml
pip install -e .
```

**Core Dependencies:**
- `ortools>=9.7.0` - CBC/GLOP solvers
- `pandas>=2.0.0`, `numpy>=1.24.0`
- `folium>=0.14.0` - Interactive maps
- `geopy>=2.4.0` - Geocoding
- `requests>=2.31.0` - OSRM/ORS API calls

**Python Version:** 3.8+ (tested up to 3.12)

## File Organization

```
model/
├── graph_creator/graph_creator.py  - Time-expanded network builder
├── optimizer/
│   ├── delivery_model.py           - MIP solver (OR-Tools CBC)
│   ├── alns_solver.py              - Metaheuristic (route-based)
│   ├── route_solution.py           - Route representation class
│   └── local_search.py             - ALNS operators
└── utils/
    ├── geocoder.py                 - Persistent geocoding cache
    ├── pre_processing.py           - Data normalization
    └── project_utils.py            - Utilities (logging, helpers)

data/geocode_cache.csv              - NEVER DELETE (persistent cache)
uploads/                            - Processed CSVs with solutions
results/optimization/               - Output artifacts
cache/                              - Distance matrix cache (JSON)
```

## External Integrations

- **OSRM:** Public demo server, no auth, handles `table` endpoint for matrices
- **Google Maps:** Distance Matrix API, requires key + enabled billing
- **ORS:** Free tier (5000 req/day), requires API key in `graph_creator.py`
- **Nominatim:** Geocoding fallback, 1 req/sec rate limit

## When Modifying Solvers

- **MIP changes:** Edit `delivery_model.py`, constraints use OR-Tools `pywraplp`
- **ALNS changes:** Operators in `alns_solver.py`, fitness in `route_solution.evaluate()`
- **New destroy operator:** Add to `self.destroy_operators` dict with weight=1.0
- **New repair operator:** Add to `self.repair_operators` dict
- **Testing:** Use `data/amazon_test_dataset_small.csv` (5 vendors) for quick iteration
