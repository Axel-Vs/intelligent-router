# Parcel Delivery Solver - System Quick Reference

**Last Updated:** December 18, 2025  
**Version:** 2.0 - Condensed for AI Context  
**Project:** VRP Optimization Platform with Flask Web UI

> **Purpose**: Concise technical reference for understanding system architecture, components, and workflows. All critical information preserved in minimal format.

---

## Quick Navigation

[Architecture](#architecture) | [Entry Points](#entry-points) | [Core Components](#core-components) | [Configuration](#configuration) | [Data Formats](#data-formats) | [Workflows](#workflows) | [Troubleshooting](#troubleshooting)

---

## Architecture

**System Flow**: CSV → Geocoding → Time-Expanded Network → Solver Selection → Route Optimization → Interactive Map + CSV Export

**Auto-Scaling**: <20 vendors = CBC MIP (exact), ≥20 vendors = ALNS metaheuristic (fast)

**Routing Providers** (priority): OSRM (default, free) → Google Maps → OpenRouteService → Haversine fallback

**Data Caching**: 
- `data/geocode_cache.csv` - **NEVER DELETE** (permanent address coordinates)
- `cache/<hash>.json` - Distance matrices (auto-generated)

## Entry Points

### 1. Flask Web UI (Primary) - `app.py`
```bash
python app.py  # http://localhost:8080
```
**Features**: 3 tabs (Optimizer, Saved Runs, Route Visualization) | Black tab navigation | Clickable "Intelligent Router" title

**Critical Routes**:
- `POST /api/optimize` - Run optimization with parameters
- `POST /api/runs/load` - Load past run for re-optimization  
- `GET /api/runs/download-input/<run_id>` - Download original CSV
- `DELETE /api/runs/<run_id>` - Delete saved run

**Recent Bug Fixes**:
- Period variable initialization (`period = None` at function start)
- Conditional column mapping (handles both JSON uploads and CSV reloads)
- CSV path fix (full path: `results/runs/{runId}/input.csv`)

### 2. Streamlit UI - `streamlit_app.py`
```bash
streamlit run streamlit_app.py
```
Simplified UI for quick optimization testing.

### 3. CLI Simulator - `example/simulator.py`
```bash
python example/simulator.py
```
Batch processing. Reads config from `model/config/*.txt`

## Core Components

### Graph Creator - `model/graph_creator/graph_creator.py`
**Purpose**: Time-expanded network builder  
**Key Params**: `discretization_constant=4` (hours), `max_feasible_distance`, `time_window_sampling_threshold`  
**Output**: Graph with (location, time) nodes and feasible arcs

### Solvers

**1. MIP (CBC) - `model/optimizer/delivery_model.py`**
- OR-Tools CBC backend | Binary decision variables per arc
- Config: `mip_gap=0.1` (10% tolerance)
- Use: <20 vendors, exact solutions
- Output: `[(vendor_id, time_period), ...]` tuples

**2. ALNS - `model/optimizer/alns_solver.py`**
- Adaptive Large Neighborhood Search | Route-based (not tensors)
- Operators: random/worst_cost/shaw/greedy/regret2 destroy + greedy/regret repair
- Config: `max_iterations=2000`, `cooling_rate=0.997`, `initial_temperature=1500`
- Use: ≥20 vendors, fast heuristics
- Output: Route objects with full details

### Geocoder - `model/utils/geocoder.py`
**Cache**: `data/geocode_cache.csv` (**NEVER DELETE**)  
**Providers**: ORS → Nominatim (rate-limited to 1 req/sec)  
**Features**: Postcode normalization, persistent caching

**Providers**:
1. **OpenRouteService** (ORS) - Primary for batch lookups
2. **Nominatim** - Free Nominatim geocoding
3. **Manual Override** - Direct coordinate columns in CSV

**Output**: (latitude, longitude) tuples

### 5. Pre-processing (`model/utils/pre_processing.py`)

**Purpose**: Normalize and validate input data

**Tasks**:
- Parse dates (flexible format handling)
- Normalize column names (handle case variants)
- Extract address components
- Validate geocoding columns
- Calculate cargo weight/volume
- Clean numeric fields

**Output**: Standardized pandas DataFrame

### 6. Route Solution (`model/optimizer/route_solution.py`)

**Purpose**: Represents and evaluates optimization solution

**Attributes**:
- `routes` - List of route objects
- `total_cost` - Sum of all route costs
- `total_distance` - Sum of all distances
- `total_time` - Total optimization time

**Methods**:
- `evaluate()` - Calculate fitness/cost
- `is_feasible()` - Check constraint violations
- `get_route_details()` - Extract route information

### 7. Local Search (`model/optimizer/local_search.py`)

**Purpose**: Implements ALNS destroy and repair operators

**Key Methods**:
- Destroy operator selection with adaptive weights
- Repair operator selection
- 2-opt local search enhancement
- Move evaluation

---

## Entry Points

### 1. Web Interface (Flask) - `app.py`

**Start Command**:
```bash
python app.py
```

**URL**: `http://localhost:8080`

**Features**:
- **3-Tab Layout**: Optimizer, Saved Runs, Route Visualization
- **Clickable Branding**: Click "Intelligent Router" title to return to Optimizer
- **Black Tab Navigation**: Modern dark tabs with white active highlights
- **Interactive Optimizer UI**: 12 configurable parameters with real-time results
- **Saved Runs Management**: 
  - Browse all past optimization runs in comparison table
  - Three action buttons per run: View Map, Input Data CSV, Route Solution CSV
  - Re-optimization: load past runs back to Optimizer
  - Delete functionality for unwanted runs
- **Route Solution CSV Export**: 14-column detailed solution with vendor/recipient details
- **Real-time Visualization**: Interactive maps with layer controls
- **REST API Backend**: Full API for programmatic access

**Key Routes**:
- `GET /` - Main interface
- `POST /api/optimize` - Run optimization
- `GET /api/runs` - List all saved runs (returns JSON array)
- `POST /api/runs/load` - Load past run for re-optimization
- `GET /api/runs/download-input/<run_id>` - Download original input CSV
- `DELETE /api/runs/<run_id>` - Delete saved run
- `POST /api/save_run` - Save optimization result
- `GET /api/map/<run_id>` - Get interactive map HTML

**Solution CSV Format** (14 columns):
- Route Number, Stop Sequence, Stop Type
- Vendor ID, Name, City, Address
- Recipient Name, City, Address
- Cargo Weight (kg), Volume (m³)
- Requested Delivery Date, Requested Loading Date

### 2. Streamlit UI - `streamlit_app.py`

**Start Command**:
```bash
streamlit run streamlit_app.py
```

**Features**:
- Simplified UI for optimization
- Interactive data upload
- Parameter adjustment
- Results visualization

### 3. CLI Simulator - `example/simulator.py`

**Start Command**:
```bash
python example/simulator.py
```

**Features**:
- Batch processing
- Configuration file-based
## Configuration

### Network Params - `model/config/network_params.txt`
```json
{
  "discretization_constant": 4,    // Time blocks (hours) - larger = faster
  "starting_depot": 8,              // Depot hours: 8am-6pm
  "closing_depot": 18,
  "max_driving": 9,                 // Regulatory compliance
  "max_weight": 24,                 // Vehicle capacity (tons)
  "max_ldms": 13.6,                // Vehicle volume (m³)
  "google_maps_api_key": null      // Optional: enables Google routing
}
```

### Model Params - `model/config/model_params.txt`
```json
{
  "mip_gap": 0.1,                  // CBC: 10% optimality tolerance
  "time_limit": 300,               // CBC: 5min timeout
  "alns_max_iterations": 2000,     // ALNS: search depth
  "cooling_rate": 0.997,           // ALNS: simulated annealing
  "initial_temperature": 1500      // ALNS: starting temp
}
```

## Data Formats

### Input CSV (Required Columns - case-insensitive)
```
Vendor Name, Street, City, Postcode, Latitude*, Longitude*
Vendor Gross Weight (kg), Loading Meters (m³)
Requested Loading Date, Requested Delivery Date
Recipient Street, City, Postcode, Latitude*, Longitude*
*optional - overrides geocoding
```
Dates: `pd.to_datetime(errors='coerce')` → `'%Y-%m-%d %H:%M:%S'`

### Solution CSV (14 columns - from Saved Runs "Route Solution CSV")
```
Route Number, Stop Sequence, Stop Type
Vendor ID, Name, City, Address
Recipient Name, City, Address
Cargo Weight (kg), Volume (m³)
Requested Delivery Date, Loading Date
```

### Processing CSV (`uploads/processed_YYYYMMDD_HHMMSS.csv`)
```
vendor_id, vendor_name, route_number, sequence_in_route
departure/arrival_time, lat, lon
distance_to_next_km, duration_to_next_minutes
cumulative_weight_kg, cumulative_volume_m3
```

## Workflows

### Optimization Pipeline
```
CSV Input → Data Normalization → Geocoding (cache check → ORS/Nominatim)
→ Time-Expanded Network (discretize time, create nodes/arcs, prune)
→ Solver Selection (<20: CBC MIP | ≥20: ALNS)
→ Route Generation (stats, distances, capacity)
→ Folium Map + CSV Export
→ Save to results/runs/{run_id}/
```

### Saved Runs Management (web/index.html)
```
Saved Runs Tab:
├─ Browse: Comparison table with routes/vendors/distance/cargo/volume/time
├─ Actions per run:
│  ├─ View Map: Open interactive visualization
│  ├─ Input Data CSV: Download original dataset
│  └─ Route Solution CSV: 14-column detailed solution
├─ Re-optimize: Load run → adjust params → re-run
└─ Delete: Remove from results/runs/
```

### File Storage Structure
```
results/runs/{run_id}/
├─ input.csv              # Original upload
├─ output.csv             # Route solution
├─ map.html               # Interactive visualization
└─ metadata.json          # Parameters + timestamp
```

### Past Run Reload (Bug Fixed Dec 2025)
```
1. User selects past run from dropdown
2. System loads: results/runs/{runId}/input.csv (full path required)
3. Column mapping conditionals handle both JSON and CSV formats
4. Period variable initialized (period = None)
5. Re-run optimization with adjusted parameters
   ├─ Map HTML
   ├─ Input CSV
   ├─ Optimization parameters
   └─ Timestamp
3. Data stored in:
   ├─ uploads/processed_YYYYMMDD_HHMMSS.csv
   ├─ results/optimization/map_YYYYMMDD_HHMMSS.html
   └─ Database entry (run metadata)
4. Run becomes available in "Saved Runs" tab
```

### Load Saved Run Workflow

```
1. User clicks "View Map" or "Download"
2. System retrieves:
   ├─ Run metadata from database
   ├─ Corresponding CSV file
   ├─ Corresponding map file
   └─ Route statistics
3. Display/download appropriate file
```

---

## External APIs

### Routing (Priority Order)
1. **OSRM** (default, free) - `http://router.project-osrm.org/table` - No auth
2. **Google Maps** - If `google_maps_api_key` in network_params.txt
3. **OpenRouteService** - 5000 req/day free, needs API key
4. **Haversine** - Fallback straight-line distance

### Geocoding (Priority Order)
1. **Cache Check** - `data/geocode_cache.csv` first
2. **ORS API** - 1 req/sec rate limit
3. **Nominatim** - 1 req/sec rate limit
Postcodes normalized: "98101.0" → "98101"

## File Structure (Key Paths)
```
app.py                              # Flask web server (localhost:8080)
streamlit_app.py                    # Streamlit alternative UI
web/index.html                      # 3-tab UI (2300+ lines)

model/
├── config/                         # *.txt config files
├── graph_creator/graph_creator.py  # Time-expanded network
├── optimizer/
│   ├── delivery_model.py           # CBC MIP (<20 vendors)
│   └── alns_solver.py              # ALNS metaheuristic (≥20)
└── utils/geocoder.py               # Caching geocoder

data/geocode_cache.csv              # *** NEVER DELETE ***
cache/<hash>.json                   # Distance matrices
results/runs/{run_id}/              # Saved run storage
uploads/processed_*.csv             # Route outputs
```

## Troubleshooting

### Empty Saved Runs Tab
Hard refresh: `Cmd+Shift+R` (Mac) / `Ctrl+Shift+F5` (Win) | Check: `curl http://localhost:8080/api/runs`

### Optimization Too Slow (>5min)
Increase `mip_gap` to 0.2 or `time_limit` to 120s in `model/config/model_params.txt` | Check console for solver: "Using CBC MIP" vs "Using ALNS"

### Geocoding Rate Limit
Increase `min_delay_seconds` to 2 in `geocoder.py` | Check cache: `wc -l data/geocode_cache.csv`

### CSV Errors
Required: Vendor Name, Gross Weight, Loading Meters, Loading/Delivery dates | Dates must parse with `pd.to_datetime()` | Weight/volume must be numeric

### Map Not Displaying
Check: `ls -la results/optimization/map_*.html` | File should be >50KB | Browser console (F12) for errors

### Past Run Reload Issues (Fixed Dec 2025)
- Period variable initialization: `period = None` at function start
- Column mapping conditionals handle JSON + CSV formats
- Full CSV path required: `results/runs/{runId}/input.csv`

## Development Quick Reference

### Adding ALNS Destroy Operator
```python
# In model/optimizer/alns_solver.py
def new_destroy(self, route):
    # Custom logic
    return modified_route, removed_customers

# Register: self.destroy_operators['new_op'] = (self.new_destroy, 1.0)
# Test: python example/simulator.py --dataset data/amazon_test_dataset_small.csv
```

### Adding Solution CSV Column
Modify route generation in `delivery_model.py` or `route_solution.py` → Update CSV writer in `app.py`

### Testing
```bash
# Quick test (5 vendors, MIP)
python example/simulator.py --dataset data/amazon_test_dataset_small.csv

# Performance comparison
time python example/simulator.py --dataset data/amazon_test_dataset_medium.csv
```

### Performance Tips
- **MIP**: Increase `mip_gap` (0.1→0.2) or `time_limit` for faster solve
- **ALNS**: Increase `max_iterations`, adjust `cooling_rate` (lower=longer search)
- **Graph**: Reduce `discretization_constant` for fewer time periods
- **Caching**: `geocode_cache.csv` + `cache/<hash>.json` reused automatically

---

**Version 2.0** - Condensed Dec 2025 for AI context efficiency  
*Full details: README.md | CHANGELOG.md | docs/DOCUMENTATION_UPDATE_SUMMARY.md*
