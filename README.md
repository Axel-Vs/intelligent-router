# 📦 Parcel Delivery Optimizer

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![OR-Tools](https://img.shields.io/badge/OR--Tools-9.7%2B-orange.svg)](https://developers.google.com/optimization)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A sophisticated optimization system for vehicle routing and parcel delivery, leveraging OR-Tools' CBC solver and real-world routing data from OSRM. The system optimizes delivery routes while considering vehicle capacity constraints, time windows, and real road networks.

## ✨ Features

- 🚚 **Multi-Vehicle Routing Optimization** - Efficiently assigns parcels to vehicles and optimizes delivery routes
- 🗺️ **Interactive Map Visualization** - Beautiful Folium-based maps with real road routing via OSRM
- ⏱️ **Time-Expanded Network** - Discrete time modeling for precise delivery scheduling
- 📊 **Comprehensive Analytics** - Detailed route statistics, capacity utilization, and performance metrics
- 🌍 **Real-World Routing** - Uses OSRM for accurate distance and travel time calculations
- 📍 **Geocoding Support** - Automatic address-to-coordinate conversion with caching
- 💡 **Flexible Solver** - CBC MIP solver with configurable optimization parameters

## 🎯 Key Capabilities

### Route Optimization
- Multi-vehicle fleet management
- Capacity constraints (weight and volume)
- Time windows and delivery scheduling
- Distance minimization
- Vehicle count optimization

### Visualization
- **Interactive Maps** with multiple tile layers (Street, Light, Dark, Terrain)
- **Route Tooltips** showing:
  - Step-by-step segment information
  - Cargo and loading pickup details
  - Distance, duration, and average speed
  - Complete route summaries
  - Capacity utilization percentages
- **Vendor Markers** displaying:
  - Cargo to pickup (weight and volume)
  - Assigned vehicle and stop number
  - Location details

### Data Processing
- CSV-based vendor and parcel data
- Geocoding with persistent cache
- Distance/time matrix calculation via OSRM
- Time discretization for optimization

## 📋 Prerequisites

- Python 3.8 or higher
- pip package manager
- Internet connection (for OSRM routing and geocoding)

## 🚀 Getting Started

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/Axel-Vs/parcel-delivery-solver.git
cd parcel-delivery-solver
```

2. **Create and activate virtual environment**
```bash
python -m venv parcel_env
source parcel_env/bin/activate  # On Windows: parcel_env\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Quick Start

1. **Prepare your data**
   - Place vendor data in `data/amazon_test_dataset.csv`
   - Ensure required columns: vendor coordinates, cargo weight, loading volume, delivery dates

2. **Configure parameters** (Optional)
   - Edit `model/config/model_params.txt` for optimization settings
   - Edit `model/config/network_params.txt` for network configuration
   - Edit `model/config/simulation_params.txt` for simulation parameters

3. **Run the simulator**
```bash
python example/simulator.py
```

4. **View results**
   - Solution summary printed to console
   - Interactive map saved to `results/optimization/routes_[date].html`
   - Solution arrays saved as `.npy` files

## 📊 Data Format

### Input CSV Structure
Your data file should include:

| Column | Description | Example |
|--------|-------------|---------|
| `vendor_latitude` | Vendor location latitude | 37.7749 |
| `vendor_longitude` | Vendor location longitude | -122.4194 |
| `recipient_latitude` | Delivery destination latitude | 47.6062 |
| `recipient_longitude` | Delivery destination longitude | -122.3321 |
| `vendor Name` | Vendor business name | Tech Company Inc. |
| `Vendor City` | Vendor city | San Francisco |
| `Vendor Postcode` | Vendor postal code | 94102 |
| `Total Shipment Weight (kg)` | Parcel weight in kilograms | 12438 |
| `Total Volume (cbm)` | Parcel volume in cubic meters | 31.5 |
| `Requested Delivery` | Delivery date/time | 2023-08-16 15:00:00 |

Example CSV:
```csv
vendor_latitude,vendor_longitude,recipient_latitude,recipient_longitude,vendor Name,Vendor City,Vendor Postcode,Total Shipment Weight (kg),Total Volume (cbm),Requested Delivery
37.7749,-122.4194,47.6062,-122.3321,Tech Company Inc.,San Francisco,94102,12438,31.5,2023-08-16 15:00:00
29.7604,-95.3698,47.6062,-122.3321,Manufacturing Co.,Houston,77001,13005,22.1,2023-08-17 14:45:00
```

## ⚙️ Configuration

### Model Parameters (`model/config/model_params.txt`)
```ini
max_nodes = 50                    # Maximum number of nodes to process
solver_time_limit = 900          # Solver timeout in seconds (15 minutes)
mip_gap_tolerance = 0.1          # MIP optimality gap (10%)
optimization_weight = 0.5        # Balance: distance (0.5) vs vehicle count (0.5)
```

### Network Parameters (`model/config/network_params.txt`)
```ini
discretization_constant = 4       # Time discretization in hours
max_driving_hours = 42           # Maximum driving time per vehicle
```

### Simulation Parameters (`model/config/simulation_params.txt`)
```ini
simulation_start_date = 2023-08-15 08:00:00
simulation_end_date = 2023-08-15 10:00:00
dataset_file = data/amazon_test_dataset.csv
```

## 🗺️ OSRM Integration

The system uses the free OSRM routing service for:
- **Distance Calculations** - Real road network distances
- **Travel Times** - Actual driving durations
- **Route Geometry** - Precise routing paths for visualization

✅ No API key required! Uses public OSRM server at `router.project-osrm.org`.

### Routing Features
- Real-world road network routing
- Automatic distance matrix generation
- Travel time estimation
- Route polyline extraction for maps

## 📈 Output & Visualization

### Console Output
```
================================================================================
                         📊 OPTIMIZATION SOLUTION SUMMARY
================================================================================

🚛 VEHICLE USAGE (y variables):
   ✓ y[1] = 1  → Vehicle 1 is USED
   ✓ y[2] = 1  → Vehicle 2 is USED
   📦 Total vehicles in solution: 2

🗺️  ROUTE ASSIGNMENTS (x variables):
🚚 Vehicle 1:
   Route: Vendor 1 (San Francisco) → Depot (Seattle)
   Distance: 1299 km
   Driving Time: 15.6 hrs
   Cargo: 12,438 kg | Loading: 31.5 m³

🚚 Vehicle 2:
   Route: Vendor 2 (Houston) → Vendor 3 (Chicago) → Depot (Seattle)
   Distance: 5027 km
   Driving Time: 55.7 hrs
   Cargo: 15,280 kg | Loading: 24.85 m³

Total Distance: 6327 km
Distance reduction achieved: 24.03%
```

### Interactive Map Features

**Hover over routes** to see:
- 🚚 Vehicle ID and route step (e.g., "Step 1/2")
- 📦 Cargo pickup: weight (kg) and volume (m³)
- 📏 Segment distance and duration
- 💨 Average speed
- 📊 Complete route summary with capacity utilization

**Click on vendors** to see:
- 🏭 Vendor information (name, city, postal code)
- 📦 Cargo to pickup (weight and volume)
- 🚚 Solution stage (assigned vehicle, stop number)

**Map Controls:**
- 🗺️ Multiple tile layers (Street Map, Light, Dark, Terrain)
- 🔍 Mini-map for context
- 📏 Distance measurement tool
- 🖥️ Fullscreen mode
- 📍 Mouse position coordinates

### Saved Files
- `results/optimization/routes_[date].html` - Interactive map visualization
- `results/optimization/solution[N]_[date].npy` - Numpy arrays with decision variables
- `data/geocode_cache.csv` - Cached geocoding results

## 🔧 Project Architecture

```
parcel-delivery-solver/
├── data/                          # Input datasets and cache
│   ├── amazon_test_dataset.csv   # Main dataset
│   ├── geocode_cache.csv         # Geocoding cache
│   └── *.csv                     # Additional test datasets
├── model/
│   ├── config/                   # Configuration files
│   │   ├── model_params.txt      # Solver parameters
│   │   ├── network_params.txt    # Network configuration
│   │   └── simulation_params.txt # Simulation settings
│   ├── graph_creator/            # Time-expanded network generation
│   │   └── graph_creator.py
│   ├── optimizer/                # Optimization model
│   │   └── delivery_model.py    # Main optimization logic
│   └── utils/                    # Utility functions
│       ├── geocoder.py           # Address geocoding
│       ├── pre_processing.py     # Data preprocessing
│       └── project_utils.py      # Distance/routing utilities
├── example/
│   └── simulator.py              # Main simulation script
├── results/                      # Output directory
│   └── optimization/             # Solution files and maps
├── docs/                         # Documentation
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🧮 Algorithms & Methods

### OR-Tools CBC Solver
The **Coin-or Branch and Cut (CBC)** solver is used for solving the vehicle routing problem as a Mixed Integer Programming (MIP) model.

**Key Features:**
- Linear programming with integer constraints
- Branch and cut algorithm for optimal solutions
- Configurable time limits and gap tolerance
- Efficient handling of large-scale problems

**Optimization Objectives:**
- Minimize total distance traveled
- Minimize number of vehicles used
- Balance trade-off via `optimization_weight` parameter

### Time-Expanded Network
Discretizes time into fixed periods (e.g., 4 hours) to:
- Model temporal constraints efficiently
- Handle delivery time windows
- Optimize arrival schedules
- Reduce problem complexity

**Time Discretization Formula:**
```python
discrete_time = ceil(travel_time_seconds / 3600 / discretization_constant)
```

### Distance Matrix Calculation
Uses OSRM for real-world routing:
1. Query OSRM table API with vendor/depot coordinates
2. Extract distance (meters) and duration (seconds)
3. Cache results for efficiency
4. Convert to discrete time periods

## 📊 Example Use Case

### Scenario
Optimize deliveries for 3 vendors across the US with 2 vehicles available.

### Input Data
| Vendor | Location | Weight | Volume | Destination |
|--------|----------|--------|--------|-------------|
| Vendor 1 | San Francisco, CA | 12,438 kg | 31.5 m³ | Seattle, WA |
| Vendor 2 | Houston, TX | 13,005 kg | 22.1 m³ | Seattle, WA |
| Vendor 3 | Chicago, IL | 2,275 kg | 2.8 m³ | Seattle, WA |

### Optimization Results
**Vehicle 1:**
- Route: San Francisco → Seattle
- Distance: 1,299 km
- Time: 15.6 hours
- Utilization: 41.5% weight, 45.0% volume

**Vehicle 2:**
- Route: Houston → Chicago → Seattle
- Distance: 5,027 km
- Time: 55.7 hours
- Utilization: 50.9% weight, 35.5% volume

**Performance:**
- Total distance: 6,327 km
- Distance reduction: 24.03% vs direct routing
- Vehicles used: 2 out of 3 available

## 🐛 Troubleshooting

### Common Issues

**Issue: "Depot coordinates not found"**
- Solution: The system uses Seattle (47.6062, -122.3321) as default. Configure depot coordinates in your dataset or modify the code.

**Issue: "OSRM connection timeout"**
- Solution: Check internet connection. OSRM requires online access to router.project-osrm.org

**Issue: "No feasible solution found"**
- Solutions:
  - Increase `solver_time_limit` in model_params.txt
  - Increase `max_driving_hours` in network_params.txt
  - Reduce dataset size or adjust vehicle capacities

**Issue: "Geocoding failed"**
- Solution: Ensure coordinates are provided in the CSV or check Nominatim service availability

## 🤝 Contributing

We welcome contributions! Here's how you can help:

### Ways to Contribute
- 🐛 Report bugs and issues
- 💡 Suggest new features
- 📝 Improve documentation
- 🔧 Submit pull requests

### Development Setup
1. Fork the repository
2. Create a feature branch
   ```bash
   git checkout -b feature/AmazingFeature
   ```
3. Make your changes and test
4. Commit with descriptive messages
   ```bash
   git commit -m 'feat: Add some AmazingFeature'
   ```
5. Push to your branch
   ```bash
   git push origin feature/AmazingFeature
   ```
6. Open a Pull Request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **[Google OR-Tools](https://developers.google.com/optimization)** - Powerful optimization solver
- **[OSRM](http://project-osrm.org/)** - Open Source Routing Machine
- **[Folium](https://python-visualization.github.io/folium/)** - Interactive map visualization
- **[Nominatim](https://nominatim.org/)** - OpenStreetMap geocoding service
- **[GeoPy](https://geopy.readthedocs.io/)** - Geocoding library

## 📧 Contact & Support

**Author:** Axel Vargas  
**GitHub:** [@Axel-Vs](https://github.com/Axel-Vs)  
**Project:** [parcel-delivery-solver](https://github.com/Axel-Vs/parcel-delivery-solver)

For questions, issues, or suggestions:
- 📫 Open an issue on GitHub
- 💬 Start a discussion
- 🐛 Report bugs with detailed information

## 📖 Citation

If you use this project in your research or work, please cite:

```bibtex
@software{parcel_delivery_optimizer,
  author = {Vargas, Axel},
  title = {Parcel Delivery Optimizer: A Vehicle Routing Solution with OR-Tools},
  year = {2025},
  url = {https://github.com/Axel-Vs/parcel-delivery-solver},
  note = {Mixed Integer Programming approach for multi-vehicle routing optimization}
}
```

## 📚 Additional Resources

- [OR-Tools Documentation](https://developers.google.com/optimization/routing)
- [Vehicle Routing Problem](https://en.wikipedia.org/wiki/Vehicle_routing_problem)
- [OSRM API Documentation](http://project-osrm.org/docs/v5.5.1/api/)
- [Time-Expanded Networks](https://en.wikipedia.org/wiki/Time-expanded_network)

---

<p align="center">
  <b>Made with ❤️ for efficient logistics and optimization</b><br>
  ⭐ Star this repo if you find it useful!
</p>
