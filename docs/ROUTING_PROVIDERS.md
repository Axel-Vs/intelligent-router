# Routing Providers Guide

This document explains the routing providers available in the parcel delivery solver and how to configure them.

## Overview

The system uses real road distances and travel times for optimization. Multiple routing providers are supported with automatic fallback:

1. **OSRM** (Primary) - Completely free, unlimited
2. **Google Maps** (Optional) - Requires API key, $0.005/element after free tier
3. **OpenRouteService (ORS)** (Fallback) - Free with rate limits
4. **Haversine** (Final fallback) - Straight-line distance estimation

## Provider Details

### 1. OSRM (Open Source Routing Machine) ⭐ RECOMMENDED

**Advantages:**
- ✅ **Completely free** - No API key needed
- ✅ **Unlimited requests** - No rate limits or quotas
- ✅ **Real road distances** - Uses actual road network data
- ✅ **No account required** - Works out of the box
- ✅ **Fast response times** - Public demo server is reliable

**How it works:**
- Uses the public OSRM demo server: `http://router.project-osrm.org`
- Calculates actual driving distances and times
- Automatically selected as the primary provider

**Usage:**
No configuration needed! OSRM is automatically used by default.

**Example Output:**
```
🌐 Calculating distances using OSRM (free, real road distances)...
✅ OSRM calculation completed successfully
Distance Vehicle 1: 1299 km  (real road distance)
Distance Vehicle 2: 3300 km  (real road distance)
Total Distance: 4600 km
```

### 2. Google Maps Distance Matrix API

**Advantages:**
- ✅ Highly accurate routing
- ✅ Real-time traffic data (if enabled)
- ✅ Global coverage

**Disadvantages:**
- ❌ Requires API key and billing account
- ❌ Costs $0.005 per element after $200/month free tier
- ❌ Complex setup process

**Setup:**
1. Create Google Cloud project
2. Enable Distance Matrix API
3. Create API key
4. Add billing information
5. Set API key in `model/config/network_params.txt`:
   ```
   google_maps_api_key: YOUR_API_KEY_HERE
   ```

See [GOOGLE_MAPS_SETUP.md](../GOOGLE_MAPS_SETUP.md) for detailed instructions.

**Cost Example:**
- 10 locations = 100 elements (10×10 matrix)
- Cost: $0.50 per matrix calculation
- 100 matrices/month = $50/month

### 3. OpenRouteService (ORS)

**Advantages:**
- ✅ Free tier available
- ✅ Open source alternative

**Disadvantages:**
- ❌ Rate limits on free tier
- ❌ May return zero distances in some cases
- ❌ Less reliable than OSRM or Google Maps

**Usage:**
Automatically used as fallback if OSRM fails.

### 4. Haversine Distance (Final Fallback)

**Advantages:**
- ✅ Always available
- ✅ No API calls needed
- ✅ Fast calculation

**Disadvantages:**
- ❌ Straight-line distance only (as the crow flies)
- ❌ Not real road distances
- ❌ Can underestimate actual travel distance by 20-40%

**Formula:**
```python
d = 2 × R × arcsin(√(sin²((lat2-lat1)/2) + cos(lat1) × cos(lat2) × sin²((lon2-lon1)/2)))
```
where R = 6371 km (Earth's radius)

## Provider Selection Logic

The system automatically tries providers in this order:

```
1. Try OSRM (free)
   ↓ (if fails)
2. Try Google Maps (if API key configured)
   ↓ (if fails or not configured)
3. Try ORS
   ↓ (if fails)
4. Use Haversine fallback
```

## Comparison

| Feature | OSRM | Google Maps | ORS | Haversine |
|---------|------|-------------|-----|-----------|
| **Cost** | Free | $0.005/element | Free (limited) | Free |
| **API Key** | No | Yes | No | No |
| **Real Roads** | ✅ | ✅ | ✅ | ❌ |
| **Accuracy** | High | Very High | Medium | Low |
| **Rate Limits** | None | High | Medium | None |
| **Setup** | None | Complex | None | None |
| **Recommendation** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐ |

## Current Configuration

The system is currently using:
- **Primary Provider:** OSRM (free, unlimited)
- **Fallback Chain:** Google Maps → ORS → Haversine

## Logs

To see which provider is being used, check the console output:

**OSRM Success:**
```
🌐 Calculating distances using OSRM (free, real road distances)...
✅ OSRM calculation completed successfully
```

**OSRM Failure (fallback to alternatives):**
```
🌐 Calculating distances using OSRM (free, real road distances)...
OSRM failed (Network error), trying alternative providers...
Calculating distances using Google Maps API...
```

## Technical Details

### OSRM API Call
```python
# Coordinates format: lon,lat;lon,lat;lon,lat
coords_str = '-122.4194,37.7749;-87.6298,41.8781'
url = f'http://router.project-osrm.org/table/v1/driving/{coords_str}'
params = {'annotations': 'distance,duration'}

response = requests.get(url, params=params, timeout=30)
data = response.json()

# Returns:
# - distances: matrix in meters
# - durations: matrix in seconds
```

### Distance Matrix Structure
```python
# Example 3x3 matrix (Depot + 2 Vendors)
distance_matrix = [
    [0,    1299, 3300],  # From Depot
    [1299, 0,    2800],  # From Vendor 1
    [3300, 2800, 0]      # From Vendor 2
]
# Units: kilometers
```

## Troubleshooting

### "OSRM failed" Error
**Cause:** Network connectivity issue or OSRM server temporarily down

**Solution:** The system will automatically fall back to other providers. If persistent:
1. Check internet connection
2. Try again in a few minutes
3. OSRM public server is very reliable, outages are rare

### Zero Distances
**Cause:** ORS or dummy client returning invalid data

**Solution:** System automatically falls back to Haversine. Consider:
1. Using OSRM (should not have this issue)
2. Configuring Google Maps API for guaranteed accuracy

### High Costs with Google Maps
**Cause:** Large matrices or many optimization runs

**Solution:** 
1. Use OSRM instead (completely free!)
2. Reduce matrix size by clustering nearby locations
3. Cache distance matrices for repeated calculations

## Best Practices

1. **For Production:** Use OSRM (free and reliable)
2. **For Maximum Accuracy:** Add Google Maps API key as backup
3. **For Testing:** OSRM is perfect - no setup needed
4. **For Offline:** Use Haversine (included automatically as final fallback)

## References

- **OSRM Documentation:** http://project-osrm.org/
- **OSRM Demo Server:** http://router.project-osrm.org/
- **Google Maps API:** https://developers.google.com/maps/documentation/distance-matrix
- **OpenRouteService:** https://openrouteservice.org/

---

**Last Updated:** 2024
**Current Provider:** OSRM (default, free, unlimited)
