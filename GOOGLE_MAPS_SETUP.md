# Google Maps Distance Matrix API Setup

The solver now supports using Google Maps API for calculating real road distances and travel times between locations.

## Setup Instructions

### 1. Get a Google Maps API Key

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select an existing one
3. Enable the **Distance Matrix API**:
   - Go to "APIs & Services" > "Library"
   - Search for "Distance Matrix API"
   - Click "Enable"
4. Create credentials:
   - Go to "APIs & Services" > "Credentials"
   - Click "Create Credentials" > "API Key"
   - Copy your API key

### 2. Configure the API Key

Edit `model/config/network_params.txt` and add your API key:

```json
{
  "discretization_constant": 4,
  ...
  "google_maps_api_key": "YOUR_API_KEY_HERE"
}
```

### 3. Run the Solver

The solver will automatically use Google Maps when a valid API key is configured:

```bash
parcel_env/bin/python example/simulator.py
```

You should see in the output:
```
Using Google Maps for distance/time calculations
Calculating distances using Google Maps API...
Google Maps distance calculation completed
```

## Cost Considerations

- Google Maps Distance Matrix API charges per element (origin-destination pair)
- First $200/month is free (approximately 40,000 elements)
- After that: $0.005 per element
- For a 10-node problem: 10×10 = 100 elements per run

## Fallback Behavior

If no Google Maps API key is provided or if there's an error:
1. System tries OpenRouteService API
2. If that fails, falls back to Haversine (straight-line) distance calculation

## Advantages of Google Maps

- **Real road distances**: Actual driving routes, not straight-line
- **Accurate travel times**: Based on typical traffic conditions
- **Up-to-date data**: Current road networks and conditions
- **Truck routing**: Respects truck restrictions and weight limits

## Example Comparison

**Haversine (straight-line):**
- San Francisco to Los Angeles: ~550 km
- Travel time estimate: 9.2 hours (at 60 km/h)

**Google Maps (actual route):**
- San Francisco to Los Angeles: ~615 km (via I-5)
- Travel time: 5.5-6.5 hours (actual highway speeds)

## Troubleshooting

**"Using Haversine distance calculation" message:**
- Check that `google_maps_api_key` is set in `network_params.txt`
- Verify the API key is correct
- Ensure Distance Matrix API is enabled in Google Cloud Console
- Check you have remaining quota

**API quota exceeded:**
- Monitor usage in Google Cloud Console
- Consider implementing caching for repeated runs
- Increase billing limit if needed
