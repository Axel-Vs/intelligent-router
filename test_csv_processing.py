import pandas as pd
import os
import sys

# City coordinate fallbacks for common cities
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

print("Testing CSV processing...\n")

# Load CSV
csv_path = 'data/amazon_test_dataset.csv'
df = pd.read_csv(csv_path)
print(f"✅ Loaded {len(df)} rows")

# Check column normalization
if 'Vendor Name' in df.columns:
    df['vendor Name'] = df['Vendor Name'].astype(str)
    print("✅ Normalized 'Vendor Name' to 'vendor Name'")

# Check geocoding
vendor_cities = df['Vendor City'].str.strip().unique()
print(f"\n📍 Found {len(vendor_cities)} unique vendor cities:")
for city in vendor_cities:
    if city in city_coords:
        print(f"   ✅ {city} - Will use city fallback")
    else:
        print(f"   ⚠️  {city} - Needs Nominatim geocoding")

# Check dates
print(f"\n📅 Date range:")
loading_dates = pd.to_datetime(df['Requested Loading Date'], errors='coerce')
delivery_dates = pd.to_datetime(df['Requested Delivery Date'], errors='coerce')
print(f"   Loading: {loading_dates.min()} to {loading_dates.max()}")
print(f"   Delivery: {delivery_dates.min()} to {delivery_dates.max()}")

# Check data quality
print(f"\n📊 Data quality:")
print(f"   Total weight: {df['Vendor Gross Weight'].sum():.0f} kg")
print(f"   Total volume: {df['Vendor Dimensions in m3'].sum():.1f} m³")
print(f"   Total loading meters: {df['Vendor Loading Meters'].sum():.1f} m")

print("\n✅ CSV is ready for optimization!")
