import os
import os
import time
import requests
import pandas as pd
from typing import Tuple, List, Optional

# Try to import geopy-specific exception types for more precise handling
try:
    from geopy.exc import GeocoderTimedOut, GeocoderInsufficientPrivileges
    from geopy.adapters import AdapterHTTPError
except Exception:
    GeocoderTimedOut = TimeoutError
    GeocoderInsufficientPrivileges = Exception
    AdapterHTTPError = Exception


class Geocoder:
    """Geocoder wrapper with caching, sanitization, and retry/backoff.

    Uses geopy.Nominatim when available. Falls back to a no-op geocoder that
    returns default coordinates if geopy is not installed.
    """

    def __init__(
        self,
        backend: str = "nominatim",
        api_key: Optional[str] = None,
        cache_path: str = "data/geocode_cache.csv",
        user_agent: str = "parcel_geocoder",
        min_delay_seconds: float = 1.0,
        timeout: int = 10,
        max_attempts: int = 3,
    ):
        self.backend = backend
        self.api_key = api_key
        self.cache_path = cache_path
        self.user_agent = user_agent
        self.min_delay_seconds = min_delay_seconds
        self.timeout = timeout
        self.max_attempts = max_attempts

        # ensure cache exists
        self.cache = {}
        self._load_cache()

        # ORS key (optional) - prefer ORS if available
        self.ors_key = os.environ.get('ORS_API_KEY')
        if self.ors_key:
            # prefer ORS backend when key available
            self.backend = 'ors'

        # try to setup geopy backend
        try:
            from geopy.geocoders import Nominatim
            from geopy.extra.rate_limiter import RateLimiter

            # Nominatim usage policy asks for a descriptive user_agent including contact info
            self._geolocator = Nominatim(user_agent=self.user_agent)
            # RateLimiter will apply min_delay_seconds between calls
            self._geocode = RateLimiter(
                self._geolocator.geocode, min_delay_seconds=self.min_delay_seconds, max_retries=1
            )
            self._available = True
        except Exception:
            self._geocode = None
            self._available = False

    def _load_cache(self) -> None:
        """Load cache from CSV; be defensive about file format and NaNs."""
        if not isinstance(self.cache, dict):
            self.cache = {}

        if os.path.exists(self.cache_path):
            try:
                df = pd.read_csv(self.cache_path)
                if "address" in df.columns and "lat" in df.columns and "lon" in df.columns:
                    for _, r in df.iterrows():
                        addr = r.get("address")
                        if pd.isna(addr) or str(addr).strip() == "":
                            continue
                        lat = r.get("lat")
                        lon = r.get("lon")
                        if pd.isna(lat):
                            lat = None
                        if pd.isna(lon):
                            lon = None
                        self.cache[str(addr)] = (lat, lon)
            except Exception:
                # ignore malformed cache
                self.cache = {}

    def _append_cache(self, entries: List[dict]) -> None:
        if not entries:
            return
        df = pd.DataFrame(entries)
        try:
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            if os.path.exists(self.cache_path):
                df.to_csv(self.cache_path, mode="a", header=False, index=False)
            else:
                df.to_csv(self.cache_path, index=False)
        except Exception:
            pass

    def _geocode_ors(self, address: str) -> Tuple[Optional[float], Optional[float]]:
        """Geocode using OpenRouteService Search API (requires ORS_API_KEY in env)."""
        if not self.ors_key:
            return None, None
        try:
            url = 'https://api.openrouteservice.org/geocode/search'
            headers = {'Authorization': self.ors_key}
            params = {'text': address, 'size': 1}
            resp = requests.get(url, headers=headers, params=params, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            features = data.get('features') or []
            if features:
                coords = features[0].get('geometry', {}).get('coordinates')
                if coords and len(coords) >= 2:
                    lon, lat = coords[0], coords[1]
                    return lat, lon
        except Exception:
            return None, None
        return None, None

    def _sanitize_value(self, val, colname: str) -> str:
        """Sanitize a single column value for address building.

        - Coerce float postcodes like 98101.0 -> '98101'
        - Strip whitespace and trailing punctuation
        """
        if pd.isna(val):
            return ""
        # postcode-like columns
        if "post" in colname.lower() or "zip" in colname.lower():
            try:
                if isinstance(val, float) and val.is_integer():
                    return str(int(val))
                s = str(val)
                if s.endswith(".0"):
                    return s[:-2]
            except Exception:
                pass

        s = str(val).strip()
        s = s.rstrip(".,")
        return s

    def _build_address(self, row: dict, cols: List[str]) -> str:
        parts: List[str] = []
        for c in cols:
            v = row.get(c, "")
            v_clean = self._sanitize_value(v, c)
            if v_clean == "":
                continue
            parts.append(v_clean)
        return ", ".join(parts)

    def geocode_address(self, address: str, force_refresh: bool = False) -> Tuple[Optional[float], Optional[float]]:
        """Return (lat, lon) for an address, using cache then backend.

        If force_refresh=True, ignore existing cache and re-query the backend.
        """
        if not address or str(address).strip() == "":
            return None, None
        # normalize address string
        address = str(address).strip()

        # Cached value
        if not force_refresh and address in self.cache:
            cached = self.cache[address]
            if cached is not None and not (pd.isna(cached[0]) and pd.isna(cached[1])):
                return cached
            # if cached is (None,None) but backend unavailable, return it
            if not self._available:
                return cached

        if not self._available:
            # record negative cache and return
            self.cache[address] = (None, None)
            return None, None

        # Try with exponential backoff
        # If ORS key is present and we prefer ORS, try ORS first
        if getattr(self, 'ors_key', None):
            lat, lon = self._geocode_ors(address)
            if lat is not None and lon is not None:
                self.cache[address] = (lat, lon)
                # persist only successful lookups to avoid polluting cache with negatives
                self._append_cache([{"address": address, "lat": lat, "lon": lon}])
                return lat, lon

        attempt = 0
        lat = lon = None
        while attempt < self.max_attempts:
            try:
                loc = self._geocode(address, timeout=self.timeout)
                if loc:
                    lat, lon = loc.latitude, loc.longitude
                else:
                    lat, lon = None, None
                break
            except Exception as ex:
                # only retry on timeout-like errors
                name = type(ex).__name__
                msg = str(ex)
                # If we get HTTP 403 / insufficient privileges, disable backend to avoid spamming
                if isinstance(ex, GeocoderInsufficientPrivileges) or "403" in msg or "Forbidden" in msg:
                    self._available = False
                    lat, lon = None, None
                    break
                # AdapterHTTPError may represent transient server-side errors (502, 503) or rate limits (429).
                # Treat common server/rate errors as retryable with exponential backoff.
                if isinstance(ex, AdapterHTTPError) or any(code in msg for code in ("429", "502", "503")):
                    sleep_for = (2 ** attempt) * 1.0
                    time.sleep(sleep_for)
                    attempt += 1
                    continue
                # handle explicit geopy timeout exception or standard TimeoutError
                if isinstance(ex, GeocoderTimedOut) or isinstance(ex, TimeoutError) or "TimedOut" in name:
                    sleep_for = (2 ** attempt) * 1.0
                    time.sleep(sleep_for)
                    attempt += 1
                    continue
                # other errors: stop retrying
                lat, lon = None, None
                break

        # cache result in memory
        self.cache[address] = (lat, lon)
        # persist only successful lookups (avoid writing (None,None) rows)
        if lat is not None and lon is not None:
            self._append_cache([{"address": address, "lat": lat, "lon": lon}])
        return lat, lon

    def geocode_dataframe(
        self,
        df: pd.DataFrame,
        vendor_cols: Optional[List[str]] = None,
        recipient_cols: Optional[List[str]] = None,
        default_coords: Tuple[Optional[float], Optional[float]] = (None, None),
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        if vendor_cols is None:
            vendor_cols = [
                "Vendor Street",
                "Vendor Postcode",
                "Vendor City",
                "Vendor Country Name",
            ]
        if recipient_cols is None:
            recipient_cols = [
                "Recipient Street",
                "Recipient Postcode",
                "Recipient City",
                "Recipient Country Name",
            ]

        # Step 1: Build all addresses first
        vendor_addresses = []
        recipient_addresses = []
        for _, row in df.iterrows():
            vaddr = self._build_address(row, vendor_cols)
            if not vaddr:
                vaddr = self._build_address(row, [c.lower() for c in vendor_cols])
            vendor_addresses.append(vaddr)
            
            raddr = self._build_address(row, recipient_cols)
            if not raddr:
                raddr = self._build_address(row, [c.lower() for c in recipient_cols])
            recipient_addresses.append(raddr)

        # Step 2: Lookup all addresses in cache first (VLOOKUP style)
        vendor_lons: List[Optional[float]] = []
        vendor_lats: List[Optional[float]] = []
        recip_lons: List[Optional[float]] = []
        recip_lats: List[Optional[float]] = []
        
        missing_vendor_indices = []
        missing_recip_indices = []

        for idx, (vaddr, raddr) in enumerate(zip(vendor_addresses, recipient_addresses)):
            # Vendor lookup
            if vaddr and vaddr in self.cache:
                cached_v = self.cache[vaddr]
                if cached_v and cached_v[0] is not None and cached_v[1] is not None:
                    vendor_lats.append(cached_v[0])
                    vendor_lons.append(cached_v[1])
                else:
                    vendor_lats.append(None)
                    vendor_lons.append(None)
                    missing_vendor_indices.append(idx)
            else:
                vendor_lats.append(None)
                vendor_lons.append(None)
                missing_vendor_indices.append(idx)
            
            # Recipient lookup
            if raddr and raddr in self.cache:
                cached_r = self.cache[raddr]
                if cached_r and cached_r[0] is not None and cached_r[1] is not None:
                    recip_lats.append(cached_r[0])
                    recip_lons.append(cached_r[1])
                else:
                    recip_lats.append(None)
                    recip_lons.append(None)
                    missing_recip_indices.append(idx)
            else:
                recip_lats.append(None)
                recip_lons.append(None)
                missing_recip_indices.append(idx)

        # Step 3: Geocode only the missing addresses
        new_cache_entries = []
        
        for idx in missing_vendor_indices:
            row = df.iloc[idx]
            vaddr = vendor_addresses[idx]
            
            vlat, vlon = self.geocode_address(vaddr, force_refresh=force_refresh)
            
            # If full address lookup failed, try a postcode/city fallback
            if (vlat is None or vlon is None) and any(col in row and pd.notna(row.get(col)) for col in ['Vendor Postcode', 'Vendor City', 'Vendor Country', 'Vendor Country Name']):
                fallback_cols = []
                if pd.notna(row.get('Vendor Postcode')):
                    fallback_cols.append('Vendor Postcode')
                if pd.notna(row.get('Vendor City')):
                    fallback_cols.append('Vendor City')
                if pd.notna(row.get('Vendor Country Name')):
                    fallback_cols.append('Vendor Country Name')
                elif pd.notna(row.get('Vendor Country')):
                    fallback_cols.append('Vendor Country')
                if fallback_cols:
                    vaddr_fb = self._build_address(row, fallback_cols)
                    vlat_fb, vlon_fb = self.geocode_address(vaddr_fb, force_refresh=force_refresh)
                    if vlat_fb is not None and vlon_fb is not None:
                        vlat, vlon = vlat_fb, vlon_fb
            
            vendor_lats[idx] = vlat
            vendor_lons[idx] = vlon
        
        for idx in missing_recip_indices:
            row = df.iloc[idx]
            raddr = recipient_addresses[idx]
            
            rlat, rlon = self.geocode_address(raddr, force_refresh=force_refresh)
            
            # Recipient fallback: try postcode/city when full address fails
            if (rlat is None or rlon is None) and any(col in row and pd.notna(row.get(col)) for col in ['Recipient Postcode', 'Recipient City', 'Recipient Country', 'Recipient Country Name']):
                fallback_cols = []
                if pd.notna(row.get('Recipient Postcode')):
                    fallback_cols.append('Recipient Postcode')
                if pd.notna(row.get('Recipient City')):
                    fallback_cols.append('Recipient City')
                if pd.notna(row.get('Recipient Country Name')):
                    fallback_cols.append('Recipient Country Name')
                elif pd.notna(row.get('Recipient Country')):
                    fallback_cols.append('Recipient Country')
                if fallback_cols:
                    raddr_fb = self._build_address(row, fallback_cols)
                    rlat_fb, rlon_fb = self.geocode_address(raddr_fb, force_refresh=force_refresh)
                    if rlat_fb is not None and rlon_fb is not None:
                        rlat, rlon = rlat_fb, rlon_fb
            
            recip_lats[idx] = rlat
            recip_lons[idx] = rlon

        df['vendor_longitude'] = vendor_lons
        df['vendor_latitude'] = vendor_lats
        df['recipient_longitude'] = recip_lons
        df['recipient_latitude'] = recip_lats

        return df

    def refresh_negative_cache(self, sleep_between: float = None, max_attempts: int = None):
        """Retry geocoding for addresses that are currently cached as (None, None).

        - sleep_between: seconds to wait between requests (defaults to min_delay_seconds)
        - max_attempts: maximum attempts per address (defaults to self.max_attempts)
        """
        if sleep_between is None:
            sleep_between = float(self.min_delay_seconds or 1.0)
        if max_attempts is None:
            max_attempts = int(self.max_attempts or 3)

        # collect negative entries
        negatives = [addr for addr, coords in list(self.cache.items()) if coords == (None, None)]
        for addr in negatives:
            # attempt re-geocode with force
            for attempt in range(max_attempts):
                lat, lon = self.geocode_address(addr, force_refresh=True)
                # if success, break
                if lat is not None and lon is not None:
                    break
                time.sleep(sleep_between)

        return None
