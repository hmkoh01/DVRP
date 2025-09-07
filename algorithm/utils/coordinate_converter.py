# -*- coding: utf-8 -*-
import math
import airsim
from typing import Dict, Tuple, Optional

class CoordinateConverter:
    def __init__(self, base_lat: float, base_lon: float):
        self.base_lat = base_lat
        self.base_lon = base_lon
        self.cos_lat = math.cos(math.radians(base_lat))

    def sanitize_llh(self, loc: Dict) -> Optional[Dict]:
        try:
            lat = float(loc.get("latitude"))
            lon = float(loc.get("longitude"))
            h   = float(loc.get("height", 0.0))
        except Exception:
            return None

        # 기본 범위 체크
        def in_range(a, lo, hi): return (a is not None) and (lo <= a <= hi)
        ok = in_range(lat, -90, 90) and in_range(lon, -180, 180)

        # 한국 데이터 흔한 스왑 케이스: lat≈120~140, lon≈30~45 → 스왑
        if not ok and in_range(lat, 120, 140) and in_range(lon, 25, 55):
            lat, lon = lon, lat
            ok = in_range(lat, -90, 90) and in_range(lon, -180, 180)

        return {"latitude": lat, "longitude": lon, "height": h} if ok else None

    def llh_to_ned(self, lat: float, lon: float, height_m: float) -> airsim.Vector3r:
        lat_diff = lat - self.base_lat
        lon_diff = lon - self.base_lon
        x = lat_diff * 111_320.0                   # North (x) ← latitude
        y = lon_diff * 111_320.0 * self.cos_lat    # East  (y) ← longitude
        z = -float(height_m)                       # Down (z) ← 상공은 음수
        return airsim.Vector3r(x, y, z)

    def loc_to_ned(self, loc: Dict) -> airsim.Vector3r:
        lat = float(loc["latitude"]); lon = float(loc["longitude"]); h = float(loc.get("height", 0.0))
        return self.llh_to_ned(lat, lon, h)

    def ned_to_llh(self, x: float, y: float, z: float) -> Tuple[float, float, float]:
        lat = self.base_lat + (x / 111_320.0)
        lon = self.base_lon + (y / (111_320.0 * self.cos_lat))
        h   = -z
        return lat, lon, h
