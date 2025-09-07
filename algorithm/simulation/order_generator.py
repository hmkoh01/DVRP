# algorithm/simulation/order_generator.py
# -*- coding: utf-8 -*-
import random
import itertools
from typing import Any, Dict, List, Optional
import pandas as pd  # ✅ 추가

class OrderGenerator:
    """
    실시간 주문 생성기
    - building_data(DataFrame 또는 List[Dict])를 받아 식당/고객 후보를 구성
    - 후보가 없으면 주변 무작위 좌표로 fallback
    """
    def __init__(
        self,
        building_data: Optional[Any] = None,  # DataFrame 또는 List[Dict]
        prob_per_tick: float = 0.25,
        rng_seed: int = 42,
        restaurant_key: str = "is_restaurant",
        customer_key: str = "is_customer",
        default_height_m: float = 30.0,
    ):
        self.rng = random.Random(rng_seed)
        self.prob_per_tick = prob_per_tick
        self.default_height_m = default_height_m

        # ✅ DataFrame도, 리스트도 안전하게 records로 통일
        records: List[Dict] = []
        if building_data is not None:
            if isinstance(building_data, pd.DataFrame):
                records = building_data.to_dict("records")
            elif isinstance(building_data, list):
                records = building_data
            else:
                records = []

        # 라벨이 있으면 필터, 없으면 랜덤 분할
        self.restaurants: List[Dict] = []
        self.customers: List[Dict] = []
        if len(records) > 0:
            labeled_rest = [b for b in records if b.get(restaurant_key) is True]
            labeled_cust = [b for b in records if b.get(customer_key) is True]
            if labeled_rest and labeled_cust:
                self.restaurants = labeled_rest
                self.customers = labeled_cust
            else:
                k = max(1, int(0.2 * len(records)))
                self.restaurants = self.rng.sample(records, k)
                # 샘플에 포함되지 않은 나머지를 고객으로
                rest_set = set(id(x) for x in self.restaurants)
                self.customers = [b for b in records if id(b) not in rest_set]

        self._req_counter = itertools.count(1)

    def _pick_loc(self, p: Dict) -> Dict:
        # 다양한 키 이름 방어적 처리
        lat = p.get("latitude") or p.get("lat") or p.get("Latitude")
        lon = p.get("longitude") or p.get("lon") or p.get("Longitude")
        return {
            "latitude": float(lat),
            "longitude": float(lon),
            "height": float(p.get("height", self.default_height_m)),
        }

    def maybe_generate(self, max_new: int = 2) -> List[Dict[str, Any]]:
        """
        매 tick 호출. 확률적으로 0~max_new개 주문 생성.
        """
        out: List[Dict[str, Any]] = []
        for _ in range(max_new):
            if self.rng.random() <= self.prob_per_tick:
                rid = next(self._req_counter)

                if self.restaurants and self.customers:
                    r = self._pick_loc(self.rng.choice(self.restaurants))
                    c = self._pick_loc(self.rng.choice(self.customers))
                else:
                    # 좌표 데이터가 없으면 캠퍼스 주변 임의 좌표로 fallback
                    base_lat, base_lon = 36.0139, 129.3261
                    r = {
                        "latitude": base_lat + self.rng.uniform(-0.003, 0.003),
                        "longitude": base_lon + self.rng.uniform(-0.003, 0.003),
                        "height": self.default_height_m,
                    }
                    c = {
                        "latitude": base_lat + self.rng.uniform(-0.003, 0.003),
                        "longitude": base_lon + self.rng.uniform(-0.003, 0.003),
                        "height": self.default_height_m,
                    }

                out.append({
                    "request_id": f"REQ-{rid:06d}",
                    "order_id":   f"ORD-{rid:06d}",
                    "restaurant_location": r,
                    "customer_location":   c,
                })
        return out
