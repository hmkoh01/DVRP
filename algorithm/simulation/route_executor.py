# -*- coding: utf-8 -*-
from typing import Dict, List, Tuple, Any
import airsim

class RouteExecutor:
    """
    RouteManager 의 활성 경로를 AirSim 큐로 흘려보내는 어댑터
    - 드론 논리 ID(예: 'drone_depot_1_1') -> AirSim vehicle name('Drone_1') 매핑
    - 경로 변경 감지 후, NED 웨이포인트로 변환하여 enqueue
    """
    def __init__(self, route_manager, converter, enqueue_fn, is_vehicle_idle_fn, logical_to_vehicle: Dict[str, str]):
        self.rm = route_manager
        self.cv = converter
        self.enqueue = enqueue_fn               # (vehicle_name:str, waypoints:List[airsim.Vector3r]) -> None
        self.is_vehicle_idle = is_vehicle_idle_fn
        self.map_lv = logical_to_vehicle

        # 마지막으로 푸시한 경로 시그니처(중복 enqueue 방지)
        self._last_sig: Dict[str, Tuple] = {}

    @staticmethod
    def _route_signature(route: List[Dict]) -> Tuple:
        # (type, order_id, round(lat,6), round(lon,6), round(h,1)) 의 튜플열
        sig = []
        for p in route:
            loc = p["location"]
            sig.append((
                p.get("type"),
                p.get("order_id"),
                round(float(loc["latitude"]), 6),
                round(float(loc["longitude"]), 6),
                round(float(loc.get("height", 0.0)), 1),
            ))
        return tuple(sig)

    def sync_active_routes(self):
        """
        RouteManager 의 active_routes 를 살펴보고,
        변화가 있는 드론의 경로를 AirSim 큐에 등록
        """
        active = self.rm.get_active_routes()  # {drone_id: [points...]}
        for logical_id, route in active.items():
            vehicle = self.map_lv.get(logical_id)
            if not vehicle:
                continue

            sig = self._route_signature(route)
            if self._last_sig.get(logical_id) == sig:
                continue  # 변화 없음

            # 드론이 idle이고, 큐가 비어 있어야 안전하게 전체 경로 enqueue 가능
            if not self.is_vehicle_idle(vehicle):
                # 바쁘면 다음 tick 에 재시도
                continue

            # 웨이포인트 변환
            waypoints: List[airsim.Vector3r] = [self.cv.loc_to_ned(p["location"]) for p in route]
            if waypoints:
                self.enqueue(vehicle, waypoints)
                self._last_sig[logical_id] = sig
