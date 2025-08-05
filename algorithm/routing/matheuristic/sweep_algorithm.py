"""
Sweep Algorithm 구현
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
import math

class SweepAlgorithm:
    """
    Sweep Algorithm 솔버
    """
    
    def __init__(self, depots, delivery_requests, drone_config):
        """
        초기화
        """
        self.depots = depots
        self.delivery_requests = delivery_requests
        self.drone_config = drone_config
        self.routes = []
        
    def solve(self):
        """
        Sweep Algorithm으로 경로 최적화
        """
        print("=== Sweep Algorithm 실행 ===")
        
        if not self.delivery_requests:
            print("배달 요청이 없습니다.")
            return []
        
        # 각 depot별로 별도 처리
        for depot in self.depots:
            depot_routes = self._solve_for_depot(depot)
            self.routes.extend(depot_routes)
        
        return self.routes
    
    def _solve_for_depot(self, depot):
        """
        특정 depot에 대한 경로 최적화
        """
        # 해당 depot의 배달 요청 필터링
        depot_requests = self._get_depot_requests(depot)
        
        if not depot_requests:
            return []
        
        # 각 요청의 각도 계산
        requests_with_angles = self._calculate_angles(depot, depot_requests)
        
        # 각도순 정렬
        requests_with_angles.sort(key=lambda x: x['angle'])
        
        # Sweep 알고리즘으로 경로 생성
        routes = self._build_sweep_routes(depot, requests_with_angles)
        
        return routes
    
    def _get_depot_requests(self, depot):
        """
        depot에 할당된 배달 요청 반환
        """
        depot_requests = []
        
        for request in self.delivery_requests:
            distance = self._calculate_distance(
                depot['latitude'], depot['longitude'],
                request['pickup_lat'], request['pickup_lon']
            )
            
            # 3km 이내의 요청을 해당 depot에 할당
            if distance <= 0.003:
                depot_requests.append(request)
        
        return depot_requests
    
    def _calculate_angles(self, depot, requests):
        """
        depot를 중심으로 각 요청의 각도 계산
        """
        requests_with_angles = []
        
        for request in requests:
            # depot에서 요청까지의 벡터
            dx = request['pickup_lon'] - depot['longitude']
            dy = request['pickup_lat'] - depot['latitude']
            
            # 각도 계산 (라디안)
            angle = math.atan2(dy, dx)
            
            # 각도를 0-360도로 변환
            angle_degrees = math.degrees(angle)
            if angle_degrees < 0:
                angle_degrees += 360
            
            requests_with_angles.append({
                'request': request,
                'angle': angle_degrees,
                'distance': self._calculate_distance(
                    depot['latitude'], depot['longitude'],
                    request['pickup_lat'], request['pickup_lon']
                )
            })
        
        return requests_with_angles
    
    def _build_sweep_routes(self, depot, requests_with_angles):
        """
        Sweep 알고리즘으로 경로 생성
        """
        routes = []
        current_route = []
        current_load = 0
        current_time = 0
        
        for req_data in requests_with_angles:
            request = req_data['request']
            
            # 현재 요청의 시간과 무게 계산
            request_time = self._calculate_request_time(depot, request)
            request_weight = request.get('weight', 1.0)
            
            # 제약 조건 확인
            if (current_load + request_weight <= self.drone_config['max_payload'] and
                current_time + request_time <= self.drone_config['battery_capacity'] * 60):
                
                # 현재 경로에 추가
                current_route.append(request)
                current_load += request_weight
                current_time += request_time
            else:
                # 현재 경로 완성
                if current_route:
                    route = {
                        'depot_id': depot['id'],
                        'requests': current_route.copy(),
                        'total_distance': self._calculate_route_distance(depot, current_route),
                        'total_time': self._calculate_route_time(depot, current_route),
                        'total_weight': current_load
                    }
                    routes.append(route)
                
                # 새 경로 시작
                current_route = [request]
                current_load = request_weight
                current_time = request_time
        
        # 마지막 경로 추가
        if current_route:
            route = {
                'depot_id': depot['id'],
                'requests': current_route,
                'total_distance': self._calculate_route_distance(depot, current_route),
                'total_time': self._calculate_route_time(depot, current_route),
                'total_weight': current_load
            }
            routes.append(route)
        
        return routes
    
    def _calculate_request_time(self, depot, request):
        """
        단일 요청의 처리 시간 계산
        """
        # depot에서 픽업 지점까지
        pickup_distance = self._calculate_distance(
            depot['latitude'], depot['longitude'],
            request['pickup_lat'], request['pickup_lon']
        )
        
        # 픽업에서 배달까지
        delivery_distance = self._calculate_distance(
            request['pickup_lat'], request['pickup_lon'],
            request['delivery_lat'], request['delivery_lon']
        )
        
        # 배달에서 depot까지
        return_distance = self._calculate_distance(
            request['delivery_lat'], request['delivery_lon'],
            depot['latitude'], depot['longitude']
        )
        
        total_distance = pickup_distance + delivery_distance + return_distance
        
        # 비행 시간 (속도 15 m/s)
        flight_time = total_distance * 1000 / 15  # km를 m로 변환
        
        # 픽업/배달 시간 (2분)
        service_time = 2 * 60
        
        return flight_time + service_time
    
    def _calculate_distance(self, lat1, lon1, lat2, lon2):
        """
        두 지점 간 거리 계산 (km)
        """
        return np.sqrt((lat1 - lat2)**2 + (lon1 - lon2)**2) * 111
    
    def _calculate_route_distance(self, depot, requests):
        """
        경로의 총 거리 계산
        """
        if not requests:
            return 0
        
        total_distance = 0
        current_lat, current_lon = depot['latitude'], depot['longitude']
        
        for request in requests:
            # 픽업 지점까지
            total_distance += self._calculate_distance(
                current_lat, current_lon,
                request['pickup_lat'], request['pickup_lon']
            )
            
            # 배달 지점까지
            total_distance += self._calculate_distance(
                request['pickup_lat'], request['pickup_lon'],
                request['delivery_lat'], request['delivery_lon']
            )
            
            current_lat, current_lon = request['delivery_lat'], request['delivery_lon']
        
        # depot로 복귀
        total_distance += self._calculate_distance(
            current_lat, current_lon,
            depot['latitude'], depot['longitude']
        )
        
        return total_distance
    
    def _calculate_route_time(self, depot, requests):
        """
        경로의 총 시간 계산 (초)
        """
        distance = self._calculate_route_distance(depot, requests)
        
        # 비행 시간 (속도 15 m/s)
        flight_time = distance * 1000 / 15  # km를 m로 변환
        
        # 픽업/배달 시간 (요청당 2분)
        service_time = len(requests) * 2 * 60
        
        return flight_time + service_time 