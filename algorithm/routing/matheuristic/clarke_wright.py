"""
Clarke-Wright Savings Algorithm 구현
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple

class ClarkeWrightSolver:
    """
    Clarke-Wright Savings Algorithm 솔버
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
        Clarke-Wright 알고리즘으로 경로 최적화
        """
        print("=== Clarke-Wright Savings Algorithm 실행 ===")
        
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
        
        # Savings 계산
        savings = self._calculate_savings(depot, depot_requests)
        
        # 경로 생성
        routes = self._build_routes(depot, depot_requests, savings)
        
        return routes
    
    def _get_depot_requests(self, depot):
        """
        depot에 할당된 배달 요청 반환
        """
        # 간단한 거리 기반 할당
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
    
    def _calculate_savings(self, depot, requests):
        """
        Savings 값 계산
        """
        savings = []
        
        for i in range(len(requests)):
            for j in range(i + 1, len(requests)):
                req1 = requests[i]
                req2 = requests[j]
                
                # 개별 경로 비용
                cost1 = self._calculate_distance(
                    depot['latitude'], depot['longitude'],
                    req1['pickup_lat'], req1['pickup_lon']
                ) + self._calculate_distance(
                    req1['pickup_lat'], req1['pickup_lon'],
                    req1['delivery_lat'], req1['delivery_lon']
                ) + self._calculate_distance(
                    req1['delivery_lat'], req1['delivery_lon'],
                    depot['latitude'], depot['longitude']
                )
                
                cost2 = self._calculate_distance(
                    depot['latitude'], depot['longitude'],
                    req2['pickup_lat'], req2['pickup_lon']
                ) + self._calculate_distance(
                    req2['pickup_lat'], req2['pickup_lon'],
                    req2['delivery_lat'], req2['delivery_lon']
                ) + self._calculate_distance(
                    req2['delivery_lat'], req2['delivery_lon'],
                    depot['latitude'], depot['longitude']
                )
                
                # 결합 경로 비용
                combined_cost = self._calculate_distance(
                    depot['latitude'], depot['longitude'],
                    req1['pickup_lat'], req1['pickup_lon']
                ) + self._calculate_distance(
                    req1['pickup_lat'], req1['pickup_lon'],
                    req1['delivery_lat'], req1['delivery_lon']
                ) + self._calculate_distance(
                    req1['delivery_lat'], req1['delivery_lon'],
                    req2['pickup_lat'], req2['pickup_lon']
                ) + self._calculate_distance(
                    req2['pickup_lat'], req2['pickup_lon'],
                    req2['delivery_lat'], req2['delivery_lon']
                ) + self._calculate_distance(
                    req2['delivery_lat'], req2['delivery_lon'],
                    depot['latitude'], depot['longitude']
                )
                
                # Savings 계산
                saving = cost1 + cost2 - combined_cost
                
                savings.append({
                    'request1': req1,
                    'request2': req2,
                    'saving': saving,
                    'combined_cost': combined_cost
                })
        
        # Savings 내림차순 정렬
        savings.sort(key=lambda x: x['saving'], reverse=True)
        
        return savings
    
    def _build_routes(self, depot, requests, savings):
        """
        Savings를 기반으로 경로 생성
        """
        routes = []
        used_requests = set()
        
        # 각 요청을 개별 경로로 시작
        for request in requests:
            route = {
                'depot_id': depot['id'],
                'requests': [request],
                'total_distance': self._calculate_route_distance(depot, [request]),
                'total_time': self._calculate_route_time(depot, [request])
            }
            routes.append(route)
        
        # Savings를 기반으로 경로 병합
        for saving in savings:
            req1 = saving['request1']
            req2 = saving['request2']
            
            # 이미 사용된 요청은 건너뛰기
            if req1 in used_requests or req2 in used_requests:
                continue
            
            # 경로 찾기
            route1_idx = self._find_route_with_request(routes, req1)
            route2_idx = self._find_route_with_request(routes, req2)
            
            if route1_idx is None or route2_idx is None:
                continue
            
            # 경로 병합 가능성 확인
            if self._can_merge_routes(routes[route1_idx], routes[route2_idx], depot):
                # 경로 병합
                merged_route = self._merge_routes(routes[route1_idx], routes[route2_idx], depot)
                
                # 기존 경로 제거
                routes.pop(max(route1_idx, route2_idx))
                routes.pop(min(route1_idx, route2_idx))
                
                # 병합된 경로 추가
                routes.append(merged_route)
                
                used_requests.add(req1)
                used_requests.add(req2)
        
        return routes
    
    def _find_route_with_request(self, routes, request):
        """
        특정 요청을 포함하는 경로 인덱스 찾기
        """
        for i, route in enumerate(routes):
            if request in route['requests']:
                return i
        return None
    
    def _can_merge_routes(self, route1, route2, depot):
        """
        두 경로 병합 가능성 확인
        """
        # 드론 제약 조건 확인
        total_requests = len(route1['requests']) + len(route2['requests'])
        if total_requests > 5:  # 최대 5개 요청
            return False
        
        # 시간 제약 확인
        merged_time = route1['total_time'] + route2['total_time']
        if merged_time > self.drone_config['battery_capacity'] * 60:  # 분을 초로 변환
            return False
        
        return True
    
    def _merge_routes(self, route1, route2, depot):
        """
        두 경로 병합
        """
        merged_requests = route1['requests'] + route2['requests']
        
        return {
            'depot_id': depot['id'],
            'requests': merged_requests,
            'total_distance': self._calculate_route_distance(depot, merged_requests),
            'total_time': self._calculate_route_time(depot, merged_requests)
        }
    
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