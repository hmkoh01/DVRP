"""
시뮬레이티드 어닐링을 사용한 드론 배달 경로 최적화
"""

import numpy as np
import random
import copy
import math
from typing import List, Dict, Any
import warnings
warnings.filterwarnings('ignore')

class SimulatedAnnealing:
    """
    시뮬레이티드 어닐링 클래스
    """
    
    def __init__(self, optimization_target='cost', initial_temperature=1000, 
                 cooling_rate=0.95, min_temperature=1):
        """
        초기화
        """
        self.optimization_target = optimization_target
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.min_temperature = min_temperature
    
    def optimize(self, delivery_requests, depots):
        """
        경로 최적화 실행
        """
        print(f"시뮬레이티드 어닐링 최적화 시작 (목표: {self.optimization_target})")
        
        # 문제 데이터 준비
        self.delivery_requests = delivery_requests
        self.depots = depots
        
        # 초기 해 생성
        current_solution = self._create_initial_solution()
        current_cost = self._calculate_solution_cost(current_solution)
        
        # 최적 해 추적
        best_solution = copy.deepcopy(current_solution)
        best_cost = current_cost
        
        # 온도 초기화
        temperature = self.initial_temperature
        
        # 반복 실행
        iteration = 0
        while temperature > self.min_temperature:
            # 이웃 해 생성
            neighbor_solution = self._generate_neighbor(current_solution)
            neighbor_cost = self._calculate_solution_cost(neighbor_solution)
            
            # 비용 차이 계산
            cost_difference = neighbor_cost - current_cost
            
            # 수용 확률 계산
            if cost_difference < 0:
                # 더 나은 해인 경우 항상 수용
                acceptance_probability = 1.0
            else:
                # 더 나쁜 해인 경우 확률적으로 수용
                acceptance_probability = math.exp(-cost_difference / temperature)
            
            # 해 수용 여부 결정
            if random.random() < acceptance_probability:
                current_solution = neighbor_solution
                current_cost = neighbor_cost
                
                # 최적 해 업데이트
                if current_cost < best_cost:
                    best_solution = copy.deepcopy(current_solution)
                    best_cost = current_cost
            
            # 온도 감소
            temperature *= self.cooling_rate
            iteration += 1
            
            if iteration % 100 == 0:
                print(f"반복 {iteration}: 온도 = {temperature:.2f}, 최고 비용 = {best_cost:.2f}")
        
        # 최적 해를 경로 형태로 변환
        optimal_routes = self._convert_to_routes(best_solution)
        
        print(f"시뮬레이티드 어닐링 최적화 완료!")
        print(f"최종 최고 비용: {best_cost:.2f}")
        
        return optimal_routes
    
    def _create_initial_solution(self):
        """
        초기 해 생성
        """
        n_requests = len(self.delivery_requests)
        n_depots = len(self.depots)
        
        # 각 요청을 어떤 드론이 처리할지 결정
        drone_assignments = []
        for request_id in range(n_requests):
            drone_id = random.randint(0, n_depots - 1)
            drone_assignments.append(drone_id)
        
        # 각 드론별 경로 구성
        routes = []
        for drone_id in range(n_depots):
            # 해당 드론이 처리할 요청들
            assigned_requests = [i for i, d in enumerate(drone_assignments) if d == drone_id]
            
            if assigned_requests:
                # 각 요청에 대해 [식당, 고객] 순서로 방문
                route = []
                for req_id in assigned_requests:
                    restaurant_idx = n_depots + req_id * 2
                    customer_idx = restaurant_idx + 1
                    route.extend([restaurant_idx, customer_idx])
                
                # 경로 순서를 무작위로 섞기
                random.shuffle(route)
                routes.append(route)
            else:
                routes.append([])
        
        return {
            'drone_assignments': drone_assignments,
            'routes': routes
        }
    
    def _generate_neighbor(self, solution):
        """
        이웃 해 생성
        """
        neighbor = copy.deepcopy(solution)
        
        # 돌연변이 연산 선택
        mutation_type = random.choice(['swap_assignments', 'swap_route', 'reverse_route'])
        
        if mutation_type == 'swap_assignments':
            # 드론 할당 교환
            if len(neighbor['drone_assignments']) > 1:
                i, j = random.sample(range(len(neighbor['drone_assignments'])), 2)
                neighbor['drone_assignments'][i], neighbor['drone_assignments'][j] = \
                    neighbor['drone_assignments'][j], neighbor['drone_assignments'][i]
                
                # 경로 재구성
                neighbor['routes'] = self._reconstruct_routes(neighbor['drone_assignments'])
        
        elif mutation_type == 'swap_route':
            # 경로 내 두 지점 교환
            non_empty_routes = [i for i, route in enumerate(neighbor['routes']) if len(route) > 1]
            if non_empty_routes:
                route_idx = random.choice(non_empty_routes)
                route = neighbor['routes'][route_idx]
                
                if len(route) > 1:
                    i, j = random.sample(range(len(route)), 2)
                    route[i], route[j] = route[j], route[i]
        
        elif mutation_type == 'reverse_route':
            # 경로의 일부 역순
            non_empty_routes = [i for i, route in enumerate(neighbor['routes']) if len(route) > 2]
            if non_empty_routes:
                route_idx = random.choice(non_empty_routes)
                route = neighbor['routes'][route_idx]
                
                if len(route) > 2:
                    start, end = sorted(random.sample(range(len(route)), 2))
                    route[start:end+1] = reversed(route[start:end+1])
        
        return neighbor
    
    def _reconstruct_routes(self, drone_assignments):
        """
        드론 할당에 따라 경로 재구성
        """
        n_depots = len(self.depots)
        routes = [[] for _ in range(n_depots)]
        
        for request_id, drone_id in enumerate(drone_assignments):
            restaurant_idx = n_depots + request_id * 2
            customer_idx = restaurant_idx + 1
            routes[drone_id].extend([restaurant_idx, customer_idx])
        
        # 각 경로의 순서를 무작위로 섞기
        for route in routes:
            if route:
                random.shuffle(route)
        
        return routes
    
    def _calculate_solution_cost(self, solution):
        """
        해의 비용 계산
        """
        total_cost = 0
        total_time = 0
        
        for drone_id, route in enumerate(solution['routes']):
            if not route:
                continue
            
            # 드론의 경로 비용 계산
            drone_cost, drone_time = self._calculate_route_cost(drone_id, route)
            total_cost += drone_cost
            total_time = max(total_time, drone_time)
        
        # 목표에 따른 비용 반환
        if self.optimization_target == 'cost':
            return total_cost
        else:  # time
            return total_time
    
    def _calculate_route_cost(self, drone_id, route):
        """
        특정 드론의 경로 비용 계산
        """
        if not route:
            return 0, 0
        
        # 드론 설정 (간단한 가정)
        max_speed = 15.0  # m/s
        
        total_cost = 0
        total_time = 0
        
        # Depot에서 시작
        current_point = self.depots[drone_id]
        
        for point_idx in route:
            # 다음 지점 찾기
            if point_idx < len(self.depots):
                next_point = self.depots[point_idx]
            else:
                request_idx = (point_idx - len(self.depots)) // 2
                is_restaurant = (point_idx - len(self.depots)) % 2 == 0
                
                if is_restaurant:
                    next_point = self.delivery_requests[request_idx]['restaurant_location']
                else:
                    next_point = self.delivery_requests[request_idx]['customer_location']
            
            # 거리 계산
            dx = (next_point['longitude'] - current_point['longitude']) * 111000
            dy = (next_point['latitude'] - current_point['latitude']) * 111000
            dz = next_point['height'] - current_point['height']
            
            distance = np.sqrt(dx**2 + dy**2 + dz**2)
            
            # 시간 및 비용 계산
            time = distance / max_speed
            cost = distance * 0.5 + time / 3600 * 100  # 연료비 + 운영비
            
            total_cost += cost
            total_time += time
            
            current_point = next_point
        
        return total_cost, total_time
    
    def _convert_to_routes(self, solution):
        """
        최적 해를 경로 형태로 변환
        """
        routes = []
        
        for drone_id, route in enumerate(solution['routes']):
            if not route:
                continue
            
            # 경로 포인트 생성
            path_points = []
            
            # Depot에서 시작
            depot = self.depots[drone_id]
            path_points.append({
                'longitude': depot['longitude'],
                'latitude': depot['latitude'],
                'height': depot['height'],
                'type': 'depot'
            })
            
            # 경로 따라 포인트 추가
            for point_idx in route:
                if point_idx < len(self.depots):
                    # Depot
                    depot = self.depots[point_idx]
                    path_points.append({
                        'longitude': depot['longitude'],
                        'latitude': depot['latitude'],
                        'height': depot['height'],
                        'type': 'depot'
                    })
                else:
                    # 배달 요청
                    request_idx = (point_idx - len(self.depots)) // 2
                    is_restaurant = (point_idx - len(self.depots)) % 2 == 0
                    
                    if is_restaurant:
                        location = self.delivery_requests[request_idx]['restaurant_location']
                        point_type = 'restaurant'
                    else:
                        location = self.delivery_requests[request_idx]['customer_location']
                        point_type = 'customer'
                    
                    path_points.append({
                        'longitude': location['longitude'],
                        'latitude': location['latitude'],
                        'height': location['height'],
                        'type': point_type,
                        'request_id': request_idx
                    })
            
            # 경로 비용 계산
            cost, time, distance = self._calculate_route_cost(drone_id, route)
            
            routes.append({
                'drone_id': drone_id,
                'path': path_points,
                'total_distance': distance,
                'total_time': time,
                'total_cost': cost
            })
        
        return routes 