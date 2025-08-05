"""
개미 군집 최적화를 사용한 드론 배달 경로 최적화
"""

import numpy as np
import random
import copy
from typing import List, Dict, Any
import warnings
warnings.filterwarnings('ignore')

class AntColonyOptimization:
    """
    개미 군집 최적화 클래스
    """
    
    def __init__(self, optimization_target='cost', n_ants=30, n_iterations=50, 
                 evaporation_rate=0.1, alpha=1.0, beta=2.0):
        """
        초기화
        """
        self.optimization_target = optimization_target
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.evaporation_rate = evaporation_rate
        self.alpha = alpha  # 페로몬 중요도
        self.beta = beta    # 거리 중요도
        
        # 설정 파일에서 파라미터 로드
        from ..config import ALGORITHM_CONFIG
        if 'ant_colony' in ALGORITHM_CONFIG:
            config = ALGORITHM_CONFIG['ant_colony']
            self.n_ants = config.get('n_ants', n_ants)
            self.n_iterations = config.get('n_iterations', n_iterations)
            self.evaporation_rate = config.get('evaporation_rate', evaporation_rate)
            self.alpha = config.get('alpha', alpha)
            self.beta = config.get('beta', beta)
    
    def optimize(self, delivery_requests, depots):
        """
        경로 최적화 실행
        """
        print(f"개미 군집 최적화 시작 (목표: {self.optimization_target})")
        
        # 문제 데이터 준비
        self.delivery_requests = delivery_requests
        self.depots = depots
        
        # 거리 매트릭스 계산
        self.distance_matrix = self._calculate_distance_matrix()
        
        # 페로몬 매트릭스 초기화
        self.pheromone_matrix = np.ones_like(self.distance_matrix) * 0.1
        
        # 최적 해 추적
        best_solution = None
        best_cost = float('inf')
        
        # 반복 실행
        for iteration in range(self.n_iterations):
            # 모든 개미에 대해 경로 생성
            ant_solutions = []
            ant_costs = []
            
            for ant in range(self.n_ants):
                solution = self._construct_solution()
                cost = self._calculate_solution_cost(solution)
                
                ant_solutions.append(solution)
                ant_costs.append(cost)
                
                # 최적 해 업데이트
                if cost < best_cost:
                    best_cost = cost
                    best_solution = copy.deepcopy(solution)
            
            # 페로몬 업데이트
            self._update_pheromones(ant_solutions, ant_costs)
            
            if iteration % 10 == 0:
                print(f"반복 {iteration}: 최고 비용 = {best_cost:.2f}")
        
        # 최적 해를 경로 형태로 변환
        optimal_routes = self._convert_to_routes(best_solution)
        
        print(f"개미 군집 최적화 완료!")
        print(f"최종 최고 비용: {best_cost:.2f}")
        
        return optimal_routes
    
    def _calculate_distance_matrix(self):
        """
        거리 매트릭스 계산
        """
        n_requests = len(self.delivery_requests)
        n_depots = len(self.depots)
        n_points = n_requests * 2 + n_depots  # 식당 + 고객 + depot
        
        distance_matrix = np.zeros((n_points, n_points))
        
        # 모든 점들의 좌표 수집
        points = []
        
        # Depot 좌표
        for depot in self.depots:
            points.append({
                'type': 'depot',
                'longitude': depot['longitude'],
                'latitude': depot['latitude'],
                'height': depot['height']
            })
        
        # 배달 요청 좌표 (식당 + 고객)
        for request in self.delivery_requests:
            # 식당
            points.append({
                'type': 'restaurant',
                'longitude': request['restaurant_location']['longitude'],
                'latitude': request['restaurant_location']['latitude'],
                'height': request['restaurant_location']['height']
            })
            # 고객
            points.append({
                'type': 'customer',
                'longitude': request['customer_location']['longitude'],
                'latitude': request['customer_location']['latitude'],
                'height': request['customer_location']['height']
            })
        
        # 거리 계산
        for i in range(n_points):
            for j in range(n_points):
                if i != j:
                    point1 = points[i]
                    point2 = points[j]
                    
                    # 3D 거리 계산
                    dx = (point2['longitude'] - point1['longitude']) * 111000
                    dy = (point2['latitude'] - point1['latitude']) * 111000
                    dz = point2['height'] - point1['height']
                    
                    distance = np.sqrt(dx**2 + dy**2 + dz**2)
                    distance_matrix[i, j] = distance
        
        return distance_matrix
    
    def _construct_solution(self):
        """
        개미가 경로를 구성하는 과정
        """
        n_requests = len(self.delivery_requests)
        n_depots = len(self.depots)
        
        # 각 요청을 어떤 드론이 처리할지 결정
        drone_assignments = []
        for request_id in range(n_requests):
            # 개미가 확률적으로 드론 선택
            drone_id = random.randint(0, len(self.depots) - 1)
            drone_assignments.append(drone_id)
        
        # 각 드론별 경로 구성
        routes = []
        for drone_id in range(len(self.depots)):
            # 해당 드론이 처리할 요청들
            assigned_requests = [i for i, d in enumerate(drone_assignments) if d == drone_id]
            
            if not assigned_requests:
                routes.append([])
                continue
            
            # 경로 구성 (ACO 알고리즘)
            route = self._construct_ant_route(drone_id, assigned_requests)
            routes.append(route)
        
        return {
            'drone_assignments': drone_assignments,
            'routes': routes
        }
    
    def _construct_ant_route(self, drone_id, assigned_requests):
        """
        개미가 특정 드론의 경로를 구성
        """
        n_depots = len(self.depots)
        route = []
        unvisited = []
        
        # 방문할 지점들 생성 (식당 + 고객)
        for req_id in assigned_requests:
            restaurant_idx = n_depots + req_id * 2
            customer_idx = restaurant_idx + 1
            unvisited.extend([restaurant_idx, customer_idx])
        
        # 현재 위치 (depot)
        current = drone_id
        
        while unvisited:
            # 다음 방문할 지점 선택 (확률적)
            next_point = self._select_next_point(current, unvisited)
            
            if next_point is not None:
                route.append(next_point)
                unvisited.remove(next_point)
                current = next_point
            else:
                break
        
        return route
    
    def _select_next_point(self, current, unvisited):
        """
        확률적으로 다음 방문할 지점 선택
        """
        if not unvisited:
            return None
        
        # 각 미방문 지점의 확률 계산
        probabilities = []
        for point in unvisited:
            # 페로몬과 거리의 조합
            pheromone = self.pheromone_matrix[current, point]
            distance = self.distance_matrix[current, point]
            
            # 거리가 0인 경우 처리
            if distance == 0:
                distance = 0.1
            
            # 확률 계산
            prob = (pheromone ** self.alpha) * ((1.0 / distance) ** self.beta)
            probabilities.append(prob)
        
        # 확률 정규화
        total_prob = sum(probabilities)
        if total_prob == 0:
            # 모든 확률이 0인 경우 균등 분포
            probabilities = [1.0 / len(unvisited)] * len(unvisited)
        else:
            probabilities = [p / total_prob for p in probabilities]
        
        # 확률적 선택
        selected_idx = np.random.choice(len(unvisited), p=probabilities)
        return unvisited[selected_idx]
    
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
    
    def _update_pheromones(self, ant_solutions, ant_costs):
        """
        페로몬 매트릭스 업데이트
        """
        # 페로몬 증발
        self.pheromone_matrix *= (1 - self.evaporation_rate)
        
        # 각 개미의 경로에 페로몬 추가
        for solution, cost in zip(ant_solutions, ant_costs):
            # 비용이 낮을수록 더 많은 페로몬 추가
            pheromone_deposit = 1.0 / (1.0 + cost)
            
            # 경로의 각 엣지에 페로몬 추가
            for drone_id, route in enumerate(solution['routes']):
                if not route:
                    continue
                
                # Depot에서 시작
                current = drone_id
                
                for next_point in route:
                    self.pheromone_matrix[current, next_point] += pheromone_deposit
                    current = next_point
    
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