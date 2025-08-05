"""
입자 군집 최적화를 사용한 드론 배달 경로 최적화
"""

import numpy as np
import random
import copy
from typing import List, Dict, Any
import warnings
warnings.filterwarnings('ignore')

class ParticleSwarmOptimization:
    """
    입자 군집 최적화 클래스
    """
    
    def __init__(self, optimization_target='cost', n_particles=30, n_iterations=50, 
                 w=0.7, c1=2.0, c2=2.0):
        """
        초기화
        """
        self.optimization_target = optimization_target
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.w = w  # 관성 가중치
        self.c1 = c1  # 개인 학습률
        self.c2 = c2  # 사회 학습률
        
        # 설정 파일에서 파라미터 로드
        from ..config import ALGORITHM_CONFIG
        if 'particle_swarm' in ALGORITHM_CONFIG:
            config = ALGORITHM_CONFIG['particle_swarm']
            self.n_particles = config.get('n_particles', n_particles)
            self.n_iterations = config.get('n_iterations', n_iterations)
            self.w = config.get('w', w)
            self.c1 = config.get('c1', c1)
            self.c2 = config.get('c2', c2)
    
    def optimize(self, delivery_requests, depots):
        """
        경로 최적화 실행
        """
        print(f"입자 군집 최적화 시작 (목표: {self.optimization_target})")
        
        # 문제 데이터 준비
        self.delivery_requests = delivery_requests
        self.depots = depots
        
        # 입자 초기화
        particles = self._initialize_particles()
        
        # 개인 최적 위치와 전역 최적 위치 초기화
        personal_best_positions = copy.deepcopy(particles)
        personal_best_fitness = [self._calculate_fitness(p) for p in particles]
        
        global_best_idx = np.argmax(personal_best_fitness)
        global_best_position = copy.deepcopy(particles[global_best_idx])
        global_best_fitness = personal_best_fitness[global_best_idx]
        
        # 속도 초기화
        velocities = self._initialize_velocities()
        
        # 반복 실행
        for iteration in range(self.n_iterations):
            # 각 입자 업데이트
            for i in range(self.n_particles):
                # 속도 업데이트
                velocities[i] = self._update_velocity(
                    velocities[i], particles[i], 
                    personal_best_positions[i], global_best_position
                )
                
                # 위치 업데이트
                particles[i] = self._update_position(particles[i], velocities[i])
                
                # 적합도 계산
                fitness = self._calculate_fitness(particles[i])
                
                # 개인 최적 업데이트
                if fitness > personal_best_fitness[i]:
                    personal_best_fitness[i] = fitness
                    personal_best_positions[i] = copy.deepcopy(particles[i])
                    
                    # 전역 최적 업데이트
                    if fitness > global_best_fitness:
                        global_best_fitness = fitness
                        global_best_position = copy.deepcopy(particles[i])
            
            if iteration % 10 == 0:
                print(f"반복 {iteration}: 최고 적합도 = {global_best_fitness:.2f}")
        
        # 최적 해를 경로 형태로 변환
        optimal_routes = self._convert_to_routes(global_best_position)
        
        print(f"입자 군집 최적화 완료!")
        print(f"최종 최고 적합도: {global_best_fitness:.2f}")
        
        return optimal_routes
    
    def _initialize_particles(self):
        """
        입자 초기화
        """
        particles = []
        
        for _ in range(self.n_particles):
            particle = self._create_random_solution()
            particles.append(particle)
        
        return particles
    
    def _create_random_solution(self):
        """
        무작위 해 생성
        """
        n_requests = len(self.delivery_requests)
        n_depots = len(self.depots)
        
        # 각 요청을 어떤 드론이 처리할지 결정 (연속값으로 표현)
        drone_assignments = np.random.random(n_requests) * n_depots
        
        # 각 드론별 경로 순서 (연속값으로 표현)
        route_orders = []
        for drone_id in range(n_depots):
            assigned_requests = [i for i, d in enumerate(drone_assignments) if int(d) == drone_id]
            
            if assigned_requests:
                # 각 요청에 대한 방문 순서 (연속값)
                order_values = np.random.random(len(assigned_requests) * 2)  # 식당 + 고객
                route_orders.append(order_values)
            else:
                route_orders.append(np.array([]))
        
        return {
            'drone_assignments': drone_assignments,
            'route_orders': route_orders
        }
    
    def _initialize_velocities(self):
        """
        속도 초기화
        """
        velocities = []
        
        for _ in range(self.n_particles):
            n_requests = len(self.delivery_requests)
            n_depots = len(self.depots)
            
            # 드론 할당 속도
            drone_velocity = np.random.randn(n_requests) * 0.1
            
            # 경로 순서 속도
            route_velocities = []
            for drone_id in range(n_depots):
                assigned_requests = [i for i, d in enumerate(self.delivery_requests) 
                                   if int(d) == drone_id]
                if assigned_requests:
                    route_velocity = np.random.randn(len(assigned_requests) * 2) * 0.1
                    route_velocities.append(route_velocity)
                else:
                    route_velocities.append(np.array([]))
            
            velocities.append({
                'drone_assignments': drone_velocity,
                'route_orders': route_velocities
            })
        
        return velocities
    
    def _calculate_fitness(self, particle):
        """
        입자의 적합도 계산
        """
        # 연속값을 이산 해로 변환
        solution = self._discretize_solution(particle)
        
        # 비용 계산
        total_cost = 0
        total_time = 0
        
        for drone_id, route in enumerate(solution['routes']):
            if not route:
                continue
            
            # 드론의 경로 비용 계산
            drone_cost, drone_time = self._calculate_route_cost(drone_id, route)
            total_cost += drone_cost
            total_time = max(total_time, drone_time)
        
        # 적합도 계산 (목표에 따라)
        if self.optimization_target == 'cost':
            fitness = 1.0 / (1.0 + total_cost)  # 비용이 낮을수록 높은 적합도
        else:  # time
            fitness = 1.0 / (1.0 + total_time)  # 시간이 짧을수록 높은 적합도
        
        return fitness
    
    def _discretize_solution(self, particle):
        """
        연속값을 이산 해로 변환
        """
        n_requests = len(self.delivery_requests)
        n_depots = len(self.depots)
        
        # 드론 할당을 이산값으로 변환
        drone_assignments = []
        for assignment in particle['drone_assignments']:
            drone_id = int(assignment) % n_depots
            drone_assignments.append(drone_id)
        
        # 경로 순서를 이산값으로 변환
        routes = []
        for drone_id in range(n_depots):
            assigned_requests = [i for i, d in enumerate(drone_assignments) if d == drone_id]
            
            if assigned_requests and len(particle['route_orders']) > drone_id:
                order_values = particle['route_orders'][drone_id]
                
                # 순서값을 인덱스로 변환
                route = []
                for req_id in assigned_requests:
                    restaurant_idx = n_depots + req_id * 2
                    customer_idx = restaurant_idx + 1
                    
                    # 식당과 고객의 순서 결정
                    req_order_idx = assigned_requests.index(req_id) * 2
                    if req_order_idx < len(order_values):
                        if order_values[req_order_idx] < order_values[req_order_idx + 1]:
                            route.extend([restaurant_idx, customer_idx])
                        else:
                            route.extend([customer_idx, restaurant_idx])
                
                routes.append(route)
            else:
                routes.append([])
        
        return {
            'drone_assignments': drone_assignments,
            'routes': routes
        }
    
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
    
    def _update_velocity(self, velocity, position, personal_best, global_best):
        """
        속도 업데이트
        """
        new_velocity = copy.deepcopy(velocity)
        
        # 드론 할당 속도 업데이트
        r1, r2 = np.random.random(2)
        
        new_velocity['drone_assignments'] = (
            self.w * velocity['drone_assignments'] +
            self.c1 * r1 * (personal_best['drone_assignments'] - position['drone_assignments']) +
            self.c2 * r2 * (global_best['drone_assignments'] - position['drone_assignments'])
        )
        
        # 경로 순서 속도 업데이트
        for i in range(len(velocity['route_orders'])):
            if (len(velocity['route_orders'][i]) > 0 and 
                len(personal_best['route_orders'][i]) > 0 and 
                len(global_best['route_orders'][i]) > 0):
                
                r1, r2 = np.random.random(2)
                
                new_velocity['route_orders'][i] = (
                    self.w * velocity['route_orders'][i] +
                    self.c1 * r1 * (personal_best['route_orders'][i] - position['route_orders'][i]) +
                    self.c2 * r2 * (global_best['route_orders'][i] - position['route_orders'][i])
                )
        
        return new_velocity
    
    def _update_position(self, position, velocity):
        """
        위치 업데이트
        """
        new_position = copy.deepcopy(position)
        
        # 드론 할당 위치 업데이트
        new_position['drone_assignments'] += velocity['drone_assignments']
        
        # 경로 순서 위치 업데이트
        for i in range(len(position['route_orders'])):
            if len(position['route_orders'][i]) > 0:
                new_position['route_orders'][i] += velocity['route_orders'][i]
        
        return new_position
    
    def _convert_to_routes(self, particle):
        """
        최적 입자를 경로 형태로 변환
        """
        solution = self._discretize_solution(particle)
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