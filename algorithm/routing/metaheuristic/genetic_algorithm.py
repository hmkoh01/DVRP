"""
유전 알고리즘을 사용한 드론 배달 경로 최적화
"""

import numpy as np
import random
import copy
from typing import List, Dict, Any
import warnings
warnings.filterwarnings('ignore')

class GeneticAlgorithm:
    """
    유전 알고리즘 클래스
    """
    
    def __init__(self, optimization_target='cost', population_size=50, generations=100, 
                 mutation_rate=0.1, crossover_rate=0.8, elite_size=5):
        """
        초기화
        """
        self.optimization_target = optimization_target
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
        
        # 설정 파일에서 파라미터 로드
        from ..config import ALGORITHM_CONFIG
        if 'genetic' in ALGORITHM_CONFIG:
            config = ALGORITHM_CONFIG['genetic']
            self.population_size = config.get('population_size', population_size)
            self.generations = config.get('generations', generations)
            self.mutation_rate = config.get('mutation_rate', mutation_rate)
            self.crossover_rate = config.get('crossover_rate', crossover_rate)
            self.elite_size = config.get('elite_size', elite_size)
    
    def optimize(self, delivery_requests, depots, drones):
        """
        경로 최적화 실행
        """
        print(f"유전 알고리즘 최적화 시작 (목표: {self.optimization_target})")
        
        # 문제 데이터 준비
        self.delivery_requests = delivery_requests
        self.depots = depots
        self.drones = drones
        
        # 거리 매트릭스 계산
        self.distance_matrix = self._calculate_distance_matrix()
        
        # 초기 개체군 생성
        population = self._create_initial_population()
        
        # 진화 과정
        best_fitness_history = []
        avg_fitness_history = []
        
        for generation in range(self.generations):
            # 적합도 계산
            fitness_scores = [self._calculate_fitness(individual) for individual in population]
            
            # 통계 기록
            best_fitness = max(fitness_scores)
            avg_fitness = np.mean(fitness_scores)
            best_fitness_history.append(best_fitness)
            avg_fitness_history.append(avg_fitness)
            
            if generation % 10 == 0:
                print(f"세대 {generation}: 최고 적합도 = {best_fitness:.2f}, 평균 적합도 = {avg_fitness:.2f}")
            
            # 새로운 개체군 생성
            new_population = []
            
            # 엘리트 보존
            elite_indices = np.argsort(fitness_scores)[-self.elite_size:]
            for idx in elite_indices:
                new_population.append(copy.deepcopy(population[idx]))
            
            # 교차 및 돌연변이로 나머지 개체 생성
            while len(new_population) < self.population_size:
                # 선택
                parent1 = self._selection(population, fitness_scores)
                parent2 = self._selection(population, fitness_scores)
                
                # 교차
                if random.random() < self.crossover_rate:
                    child1, child2 = self._crossover(parent1, parent2)
                else:
                    child1, child2 = copy.deepcopy(parent1), copy.deepcopy(parent2)
                
                # 돌연변이
                if random.random() < self.mutation_rate:
                    child1 = self._mutation(child1)
                if random.random() < self.mutation_rate:
                    child2 = self._mutation(child2)
                
                new_population.extend([child1, child2])
            
            # 개체군 크기 조정
            population = new_population[:self.population_size]
        
        # 최적 해 찾기
        final_fitness_scores = [self._calculate_fitness(individual) for individual in population]
        best_individual_idx = np.argmax(final_fitness_scores)
        best_individual = population[best_individual_idx]
        
        # 최적 해를 경로 형태로 변환
        optimal_routes = self._convert_to_routes(best_individual)
        
        print(f"유전 알고리즘 최적화 완료!")
        print(f"최종 최고 적합도: {final_fitness_scores[best_individual_idx]:.2f}")
        
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
                'height': 0  # depot는 지면에 위치
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
                    dz = point2.get('height', 0) - point1.get('height', 0)
                    
                    distance = np.sqrt(dx**2 + dy**2 + dz**2)
                    distance_matrix[i, j] = distance
        
        return distance_matrix
    
    def _create_initial_population(self):
        """
        초기 개체군 생성
        """
        population = []
        
        for _ in range(self.population_size):
            individual = self._create_random_individual()
            population.append(individual)
        
        return population
    
    def _create_random_individual(self):
        """
        무작위 개체 생성
        """
        individual = {
            'drone_assignments': [],  # 각 요청을 어떤 드론이 처리할지
            'route_sequences': []     # 각 드론의 경로 순서
        }
        
        n_requests = len(self.delivery_requests)
        n_drones = len(self.drones)
        
        # 드론 할당 (무작위)
        for request_id in range(n_requests):
            drone_id = random.randint(0, n_drones - 1)
            individual['drone_assignments'].append(drone_id)
        
        # 경로 순서 (무작위 순열)
        for drone_id in range(n_drones):
            # 해당 드론이 처리할 요청들 찾기
            assigned_requests = [i for i, d in enumerate(individual['drone_assignments']) if d == drone_id]
            
            if assigned_requests:
                # 각 요청에 대해 [식당, 고객] 순서로 방문
                route = []
                for req_id in assigned_requests:
                    restaurant_idx = len(self.depots) + req_id * 2  # depot 다음부터 식당들
                    customer_idx = restaurant_idx + 1
                    route.extend([restaurant_idx, customer_idx])
                
                # 경로 순서를 무작위로 섞기
                random.shuffle(route)
                individual['route_sequences'].append(route)
            else:
                individual['route_sequences'].append([])
        
        return individual
    
    def _calculate_fitness(self, individual):
        """
        개체의 적합도 계산
        """
        total_cost = 0
        total_time = 0
        total_distance = 0
        
        n_drones = len(self.drones)
        
        for drone_id in range(n_drones):
            route = individual['route_sequences'][drone_id]
            
            if not route:
                continue
            
            # 드론의 경로 비용 계산
            drone_cost, drone_time, drone_distance = self._calculate_route_cost(
                drone_id, route
            )
            
            total_cost += drone_cost
            total_time = max(total_time, drone_time)  # 병렬 처리 가정
            total_distance += drone_distance
        
        # 적합도 계산 (목표에 따라)
        if self.optimization_target == 'cost':
            fitness = 1.0 / (1.0 + total_cost)  # 비용이 낮을수록 높은 적합도
        else:  # time
            fitness = 1.0 / (1.0 + total_time)  # 시간이 짧을수록 높은 적합도
        
        return fitness
    
    def _calculate_route_cost(self, drone_id, route):
        """
        특정 드론의 경로 비용 계산
        """
        if not route:
            return 0, 0, 0
        
        drone = self.drones[drone_id]
        
        total_cost = 0
        total_time = 0
        total_distance = 0
        
        # Depot에서 시작
        current_point = {
            'longitude': drone['current_lon'],
            'latitude': drone['current_lat'],
            'height': 0  # 드론은 기본적으로 지면에서 시작
        }
        
        for point_idx in route:
            # 다음 지점까지의 거리 계산
            if point_idx < len(self.depots):
                # Depot
                next_point = self.depots[point_idx]
            else:
                # 배달 요청 (식당 또는 고객)
                request_idx = (point_idx - len(self.depots)) // 2
                is_restaurant = (point_idx - len(self.depots)) % 2 == 0
                
                # 범위 체크
                if request_idx >= len(self.delivery_requests):
                    print(f"Warning: request_idx {request_idx} out of range, max: {len(self.delivery_requests)-1}")
                    continue
                
                if is_restaurant:
                    next_point = self.delivery_requests[request_idx]['restaurant_location']
                else:
                    next_point = self.delivery_requests[request_idx]['customer_location']
            
            # 거리 계산
            dx = (next_point['longitude'] - current_point['longitude']) * 111000
            dy = (next_point['latitude'] - current_point['latitude']) * 111000
            dz = next_point.get('height', 0) - current_point.get('height', 0)
            
            distance = np.sqrt(dx**2 + dy**2 + dz**2)
            total_distance += distance
            
            # 시간 계산 (속도 기반)
            time = distance / drone['max_speed']  # 초 단위
            total_time += time
            
            # 비용 계산
            fuel_cost = distance * 0.5  # km당 0.5원
            operation_cost = time / 3600 * 100  # 시간당 100원
            total_cost += fuel_cost + operation_cost
            
            # 현재 위치 업데이트
            current_point = next_point
        
        return total_cost, total_time, total_distance
    
    def _selection(self, population, fitness_scores):
        """
        토너먼트 선택
        """
        tournament_size = 3
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        
        winner_idx = tournament_indices[np.argmax(tournament_fitness)]
        return copy.deepcopy(population[winner_idx])
    
    def _crossover(self, parent1, parent2):
        """
        Order Crossover (OX)
        """
        child1 = copy.deepcopy(parent1)
        child2 = copy.deepcopy(parent2)
        
        # 드론 할당 교차
        n_requests = len(self.delivery_requests)
        if n_requests <= 1:
            return child1, child2
        crossover_point = random.randint(1, n_requests - 1)
        
        child1['drone_assignments'] = (
            parent1['drone_assignments'][:crossover_point] +
            parent2['drone_assignments'][crossover_point:]
        )
        child2['drone_assignments'] = (
            parent2['drone_assignments'][:crossover_point] +
            parent1['drone_assignments'][crossover_point:]
        )
        
        # 경로 순서 교차 (각 드론별로)
        n_drones = len(self.drones)
        for drone_id in range(n_drones):
            route1 = parent1['route_sequences'][drone_id]
            route2 = parent2['route_sequences'][drone_id]
            
            if len(route1) > 1 and len(route2) > 1:
                # Order Crossover 적용
                new_route1, new_route2 = self._order_crossover(route1, route2)
                child1['route_sequences'][drone_id] = new_route1
                child2['route_sequences'][drone_id] = new_route2
        
        return child1, child2
    
    def _order_crossover(self, route1, route2):
        """
        Order Crossover for route sequences
        """
        if len(route1) < 2 or len(route2) < 2:
            return route1, route2
        
        # 교차 구간 선택
        if len(route1) < 2:
            return route1, route2
        start = random.randint(0, len(route1) - 2)
        end = random.randint(start + 1, len(route1))
        
        # 자식 1 생성
        child1 = [-1] * len(route1)
        child1[start:end] = route1[start:end]
        
        # 나머지 위치를 route2에서 채우기
        remaining = [x for x in route2 if x not in child1[start:end]]
        j = 0
        for i in range(len(child1)):
            if child1[i] == -1:
                child1[i] = remaining[j]
                j += 1
        
        # 자식 2 생성 (역순)
        child2 = [-1] * len(route2)
        child2[start:end] = route2[start:end]
        
        remaining = [x for x in route1 if x not in child2[start:end]]
        j = 0
        for i in range(len(child2)):
            if child2[i] == -1:
                child2[i] = remaining[j]
                j += 1
        
        return child1, child2
    
    def _mutation(self, individual):
        """
        돌연변이 연산
        """
        mutated = copy.deepcopy(individual)
        
        # 드론 할당 돌연변이
        for i in range(len(mutated['drone_assignments'])):
            if random.random() < 0.1:  # 10% 확률로 돌연변이
                mutated['drone_assignments'][i] = random.randint(0, len(self.drones) - 1)
        
        # 경로 순서 돌연변이 (Swap Mutation)
        for drone_id in range(len(mutated['route_sequences'])):
            route = mutated['route_sequences'][drone_id]
            if len(route) > 1 and random.random() < 0.1:
                # 두 위치 교환
                i, j = random.sample(range(len(route)), 2)
                route[i], route[j] = route[j], route[i]
        
        return mutated
    
    def _convert_to_routes(self, individual):
        """
        최적 개체를 경로 형태로 변환
        """
        routes = []
        
        for drone_id in range(len(self.drones)):
            route = individual['route_sequences'][drone_id]
            
            if not route:
                continue
            
            # 경로 포인트 생성
            path_points = []
            
            # Depot에서 시작
            drone = self.drones[drone_id]
            path_points.append({
                'longitude': drone['current_lon'],
                'latitude': drone['current_lat'],
                'height': 0,  # 드론은 기본적으로 지면에서 시작
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
                        'height': depot.get('height', 0),  # height가 없으면 기본값 0
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
                        'height': location.get('height', 0),  # height가 없으면 기본값 0
                        'type': point_type,
                        'request_id': request_idx
                    })
            
            routes.append({
                'drone_id': drone_id,
                'path': path_points,
                'total_distance': self._calculate_route_cost(drone_id, route)[2],
                'total_time': self._calculate_route_cost(drone_id, route)[1],
                'total_cost': self._calculate_route_cost(drone_id, route)[0]
            })
        
        return routes
    
    def solve(self):
        """
        경로 최적화 실행 (메인 인터페이스)
        """
        try:
            # 최적화 실행
            optimal_routes = self.optimize(self.delivery_requests, self.depots, self.drones)
            return optimal_routes
        except Exception as e:
            print(f"유전 알고리즘 오류, 기본 경로 생성: {e}")
            # 기본 경로 생성 (오류 발생 시)
            return self._create_basic_routes()
    
    def _create_basic_routes(self):
        """
        기본 경로 생성 (오류 발생 시 사용)
        """
        routes = []
        
        # 각 드론에 대해 기본 경로 생성
        for drone_id, drone in enumerate(self.drones):
            if drone_id < len(self.delivery_requests):
                # 드론이 처리할 요청
                request = self.delivery_requests[drone_id]
                
                route = {
                    'drone_id': drone_id,
                    'path': [
                        {
                            'longitude': drone['current_lon'],
                            'latitude': drone['current_lat'],
                            'height': 0,
                            'type': 'depot'
                        },
                        {
                            'longitude': request['restaurant_location']['longitude'],
                            'latitude': request['restaurant_location']['latitude'],
                            'height': request['restaurant_location'].get('height', 0),
                            'type': 'restaurant',
                            'request_id': drone_id
                        },
                        {
                            'longitude': request['customer_location']['longitude'],
                            'latitude': request['customer_location']['latitude'],
                            'height': request['customer_location'].get('height', 0),
                            'type': 'customer',
                            'request_id': drone_id
                        }
                    ],
                    'total_distance': 1000,  # 기본값
                    'total_time': 600,       # 기본값
                    'total_cost': 500        # 기본값
                }
                routes.append(route)
        
        return routes 