"""
Deep Q-Network를 사용한 드론 배달 경로 최적화
"""

import numpy as np
import random
import copy
from typing import List, Dict, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

class DeepQNetwork:
    """
    Deep Q-Network 클래스 (간단한 구현)
    """
    
    def __init__(self, optimization_target='cost', learning_rate=0.001, 
                 discount_factor=0.95, epsilon=0.1, epsilon_decay=0.995, 
                 episodes=1000, batch_size=32, memory_size=10000):
        """
        초기화
        """
        self.optimization_target = optimization_target
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.episodes = episodes
        self.batch_size = batch_size
        self.memory_size = memory_size
        
        # 경험 리플레이 메모리
        self.memory = []
        
        # 신경망 (간단한 구현)
        self.q_network = {}
        
        # 설정 파일에서 파라미터 로드
        from ..config import ALGORITHM_CONFIG
        if 'dqn' in ALGORITHM_CONFIG:
            config = ALGORITHM_CONFIG['dqn']
            self.learning_rate = config.get('learning_rate', learning_rate)
            self.discount_factor = config.get('discount_factor', discount_factor)
            self.epsilon = config.get('epsilon', epsilon)
            self.epsilon_decay = config.get('epsilon_decay', epsilon_decay)
            self.episodes = config.get('episodes', episodes)
            self.batch_size = config.get('batch_size', batch_size)
            self.memory_size = config.get('memory_size', memory_size)
    
    def optimize(self, delivery_requests, depots, drones):
        """
        경로 최적화 실행
        """
        print(f"Deep Q-Network 최적화 시작 (목표: {self.optimization_target})")
        
        # 문제 데이터 준비
        self.delivery_requests = delivery_requests
        self.depots = depots
        self.drones = drones
        
        # 신경망 초기화
        self._initialize_network()
        
        # 학습 과정
        best_reward = float('-inf')
        best_policy = None
        
        for episode in range(self.episodes):
            # 에피소드 실행
            total_reward, policy = self._run_episode()
            
            # 최적 정책 업데이트
            if total_reward > best_reward:
                best_reward = total_reward
                best_policy = copy.deepcopy(policy)
            
            # ε 감소
            self.epsilon = max(0.01, self.epsilon * self.epsilon_decay)
            
            if episode % 100 == 0:
                print(f"에피소드 {episode}: 총 보상 = {total_reward:.2f}, ε = {self.epsilon:.3f}")
        
        # 최적 정책을 경로로 변환
        optimal_routes = self._convert_policy_to_routes(best_policy)
        
        print(f"Deep Q-Network 최적화 완료!")
        print(f"최종 최고 보상: {best_reward:.2f}")
        
        return optimal_routes
    
    def _initialize_network(self):
        """
        신경망 초기화 (간단한 구현)
        """
        n_requests = len(self.delivery_requests)
        n_drones = len(self.drones)
        
        # 각 상태-행동 쌍에 대한 Q-값을 랜덤하게 초기화
        for request_id in range(n_requests):
            for drone_id in range(n_drones):
                state = self._get_state_vector(request_id)
                state_key = tuple(state)
                action = drone_id
                
                if state_key not in self.q_network:
                    self.q_network[state_key] = {}
                
                self.q_network[state_key][action] = random.random() * 0.1
    
    def _get_state_vector(self, request_id):
        """
        상태를 벡터로 변환
        """
        if request_id >= len(self.delivery_requests):
            return [0, 0, 0, 0, 0]
        
        request = self.delivery_requests[request_id]
        
        # 상태 벡터: [우선순위, 거리, 시간대, 음식가격, 건물높이]
        priority = request['priority']
        distance = request['distance_3d'] / 1000  # km 단위로 정규화
        hour = request['request_time'].hour / 24.0  # 0-1로 정규화
        price = request['food_price'] / 30000  # 0-1로 정규화 (최대 3만원 가정)
        height = request['customer_location']['height'] / 100  # 0-1로 정규화 (최대 100m 가정)
        
        return [priority, distance, hour, price, height]
    
    def _run_episode(self):
        """
        단일 에피소드 실행
        """
        n_requests = len(self.delivery_requests)
        n_drones = len(self.drones)
        
        # 드론별 할당된 요청 추적
        drone_assignments = [[] for _ in range(n_drones)]
        total_reward = 0
        
        # 각 요청에 대해 순차적으로 결정
        for request_id in range(n_requests):
            state = self._get_state_vector(request_id)
            state_key = tuple(state)
            
            # 행동 선택 (ε-greedy)
            action = self._select_action(state_key, n_drones)
            
            # 보상 계산
            reward = self._calculate_reward(request_id, action, drone_assignments)
            total_reward += reward
            
            # 다음 상태
            next_state = None
            if request_id + 1 < n_requests:
                next_state = self._get_state_vector(request_id + 1)
                next_state_key = tuple(next_state)
            else:
                next_state_key = None
            
            # 경험을 메모리에 저장
            self._store_experience(state_key, action, reward, next_state_key, n_drones)
            
            # 신경망 학습
            if len(self.memory) >= self.batch_size:
                self._train_network()
            
            # 할당 업데이트
            drone_assignments[action].append(request_id)
        
        # 정책 생성
        policy = self._extract_policy(n_requests, n_drones)
        
        return total_reward, policy
    
    def _select_action(self, state_key, n_drones):
        """
        ε-greedy 정책으로 행동 선택
        """
        if random.random() < self.epsilon:
            # 탐험: 무작위 행동
            return random.randint(0, n_drones - 1)
        else:
            # 활용: 최적 행동
            return self._get_best_action(state_key, n_drones)
    
    def _get_best_action(self, state_key, n_drones):
        """
        현재 상태에서 최적 행동 선택
        """
        if state_key not in self.q_network:
            return random.randint(0, n_drones - 1)
        
        best_action = 0
        best_q_value = float('-inf')
        
        for action in range(n_drones):
            q_value = self.q_network[state_key].get(action, 0.0)
            
            if q_value > best_q_value:
                best_q_value = q_value
                best_action = action
        
        return best_action
    
    def _calculate_reward(self, request_id, drone_id, drone_assignments):
        """
        보상 계산
        """
        if request_id >= len(self.delivery_requests):
            return 0
        
        request = self.delivery_requests[request_id]
        drone = self.drones[drone_id]
        
        # 기본 보상 (거리 기반)
        distance = request['distance_3d']
        base_reward = -distance / 1000  # 거리가 멀수록 낮은 보상
        
        # 우선순위 보상
        priority_bonus = request['priority'] * 10
        
        # 드론 부하 페널티
        load_penalty = -len(drone_assignments[drone_id]) * 5
        
        # 시간 제약 보상
        time_reward = 0
        if request['request_time'].hour in [12, 13, 18, 19]:  # 피크 시간대
            time_reward = 20
        
        # 목표에 따른 보상 조정
        if self.optimization_target == 'cost':
            # 비용 최적화: 거리와 부하에 더 큰 가중치
            reward = base_reward * 2 + priority_bonus + load_penalty * 2 + time_reward
        else:  # time
            # 시간 최적화: 우선순위와 시간대에 더 큰 가중치
            reward = base_reward + priority_bonus * 2 + load_penalty + time_reward * 2
        
        return reward
    
    def _store_experience(self, state, action, reward, next_state, n_drones):
        """
        경험을 메모리에 저장
        """
        experience = (state, action, reward, next_state, n_drones)
        self.memory.append(experience)
        
        # 메모리 크기 제한
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)
    
    def _train_network(self):
        """
        신경망 학습 (간단한 구현)
        """
        # 배치 샘플링
        batch = random.sample(self.memory, min(self.batch_size, len(self.memory)))
        
        for state, action, reward, next_state, n_drones in batch:
            # 현재 Q-값
            if state not in self.q_network:
                self.q_network[state] = {}
            
            current_q = self.q_network[state].get(action, 0.0)
            
            # 다음 상태의 최대 Q-값
            if next_state is not None and next_state in self.q_network:
                next_q_values = [self.q_network[next_state].get(a, 0.0) for a in range(n_drones)]
                max_next_q = max(next_q_values) if next_q_values else 0.0
            else:
                max_next_q = 0.0
            
            # Q-값 업데이트
            target_q = reward + self.discount_factor * max_next_q
            new_q = current_q + self.learning_rate * (target_q - current_q)
            
            self.q_network[state][action] = new_q
    
    def _extract_policy(self, n_requests, n_drones):
        """
        학습된 신경망에서 정책 추출
        """
        policy = []
        
        for request_id in range(n_requests):
            state = self._get_state_vector(request_id)
            state_key = tuple(state)
            best_action = self._get_best_action(state_key, n_drones)
            policy.append(best_action)
        
        return policy
    
    def _convert_policy_to_routes(self, policy):
        """
        정책을 경로 형태로 변환
        """
        n_requests = len(self.delivery_requests)
        n_drones = len(self.drones)
        
        # 드론별 할당된 요청
        drone_assignments = [[] for _ in range(n_drones)]
        
        for request_id, drone_id in enumerate(policy):
            if request_id < n_requests:
                drone_assignments[drone_id].append(request_id)
        
        # 경로 생성
        routes = []
        
        for drone_id, assigned_requests in enumerate(drone_assignments):
            if not assigned_requests:
                continue
            
            # 경로 포인트 생성
            path_points = []
            
            # Depot에서 시작
            depot = self.drones[drone_id]['current_location']
            path_points.append({
                'longitude': depot['longitude'],
                'latitude': depot['latitude'],
                'height': depot['height'],
                'type': 'depot'
            })
            
            # 할당된 요청들을 우선순위 순으로 정렬
            sorted_requests = sorted(assigned_requests, 
                                   key=lambda x: self.delivery_requests[x]['priority'],
                                   reverse=True)
            
            # 경로 따라 포인트 추가
            for request_id in sorted_requests:
                request = self.delivery_requests[request_id]
                
                # 식당 방문
                path_points.append({
                    'longitude': request['restaurant_location']['longitude'],
                    'latitude': request['restaurant_location']['latitude'],
                    'height': request['restaurant_location']['height'],
                    'type': 'restaurant',
                    'request_id': request_id
                })
                
                # 고객 방문
                path_points.append({
                    'longitude': request['customer_location']['longitude'],
                    'latitude': request['customer_location']['latitude'],
                    'height': request['customer_location']['height'],
                    'type': 'customer',
                    'request_id': request_id
                })
            
            # 경로 비용 계산
            cost, time, distance = self._calculate_route_cost(drone_id, path_points)
            
            routes.append({
                'drone_id': drone_id,
                'path': path_points,
                'total_distance': distance,
                'total_time': time,
                'total_cost': cost
            })
        
        return routes
    
    def _calculate_route_cost(self, drone_id, path_points):
        """
        경로 비용 계산
        """
        if len(path_points) < 2:
            return 0, 0, 0
        
        drone = self.drones[drone_id]
        max_speed = drone['max_speed']
        
        total_cost = 0
        total_time = 0
        total_distance = 0
        
        for i in range(len(path_points) - 1):
            current = path_points[i]
            next_point = path_points[i + 1]
            
            # 거리 계산
            dx = (next_point['longitude'] - current['longitude']) * 111000
            dy = (next_point['latitude'] - current['latitude']) * 111000
            dz = next_point['height'] - current['height']
            
            distance = np.sqrt(dx**2 + dy**2 + dz**2)
            total_distance += distance
            
            # 시간 및 비용 계산
            time = distance / max_speed
            cost = distance * 0.5 + time / 3600 * 100  # 연료비 + 운영비
            
            total_cost += cost
            total_time += time
        
        return total_cost, total_time, total_distance 