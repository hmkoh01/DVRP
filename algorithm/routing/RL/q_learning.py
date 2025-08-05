"""
Q-Learning을 사용한 드론 배달 경로 최적화
"""

import numpy as np
import random
import copy
from typing import List, Dict, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

class QLearning:
    """
    Q-Learning 클래스
    """
    
    def __init__(self, optimization_target='cost', learning_rate=0.1, 
                 discount_factor=0.95, epsilon=0.1, episodes=1000):
        """
        초기화
        """
        self.optimization_target = optimization_target
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.episodes = episodes
        
        # Q-테이블
        self.q_table = {}
        
        # 설정 파일에서 파라미터 로드
        from ..config import ALGORITHM_CONFIG
        if 'q_learning' in ALGORITHM_CONFIG:
            config = ALGORITHM_CONFIG['q_learning']
            self.learning_rate = config.get('learning_rate', learning_rate)
            self.discount_factor = config.get('discount_factor', discount_factor)
            self.epsilon = config.get('epsilon', epsilon)
            self.episodes = config.get('episodes', episodes)
    
    def optimize(self, delivery_requests, depots, drones):
        """
        경로 최적화 실행
        """
        print(f"Q-Learning 최적화 시작 (목표: {self.optimization_target})")
        
        # 문제 데이터 준비
        self.delivery_requests = delivery_requests
        self.depots = depots
        self.drones = drones
        
        # 환경 초기화
        self._initialize_q_table()
        
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
            
            if episode % 100 == 0:
                print(f"에피소드 {episode}: 총 보상 = {total_reward:.2f}")
        
        # 최적 정책을 경로로 변환
        optimal_routes = self._convert_policy_to_routes(best_policy)
        
        print(f"Q-Learning 최적화 완료!")
        print(f"최종 최고 보상: {best_reward:.2f}")
        
        return optimal_routes
    
    def _initialize_q_table(self):
        """
        Q-테이블 초기화
        """
        n_requests = len(self.delivery_requests)
        n_drones = len(self.drones)
        
        # 상태-행동 쌍에 대한 Q-값 초기화
        for request_id in range(n_requests):
            for drone_id in range(n_drones):
                state = self._get_state(request_id)
                action = drone_id
                state_action = (state, action)
                self.q_table[state_action] = 0.0
    
    def _get_state(self, request_id):
        """
        상태 정의
        """
        if request_id >= len(self.delivery_requests):
            return None
        
        request = self.delivery_requests[request_id]
        
        # 상태: (요청 우선순위, 거리 범위, 시간대)
        priority = request['priority']
        distance = request['distance_3d']
        hour = request['request_time'].hour
        
        # 우선순위 범주화
        if priority < 0.5:
            priority_category = 0
        elif priority < 1.0:
            priority_category = 1
        else:
            priority_category = 2
        
        # 거리 범주화
        if distance < 500:
            distance_category = 0
        elif distance < 1000:
            distance_category = 1
        else:
            distance_category = 2
        
        # 시간대 범주화
        if 6 <= hour <= 11:
            time_category = 0  # 오전
        elif 12 <= hour <= 17:
            time_category = 1  # 오후
        else:
            time_category = 2  # 저녁/새벽
        
        return (priority_category, distance_category, time_category)
    
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
            state = self._get_state(request_id)
            
            if state is None:
                continue
            
            # 행동 선택 (ε-greedy)
            action = self._select_action(state, n_drones)
            
            # 보상 계산
            reward = self._calculate_reward(request_id, action, drone_assignments)
            total_reward += reward
            
            # 다음 상태
            next_state = self._get_state(request_id + 1) if request_id + 1 < n_requests else None
            
            # Q-값 업데이트
            self._update_q_value(state, action, reward, next_state, n_drones)
            
            # 할당 업데이트
            drone_assignments[action].append(request_id)
        
        # 정책 생성
        policy = self._extract_policy(n_requests, n_drones)
        
        return total_reward, policy
    
    def _select_action(self, state, n_drones):
        """
        ε-greedy 정책으로 행동 선택
        """
        if random.random() < self.epsilon:
            # 탐험: 무작위 행동
            return random.randint(0, n_drones - 1)
        else:
            # 활용: 최적 행동
            return self._get_best_action(state, n_drones)
    
    def _get_best_action(self, state, n_drones):
        """
        현재 상태에서 최적 행동 선택
        """
        best_action = 0
        best_q_value = float('-inf')
        
        for action in range(n_drones):
            state_action = (state, action)
            q_value = self.q_table.get(state_action, 0.0)
            
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
    
    def _update_q_value(self, state, action, reward, next_state, n_drones):
        """
        Q-값 업데이트
        """
        state_action = (state, action)
        current_q = self.q_table.get(state_action, 0.0)
        
        # 다음 상태의 최대 Q-값
        if next_state is not None:
            next_q_values = []
            for next_action in range(n_drones):
                next_state_action = (next_state, next_action)
                next_q_values.append(self.q_table.get(next_state_action, 0.0))
            max_next_q = max(next_q_values) if next_q_values else 0.0
        else:
            max_next_q = 0.0
        
        # Q-값 업데이트
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        
        self.q_table[state_action] = new_q
    
    def _extract_policy(self, n_requests, n_drones):
        """
        학습된 Q-테이블에서 정책 추출
        """
        policy = []
        
        for request_id in range(n_requests):
            state = self._get_state(request_id)
            if state is not None:
                best_action = self._get_best_action(state, n_drones)
                policy.append(best_action)
            else:
                policy.append(0)
        
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