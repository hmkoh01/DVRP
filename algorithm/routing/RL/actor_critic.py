"""
Actor-Critic을 사용한 드론 배달 경로 최적화
"""

import numpy as np
import random
import copy
from typing import List, Dict, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

class ActorCritic:
    """
    Actor-Critic 클래스 (간단한 구현)
    """
    
    def __init__(self, optimization_target='cost', learning_rate=0.001, 
                 discount_factor=0.95, episodes=1000):
        """
        초기화
        """
        self.optimization_target = optimization_target
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.episodes = episodes
        
        # Actor (정책 네트워크)
        self.actor_network = {}
        
        # Critic (가치 네트워크)
        self.critic_network = {}
    
    def optimize(self, delivery_requests, depots, drones):
        """
        경로 최적화 실행
        """
        print(f"Actor-Critic 최적화 시작 (목표: {self.optimization_target})")
        
        # 문제 데이터 준비
        self.delivery_requests = delivery_requests
        self.depots = depots
        self.drones = drones
        
        # 네트워크 초기화
        self._initialize_networks()
        
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
        
        print(f"Actor-Critic 최적화 완료!")
        print(f"최종 최고 보상: {best_reward:.2f}")
        
        return optimal_routes
    
    def _initialize_networks(self):
        """
        Actor와 Critic 네트워크 초기화
        """
        n_requests = len(self.delivery_requests)
        n_drones = len(self.drones)
        
        # 각 상태에 대한 정책과 가치를 랜덤하게 초기화
        for request_id in range(n_requests):
            state = self._get_state_vector(request_id)
            state_key = tuple(state)
            
            # Actor: 각 행동의 확률
            if state_key not in self.actor_network:
                self.actor_network[state_key] = {}
            
            for action in range(n_drones):
                self.actor_network[state_key][action] = 1.0 / n_drones  # 균등 분포
            
            # Critic: 상태 가치
            self.critic_network[state_key] = 0.0
    
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
            
            # Actor로부터 행동 선택
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
            
            # Advantage 계산
            advantage = self._calculate_advantage(state_key, action, reward, next_state_key)
            
            # 네트워크 업데이트
            self._update_networks(state_key, action, advantage, reward)
            
            # 할당 업데이트
            drone_assignments[action].append(request_id)
        
        # 정책 생성
        policy = self._extract_policy(n_requests, n_drones)
        
        return total_reward, policy
    
    def _select_action(self, state_key, n_drones):
        """
        Actor 네트워크로부터 행동 선택
        """
        if state_key not in self.actor_network:
            # 초기화되지 않은 상태인 경우 균등 분포
            return random.randint(0, n_drones - 1)
        
        # 정책에서 확률적으로 행동 선택
        actions = list(range(n_drones))
        probabilities = [self.actor_network[state_key].get(a, 1.0/n_drones) for a in actions]
        
        # 확률 정규화
        total_prob = sum(probabilities)
        if total_prob > 0:
            probabilities = [p / total_prob for p in probabilities]
        else:
            probabilities = [1.0 / n_drones] * n_drones
        
        return np.random.choice(actions, p=probabilities)
    
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
    
    def _calculate_advantage(self, state_key, action, reward, next_state_key):
        """
        Advantage 함수 계산
        """
        # 현재 상태의 가치
        current_value = self.critic_network.get(state_key, 0.0)
        
        # 다음 상태의 가치
        if next_state_key is not None:
            next_value = self.critic_network.get(next_state_key, 0.0)
        else:
            next_value = 0.0
        
        # TD 오차 계산
        td_target = reward + self.discount_factor * next_value
        td_error = td_target - current_value
        
        return td_error
    
    def _update_networks(self, state_key, action, advantage, reward):
        """
        Actor와 Critic 네트워크 업데이트
        """
        # Actor 업데이트 (정책 경사)
        if state_key in self.actor_network:
            # 현재 정책에서 선택된 행동의 확률 증가
            current_prob = self.actor_network[state_key].get(action, 1.0/len(self.drones))
            
            # Advantage가 양수면 확률 증가, 음수면 감소
            if advantage > 0:
                new_prob = min(1.0, current_prob + self.learning_rate * advantage)
            else:
                new_prob = max(0.01, current_prob + self.learning_rate * advantage)
            
            self.actor_network[state_key][action] = new_prob
            
            # 다른 행동들의 확률 조정
            other_actions = [a for a in self.actor_network[state_key].keys() if a != action]
            if other_actions:
                total_other_prob = sum(self.actor_network[state_key][a] for a in other_actions)
                if total_other_prob > 0:
                    for other_action in other_actions:
                        self.actor_network[state_key][other_action] *= (1.0 - new_prob) / total_other_prob
        
        # Critic 업데이트 (가치 함수)
        if state_key in self.critic_network:
            current_value = self.critic_network[state_key]
            target_value = reward + self.discount_factor * 0.0  # 간단한 구현
            new_value = current_value + self.learning_rate * (target_value - current_value)
            self.critic_network[state_key] = new_value
    
    def _extract_policy(self, n_requests, n_drones):
        """
        학습된 Actor 네트워크에서 정책 추출
        """
        policy = []
        
        for request_id in range(n_requests):
            state = self._get_state_vector(request_id)
            state_key = tuple(state)
            
            if state_key in self.actor_network:
                # 가장 높은 확률을 가진 행동 선택
                best_action = max(self.actor_network[state_key].keys(),
                                key=lambda a: self.actor_network[state_key][a])
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