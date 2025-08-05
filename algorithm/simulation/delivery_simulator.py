"""
드론 배달 시뮬레이터
"""

import numpy as np
import copy
from datetime import datetime, timedelta
from typing import List, Dict, Any
import warnings
warnings.filterwarnings('ignore')

class DeliverySimulator:
    """
    드론 배달 시뮬레이터 클래스
    """
    
    def __init__(self, delivery_requests, routes, depots, drones, optimization_target='cost'):
        """
        초기화
        """
        self.delivery_requests = delivery_requests
        self.routes = routes
        self.depots = depots
        self.drones = drones
        self.optimization_target = optimization_target
        
        # 시뮬레이션 상태
        self.current_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        self.simulation_duration = timedelta(hours=24)
        self.time_step = timedelta(minutes=1)
        
        # 드론 상태
        self.drone_states = {}
        self.delivery_results = []
        
        # 설정 파일에서 파라미터 로드
        from ..config import SIMULATION_CONFIG
        self.time_step = timedelta(seconds=SIMULATION_CONFIG['time_step'])
        self.simulation_duration = timedelta(minutes=SIMULATION_CONFIG['simulation_duration'])
    
    def run_simulation(self):
        """
        시뮬레이션 실행
        """
        print("드론 배달 시뮬레이션 시작...")
        
        # 초기화
        self._initialize_simulation()
        
        # 시뮬레이션 루프
        simulation_time = 0
        while simulation_time < self.simulation_duration:
            # 현재 시간 업데이트
            self.current_time += self.time_step
            simulation_time += self.time_step
            
            # 드론 상태 업데이트
            self._update_drone_states()
            
            # 배달 요청 처리
            self._process_delivery_requests()
            
            # 충돌 검사
            self._check_collisions()
            
            # 배터리 관리
            self._manage_battery()
            
            # 결과 기록
            if simulation_time.total_seconds() % 300 == 0:  # 5분마다 기록
                self._record_simulation_state()
        
        # 최종 결과 생성
        final_results = self._generate_final_results()
        
        print("드론 배달 시뮬레이션 완료!")
        
        return final_results
    
    def _initialize_simulation(self):
        """
        시뮬레이션 초기화
        """
        # 드론 상태 초기화
        for drone in self.drones:
            self.drone_states[drone['drone_id']] = {
                'drone_id': drone['drone_id'],
                'current_location': copy.deepcopy(drone['current_location']),
                'status': 'idle',  # idle, busy, charging, maintenance
                'current_route': None,
                'current_route_index': 0,
                'battery_level': drone['battery_capacity'],
                'completed_deliveries': 0,
                'total_distance': 0.0,
                'total_cost': 0.0,
                'current_delivery': None
            }
        
        # 배달 요청 상태 초기화
        for request in self.delivery_requests:
            request['status'] = 'pending'
            request['assigned_drone'] = None
            request['actual_delivery_time'] = None
            request['delivery_cost'] = 0.0
    
    def _update_drone_states(self):
        """
        드론 상태 업데이트
        """
        for drone_id, drone_state in self.drone_states.items():
            if drone_state['status'] == 'busy' and drone_state['current_route']:
                # 경로 따라 이동
                self._move_drone_along_route(drone_id)
            elif drone_state['status'] == 'charging':
                # 배터리 충전
                self._charge_drone_battery(drone_id)
    
    def _move_drone_along_route(self, drone_id):
        """
        드론을 경로 따라 이동
        """
        drone_state = self.drone_states[drone_id]
        route = drone_state['current_route']
        route_index = drone_state['current_route_index']
        
        if route_index >= len(route['path']):
            # 경로 완료
            self._complete_route(drone_id)
            return
        
        # 다음 목적지
        next_point = route['path'][route_index]
        current_location = drone_state['current_location']
        
        # 거리 계산
        distance = self._calculate_distance_3d(current_location, next_point)
        
        # 이동 시간 계산
        drone = next(d for d in self.drones if d['drone_id'] == drone_id)
        move_time = distance / drone['max_speed']
        
        # 시간 단위로 이동 (간단한 구현)
        if move_time <= self.time_step.total_seconds():
            # 목적지 도달
            drone_state['current_location'] = copy.deepcopy(next_point)
            drone_state['current_route_index'] += 1
            drone_state['total_distance'] += distance
            
            # 배터리 소모
            battery_consumption = distance * 0.01  # km당 1% 소모
            drone_state['battery_level'] -= battery_consumption
            
            # 비용 계산
            cost = distance * 0.5 + move_time / 3600 * 100  # 연료비 + 운영비
            drone_state['total_cost'] += cost
            
            # 배달 완료 체크
            if next_point['type'] == 'customer':
                self._complete_delivery(drone_id, next_point)
    
    def _complete_route(self, drone_id):
        """
        경로 완료 처리
        """
        drone_state = self.drone_states[drone_id]
        drone_state['status'] = 'idle'
        drone_state['current_route'] = None
        drone_state['current_route_index'] = 0
        drone_state['current_delivery'] = None
    
    def _complete_delivery(self, drone_id, customer_point):
        """
        배달 완료 처리
        """
        drone_state = self.drone_states[drone_id]
        
        if drone_state['current_delivery']:
            request_id = drone_state['current_delivery']
            request = next(r for r in self.delivery_requests if r['request_id'] == request_id)
            
            # 배달 완료
            request['status'] = 'completed'
            request['actual_delivery_time'] = self.current_time
            request['delivery_cost'] = drone_state['total_cost']
            
            drone_state['completed_deliveries'] += 1
            
            # 결과 기록
            self.delivery_results.append({
                'request_id': request_id,
                'drone_id': drone_id,
                'completion_time': self.current_time,
                'delivery_cost': request['delivery_cost'],
                'distance': drone_state['total_distance']
            })
    
    def _process_delivery_requests(self):
        """
        배달 요청 처리
        """
        # 대기 중인 요청들을 시간 순으로 정렬
        pending_requests = [r for r in self.delivery_requests if r['status'] == 'pending']
        pending_requests.sort(key=lambda x: x['request_time'])
        
        for request in pending_requests:
            # 요청 시간이 되었는지 확인
            if request['request_time'] <= self.current_time:
                # 사용 가능한 드론 찾기
                available_drone = self._find_available_drone()
                if available_drone:
                    # 드론 할당
                    self._assign_drone_to_request(available_drone, request)
    
    def _find_available_drone(self):
        """
        사용 가능한 드론 찾기
        """
        for drone_id, drone_state in self.drone_states.items():
            if (drone_state['status'] == 'idle' and 
                drone_state['battery_level'] > 20):  # 20% 이상 배터리
                return drone_id
        return None
    
    def _assign_drone_to_request(self, drone_id, request):
        """
        드론을 요청에 할당
        """
        # 해당 드론의 경로 찾기
        drone_route = next((r for r in self.routes if r['drone_id'] == drone_id), None)
        
        if drone_route:
            # 드론 상태 업데이트
            drone_state = self.drone_states[drone_id]
            drone_state['status'] = 'busy'
            drone_state['current_route'] = drone_route
            drone_state['current_route_index'] = 0
            drone_state['current_delivery'] = request['request_id']
            
            # 요청 상태 업데이트
            request['status'] = 'assigned'
            request['assigned_drone'] = drone_id
    
    def _check_collisions(self):
        """
        충돌 검사
        """
        # 간단한 충돌 검사 (같은 위치에 있는 드론들)
        drone_positions = {}
        
        for drone_id, drone_state in self.drone_states.items():
            if drone_state['status'] == 'busy':
                pos_key = (drone_state['current_location']['longitude'],
                          drone_state['current_location']['latitude'],
                          drone_state['current_location']['height'])
                
                if pos_key in drone_positions:
                    # 충돌 발생
                    print(f"충돌 발생: 드론 {drone_id}와 드론 {drone_positions[pos_key]}")
                    # 충돌 처리 (간단한 구현)
                    self._handle_collision(drone_id, drone_positions[pos_key])
                else:
                    drone_positions[pos_key] = drone_id
    
    def _handle_collision(self, drone1_id, drone2_id):
        """
        충돌 처리
        """
        # 두 드론 모두 정지
        self.drone_states[drone1_id]['status'] = 'maintenance'
        self.drone_states[drone2_id]['status'] = 'maintenance'
        
        # 비용 추가
        collision_cost = 1000  # 충돌 비용
        self.drone_states[drone1_id]['total_cost'] += collision_cost
        self.drone_states[drone2_id]['total_cost'] += collision_cost
    
    def _manage_battery(self):
        """
        배터리 관리
        """
        for drone_id, drone_state in self.drone_states.items():
            # 배터리가 낮으면 충전
            if drone_state['battery_level'] < 20 and drone_state['status'] == 'idle':
                drone_state['status'] = 'charging'
            
            # 충전 중이면 배터리 증가
            elif drone_state['status'] == 'charging':
                drone = next(d for d in self.drones if d['drone_id'] == drone_id)
                charge_rate = 0.5  # 분당 0.5% 충전
                drone_state['battery_level'] += charge_rate
                
                # 충전 완료
                if drone_state['battery_level'] >= drone['battery_capacity']:
                    drone_state['battery_level'] = drone['battery_capacity']
                    drone_state['status'] = 'idle'
    
    def _record_simulation_state(self):
        """
        시뮬레이션 상태 기록
        """
        # 현재 활성 드론 수
        active_drones = sum(1 for state in self.drone_states.values() 
                          if state['status'] == 'busy')
        
        # 완료된 배달 수
        completed_deliveries = sum(1 for r in self.delivery_requests 
                                 if r['status'] == 'completed')
        
        # 총 비용
        total_cost = sum(state['total_cost'] for state in self.drone_states.values())
        
        # 상태 기록 (간단한 구현)
        pass
    
    def _calculate_distance_3d(self, point1, point2):
        """
        3D 거리 계산
        """
        dx = (point2['longitude'] - point1['longitude']) * 111000
        dy = (point2['latitude'] - point1['latitude']) * 111000
        dz = point2['height'] - point1['height']
        
        return np.sqrt(dx**2 + dy**2 + dz**2)
    
    def _charge_drone_battery(self, drone_id):
        """
        드론 배터리 충전
        """
        drone_state = self.drone_states[drone_id]
        drone = next(d for d in self.drones if d['drone_id'] == drone_id)
        
        charge_rate = 0.5  # 분당 0.5% 충전
        drone_state['battery_level'] += charge_rate
        
        # 충전 완료
        if drone_state['battery_level'] >= drone['battery_capacity']:
            drone_state['battery_level'] = drone['battery_capacity']
            drone_state['status'] = 'idle'
    
    def _generate_final_results(self):
        """
        최종 결과 생성
        """
        # 완료된 배달 통계
        completed_deliveries = [r for r in self.delivery_requests if r['status'] == 'completed']
        failed_deliveries = [r for r in self.delivery_requests if r['status'] == 'pending']
        
        # 드론별 통계
        drone_statistics = {}
        for drone_id, drone_state in self.drone_states.items():
            drone_statistics[drone_id] = {
                'completed_deliveries': drone_state['completed_deliveries'],
                'total_distance': drone_state['total_distance'],
                'total_cost': drone_state['total_cost'],
                'utilization_rate': self._calculate_utilization_rate(drone_state)
            }
        
        # 전체 통계
        total_cost = sum(state['total_cost'] for state in self.drone_states.values())
        total_distance = sum(state['total_distance'] for state in self.drone_states.values())
        
        # 평균 배달 시간
        delivery_times = []
        for delivery in self.delivery_results:
            request = next(r for r in self.delivery_requests if r['request_id'] == delivery['request_id'])
            delivery_time = (delivery['completion_time'] - request['request_time']).total_seconds() / 60  # 분 단위
            delivery_times.append(delivery_time)
        
        avg_delivery_time = np.mean(delivery_times) if delivery_times else 0
        
        results = {
            'simulation_duration': self.simulation_duration.total_seconds() / 3600,  # 시간
            'total_requests': len(self.delivery_requests),
            'completed_deliveries': len(completed_deliveries),
            'failed_deliveries': len(failed_deliveries),
            'completion_rate': len(completed_deliveries) / len(self.delivery_requests),
            'total_cost': total_cost,
            'total_distance': total_distance,
            'average_delivery_time': avg_delivery_time,
            'drone_statistics': drone_statistics,
            'delivery_results': self.delivery_results,
            'drone_states': self.drone_states
        }
        
        return results
    
    def _calculate_utilization_rate(self, drone_state):
        """
        드론 활용률 계산
        """
        # 간단한 활용률 계산 (완료된 배달 수 기반)
        total_requests = len(self.delivery_requests)
        if total_requests > 0:
            return drone_state['completed_deliveries'] / total_requests
        return 0.0 