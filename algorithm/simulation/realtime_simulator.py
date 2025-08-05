"""
실시간 드론 배달 시뮬레이션 모듈
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Rectangle
import time
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

class RealtimeDroneSimulator:
    """
    실시간 드론 배달 시뮬레이터
    """
    
    def __init__(self, depots, routes, building_data, config):
        """
        초기화
        """
        self.depots = depots
        self.routes = routes
        self.building_data = building_data
        self.config = config
        
        # 시뮬레이션 상태
        self.current_time = 0
        self.drones = []
        self.delivery_requests = []
        self.completed_deliveries = []
        self.simulation_events = []
        self.dynamic_requests = []
        
        # 동적 요청 설정
        self.generation_rate = 0.1
        self.max_requests_per_batch = 3
        self.time_window_minutes = 60
        self.last_request_time = 0
        
        # 시각화 설정
        self.fig, self.ax = None, None
        self.animation = None
        self.enable_animation = True
        
        # 드론 초기화
        self._initialize_drones()
        
        # 배달 요청 생성기 초기화
        from algorithm.utils.delivery_generator import DeliveryRequestGenerator
        from algorithm.utils.data_loader import BuildingDataLoader
        
        data_loader = BuildingDataLoader()
        data_loader.load_building_data('map/postech_map/merged_maps/postech_buildings_real_height.csv')
        self.delivery_generator = DeliveryRequestGenerator(data_loader)
    
    def setup_dynamic_requests(self, generation_rate=0.3, max_requests_per_batch=5, time_window_minutes=60):
        """
        동적 요청 생성 설정
        """
        self.generation_rate = generation_rate
        self.max_requests_per_batch = max_requests_per_batch
        self.time_window_minutes = time_window_minutes
        print(f"동적 요청 설정 완료:")
        print(f"  - 생성률: {generation_rate} 요청/분")
        print(f"  - 최대 배치 크기: {max_requests_per_batch}개")
        print(f"  - 시간 윈도우: {time_window_minutes}분")
        
    def _initialize_drones(self):
        """
        드론 초기화
        """
        self.drones = []
        
        for depot in self.depots:
            # depot당 3개 드론 할당 (기본값)
            for i in range(3):
                drone = {
                    'id': f"drone_{depot['id']}_{i+1}",
                    'depot_id': depot['id'],  # depot['id']를 depot_id로 사용
                    'current_lat': depot['latitude'],
                    'current_lon': depot['longitude'],
                    'current_height': 50,  # 50m 고도
                    'status': 'idle',  # idle, flying, pickup, delivery
                    'battery': 100,  # 배터리 잔량 (%)
                    'current_route': None,
                    'current_request': None,
                    'speed': 15,  # m/s
                    'max_payload': 2.0,  # kg
                    'current_payload': 0.0
                }
                self.drones.append(drone)
    
    def run_simulation(self, duration_minutes=60, realtime_factor=1.0, enable_animation=True):
        """
        실시간 시뮬레이션 실행 (동적 요청 + 애니메이션)
        """
        print(f"=== 실시간 동적 드론 배달 시뮬레이션 시작 ===")
        print(f"시뮬레이션 시간: {duration_minutes}분")
        print(f"실시간 배율: {realtime_factor}x")
        print(f"애니메이션 활성화: {enable_animation}")
        
        self.enable_animation = enable_animation
        
        # 시각화 초기화
        if self.enable_animation:
            self._setup_visualization()
        
        # 초기 배달 요청 생성 (기존 라우트에서)
        self._generate_initial_requests()
        
        # 애니메이션 시작
        if self.enable_animation:
            try:
                self.animation = animation.FuncAnimation(
                    self.fig, self._update_frame, 
                    frames=duration_minutes * 60,  # 1초당 1프레임
                    interval=1000 / realtime_factor,  # 밀리초
                    repeat=False,
                    blit=False  # blit=False로 설정하여 더 안정적으로
                )
                plt.show()
            except Exception as e:
                print(f"애니메이션 오류, 비애니메이션 모드로 전환: {e}")
                # 애니메이션 없이 시뮬레이션 실행
                for frame in range(duration_minutes * 60):
                    self._update_frame(frame)
        else:
            # 애니메이션 없이 시뮬레이션 실행
            for frame in range(duration_minutes * 60):
                self._update_frame(frame)
        
        return self._get_simulation_results()
    
    def _generate_initial_requests(self):
        """
        초기 배달 요청 생성 (기존 라우트에서)
        """
        print("초기 배달 요청 생성 중...")
        
        for route in self.routes:
            if 'requests' in route:
                # 기존 형식 처리
                for request in route['requests']:
                    # 요청에 고유 ID 추가
                    request['id'] = f"req_{len(self.delivery_requests)}"
                    request['status'] = 'pending'
                    request['created_time'] = 0
                    self.delivery_requests.append(request)
            elif 'path' in route:
                # genetic algorithm 형식 처리
                restaurant_points = []
                customer_points = []
                
                # 먼저 모든 포인트를 분류
                for point in route['path']:
                    if point['type'] == 'restaurant':
                        restaurant_points.append(point)
                    elif point['type'] == 'customer':
                        customer_points.append(point)
                
                # restaurant와 customer 포인트를 매칭하여 요청 생성
                for i, restaurant_point in enumerate(restaurant_points):
                    if i < len(customer_points):
                        customer_point = customer_points[i]
                        request = {
                            'id': f"req_{len(self.delivery_requests)}",
                            'request_id': restaurant_point.get('request_id', i),
                            'status': 'pending',
                            'created_time': 0,
                            'restaurant_location': restaurant_point,
                            'customer_location': customer_point
                        }
                        self.delivery_requests.append(request)
                        print(f"요청 생성: {request['id']} - 식당: ({restaurant_point['longitude']:.4f}, {restaurant_point['latitude']:.4f}), 고객: ({customer_point['longitude']:.4f}, {customer_point['latitude']:.4f})")
        
        print(f"초기 요청 생성 완료: {len(self.delivery_requests)}개")
    
    def _generate_dynamic_requests(self):
        """
        동적 배달 요청 생성
        """
        try:
            # 포아송 분포를 사용한 요청 수 결정
            expected_requests = self.generation_rate * 0.5  # 30초 = 0.5분
            num_requests = min(np.random.poisson(expected_requests), self.max_requests_per_batch)
            
            if num_requests > 0:
                # 현재 시간 기준으로 요청 생성
                current_datetime = datetime.now()
                
                for _ in range(num_requests):
                    request = self.delivery_generator._generate_single_request(
                        current_datetime, 
                        current_datetime + timedelta(minutes=1), 
                        len(self.delivery_requests)
                    )
                    
                    if request:
                        # 요청에 시뮬레이션 정보 추가
                        request['id'] = f"dynamic_req_{len(self.delivery_requests)}"
                        request['status'] = 'pending'
                        request['created_time'] = self.current_time
                        request['is_dynamic'] = True
                        
                        self.delivery_requests.append(request)
                        self.dynamic_requests.append(request)
                        
                        print(f"시간 {self.current_time}s: 새로운 동적 요청 생성 - {request['id']}")
                
                print(f"동적 요청 생성 완료: {num_requests}개 (총 요청: {len(self.delivery_requests)}개)")
        
        except Exception as e:
            print(f"동적 요청 생성 중 오류: {e}")
    
    def _setup_visualization(self):
        """
        시각화 설정
        """
        self.fig, self.ax = plt.subplots(figsize=(15, 10))
        
        # 건물 플롯
        self.ax.scatter(
            self.building_data['longitude'],
            self.building_data['latitude'],
            c='lightgray', s=5, alpha=0.3, label='Buildings'
        )
        
        # Depot 플롯 (출발지)
        for depot in self.depots:
            self.ax.scatter(
                depot['longitude'], depot['latitude'],
                c='red', s=300, marker='s', edgecolors='black', linewidth=2, 
                label=f"Depot {depot['id']}" if depot['id'] == self.depots[0]['id'] else ""
            )
            # Depot ID 표시
            self.ax.text(depot['longitude'], depot['latitude'], f"D{depot['id'].split('_')[-1]}", 
                        fontsize=10, ha='center', va='center', weight='bold', color='white')
        
        # 요청의 출발지(식당)와 도착지(고객) 표시
        self._plot_request_locations()
        
        # 경로 플롯
        for route in self.routes:
            route_points = self._get_route_points(route)
            if route_points:
                lons, lats = zip(*route_points)
                self.ax.plot(lons, lats, 'b--', alpha=0.2, linewidth=1)
        
        self.ax.set_xlabel('Longitude')
        self.ax.set_ylabel('Latitude')
        self.ax.set_title('Real-time Drone Delivery Simulation', fontsize=14, weight='bold')
        self.ax.legend(loc='upper right')
        self.ax.grid(True, alpha=0.2)
        
        # 축 범위 설정
        self.ax.set_xlim(self.building_data['longitude'].min() - 100, self.building_data['longitude'].max() + 100)
        self.ax.set_ylim(self.building_data['latitude'].min() - 100, self.building_data['latitude'].max() + 100)
    
    def _plot_request_locations(self):
        """
        요청의 출발지(식당)와 도착지(고객) 위치 표시
        """
        # 초기 요청들의 위치 표시
        for request in self.delivery_requests:
            if 'restaurant_location' in request and request['restaurant_location']:
                # 식당 위치 (출발지) - 파란색 원
                rest_lon = request['restaurant_location']['longitude']
                rest_lat = request['restaurant_location']['latitude']
                self.ax.scatter(rest_lon, rest_lat, c='blue', s=100, marker='o', 
                              alpha=0.7, edgecolors='darkblue', linewidth=1, label='Restaurants' if request == self.delivery_requests[0] else "")
                self.ax.text(rest_lon, rest_lat, 'R', fontsize=8, ha='center', va='center', 
                           weight='bold', color='white')
            
            if 'customer_location' in request and request['customer_location']:
                # 고객 위치 (도착지) - 초록색 삼각형
                cust_lon = request['customer_location']['longitude']
                cust_lat = request['customer_location']['latitude']
                self.ax.scatter(cust_lon, cust_lat, c='green', s=100, marker='^', 
                              alpha=0.7, edgecolors='darkgreen', linewidth=1, label='Customers' if request == self.delivery_requests[0] else "")
                self.ax.text(cust_lon, cust_lat, 'C', fontsize=8, ha='center', va='center', 
                           weight='bold', color='white')
    
    def _get_route_points(self, route):
        """
        경로의 모든 지점 반환
        """
        points = []
        
        # genetic algorithm 형식의 경로 처리
        if 'path' in route:
            for point in route['path']:
                points.append((point['longitude'], point['latitude']))
        else:
            # 기존 형식 처리
            depot = next(d for d in self.depots if d['id'] == route['depot_id'])
            points.append((depot['longitude'], depot['latitude']))
            
            # 각 요청의 픽업/배달 지점
            for request in route['requests']:
                points.append((request['pickup_lon'], request['pickup_lat']))
                points.append((request['delivery_lon'], request['delivery_lat']))
            
            # depot로 복귀
            points.append((depot['longitude'], depot['latitude']))
        
        return points
    
    def _generate_simulation_events(self, duration_minutes):
        """
        시뮬레이션 이벤트 생성
        """
        self.simulation_events = []
        
        # 배달 요청 생성 (시간별)
        for minute in range(0, duration_minutes, 5):  # 5분마다 요청
            for route in self.routes:
                for request in route['requests']:
                    event = {
                        'time': minute * 60,  # 초 단위
                        'type': 'delivery_request',
                        'request': request,
                        'route': route
                    }
                    self.simulation_events.append(event)
        
        # 시간순 정렬
        self.simulation_events.sort(key=lambda x: x['time'])
    
    def _update_frame(self, frame):
        """
        애니메이션 프레임 업데이트 (동적 요청 포함)
        """
        self.current_time = frame
        
        # 동적 요청 생성 (매 30초마다)
        if self.current_time % 30 == 0 and self.current_time > 0:
            self._generate_dynamic_requests()
        
        # 이벤트 처리
        self._process_events()
        
        # 드론 상태 업데이트
        self._update_drones()
        
        # 시각화 업데이트
        if self.enable_animation:
            self._update_visualization()
        
        # 진행 상황 출력 (매 10초마다)
        if self.current_time % 10 == 0:
            active_drones = len([d for d in self.drones if d['status'] != 'idle'])
            print(f"시간 {self.current_time}s: 활성 드론 {active_drones}개, 완료된 배달 {len(self.completed_deliveries)}개")
        
        return self.ax,
    
    def _process_events(self):
        """
        현재 시간의 이벤트 처리 (동적 요청 포함)
        """
        # 기존 이벤트 처리
        current_events = [e for e in self.simulation_events if e['time'] == self.current_time]
        
        for event in current_events:
            if event['type'] == 'delivery_request':
                self._assign_delivery_request(event['request'], event['route'])
        
        # 대기 중인 요청들을 드론에 할당 (매 프레임마다 시도)
        pending_requests = [req for req in self.delivery_requests if req['status'] == 'pending']
        
        for request in pending_requests:
            # customer_location이 None이면 건너뛰기
            if request.get('customer_location') is None:
                continue
                
            # 가장 가까운 depot의 유휴 드론 찾기
            assigned = self._assign_request_to_nearest_drone(request)
            if assigned:
                request['status'] = 'assigned'
                print(f"시간 {self.current_time}s: 요청 {request['id']} 할당됨")
        
        # 동적 요청도 처리
        dynamic_pending = [req for req in self.dynamic_requests if req['status'] == 'pending']
        for request in dynamic_pending:
            assigned = self._assign_request_to_nearest_drone(request)
            if assigned:
                request['status'] = 'assigned'
                print(f"시간 {self.current_time}s: 동적 요청 {request['id']} 할당됨")
        
        # 디버깅: 요청 상태 출력 (매 10초마다)
        if self.current_time % 10 == 0:
            pending_count = len([req for req in self.delivery_requests if req['status'] == 'pending'])
            assigned_count = len([req for req in self.delivery_requests if req['status'] == 'assigned'])
            idle_drones = len([d for d in self.drones if d['status'] == 'idle'])
            
            # 요청 구조 디버깅
            if pending_count > 0:
                sample_request = next(req for req in self.delivery_requests if req['status'] == 'pending')
                print(f"시간 {self.current_time}s: 대기 요청 {pending_count}개, 할당된 요청 {assigned_count}개, 유휴 드론 {idle_drones}개")
                print(f"  샘플 요청 구조: {list(sample_request.keys())}")
                print(f"  customer_location 존재: {'customer_location' in sample_request}")
                if 'customer_location' in sample_request:
                    print(f"  customer_location 값: {sample_request['customer_location']}")
            else:
                print(f"시간 {self.current_time}s: 대기 요청 {pending_count}개, 할당된 요청 {assigned_count}개, 유휴 드론 {idle_drones}개")
    
    def _assign_request_to_nearest_drone(self, request):
        """
        요청을 가장 가까운 유휴 드론에 할당
        """
        idle_drones = [d for d in self.drones if d['status'] == 'idle']
        
        if not idle_drones:
            return False
        
        # 고객 위치
        if request.get('customer_location') is None:
            return False  # 고객 위치가 없으면 할당 불가
        
        customer_lat = request['customer_location']['latitude']
        customer_lon = request['customer_location']['longitude']
        
        # 가장 가까운 드론 찾기
        nearest_drone = None
        min_distance = float('inf')
        
        for drone in idle_drones:
            distance = self._calculate_distance(
                drone['current_lat'], drone['current_lon'],
                customer_lat, customer_lon
            )
            
            if distance < min_distance:
                min_distance = distance
                nearest_drone = drone
        
        if nearest_drone:
            nearest_drone['current_request'] = request
            nearest_drone['status'] = 'flying'
            nearest_drone['target_type'] = 'pickup'  # 먼저 픽업
            
            print(f"시간 {self.current_time}s: 요청 {request['id']}를 드론 {nearest_drone['id']}에 할당 (거리: {min_distance:.4f}km)")
            return True
        
        return False
    
    def _assign_delivery_request(self, request, route):
        """
        배달 요청을 드론에 할당
        """
        # genetic algorithm 형식의 경로 처리
        if 'drone_id' in route:
            # 해당 드론 찾기
            drone = next((d for d in self.drones if d['id'] == f"drone_{d['depot_id']}_{route['drone_id']+1}"), None)
        else:
            # 기존 형식 처리
            depot_drones = [d for d in self.drones if d['depot_id'] == route['depot_id'] and d['status'] == 'idle']
            drone = depot_drones[0] if depot_drones else None
        
        if drone and drone['status'] == 'idle':
            drone['current_request'] = request
            drone['current_route'] = route
            drone['status'] = 'flying'
            
            print(f"시간 {self.current_time}s: 드론 {drone['id']}가 요청 {request['id']} 할당됨")
    
    def _update_drones(self):
        """
        드론 상태 업데이트
        """
        for drone in self.drones:
            if drone['status'] == 'idle':
                continue
            
            # 드론 이동
            self._move_drone(drone)
            
            # 배터리 소모
            drone['battery'] -= 0.1  # 1초당 0.1% 소모
            
            # 배터리 부족 시 depot로 복귀
            if drone['battery'] < 20:
                self._return_to_depot(drone)
    
    def _move_drone(self, drone):
        """
        드론 이동 처리
        """
        if not drone['current_request']:
            return
        
        request = drone['current_request']
        
        # 현재 위치에서 목표 지점까지의 거리 계산
        target_lat, target_lon = self._get_target_position(drone)
        distance = self._calculate_distance(
            drone['current_lat'], drone['current_lon'],
            target_lat, target_lon
        )
        
        # 1초 동안 이동할 수 있는 거리 (UTM 좌표계 고려)
        move_distance = drone['speed'] / 1000  # m/s를 km/s로 변환, UTM 좌표계에서는 1 단위 ≈ 1m
        
        if distance <= move_distance:
            # 목표 지점 도달
            drone['current_lat'] = target_lat
            drone['current_lon'] = target_lon
            
            # 상태 업데이트
            if drone['status'] == 'flying':
                if self._is_at_pickup(drone):
                    drone['status'] = 'pickup'
                    drone['current_payload'] = request.get('weight', 1.0)
                    print(f"시간 {self.current_time}s: 드론 {drone['id']} 픽업 완료")
                elif self._is_at_delivery(drone):
                    drone['status'] = 'delivery'
                    drone['current_payload'] = 0
                    self.completed_deliveries.append(request)
                    print(f"시간 {self.current_time}s: 드론 {drone['id']} 배달 완료")
                    
                    # 다음 목표 설정 또는 depot 복귀
                    self._set_next_target(drone)
        else:
            # 목표 지점으로 이동
            direction_lat = (target_lat - drone['current_lat']) / distance
            direction_lon = (target_lon - drone['current_lon']) / distance
            
            drone['current_lat'] += direction_lat * move_distance
            drone['current_lon'] += direction_lon * move_distance
            
            # 이동 로그 (매 5초마다)
            if self.current_time % 5 == 0:
                print(f"시간 {self.current_time}s: 드론 {drone['id']} 이동 중 - 목표까지 {distance:.4f}km 남음")
    
    def _get_target_position(self, drone):
        """
        드론의 현재 목표 위치 반환 (동적 요청 지원)
        """
        request = drone['current_request']
        
        if drone['status'] == 'flying':
            # 픽업 지점으로 이동 (식당 위치)
            if 'restaurant_location' in request and request['restaurant_location']:
                return request['restaurant_location']['latitude'], request['restaurant_location']['longitude']
            else:
                return request.get('pickup_lat', drone['current_lat']), request.get('pickup_lon', drone['current_lon'])
        elif drone['status'] == 'pickup':
            # 배달 지점으로 이동 (고객 위치)
            if 'customer_location' in request and request['customer_location']:
                return request['customer_location']['latitude'], request['customer_location']['longitude']
            else:
                return request.get('delivery_lat', drone['current_lat']), request.get('delivery_lon', drone['current_lon'])
        elif drone['status'] == 'delivery':
            # depot로 복귀
            depot = next(d for d in self.depots if d['id'] == drone['depot_id'])
            return depot['latitude'], depot['longitude']
        
        return drone['current_lat'], drone['current_lon']
    
    def _is_at_pickup(self, drone):
        """
        드론이 픽업 지점에 도달했는지 확인 (동적 요청 지원)
        """
        request = drone['current_request']
        
        if 'restaurant_location' in request:
            target_lat = request['restaurant_location']['latitude']
            target_lon = request['restaurant_location']['longitude']
        else:
            target_lat = request.get('pickup_lat', drone['current_lat'])
            target_lon = request.get('pickup_lon', drone['current_lon'])
        
        distance = self._calculate_distance(
            drone['current_lat'], drone['current_lon'],
            target_lat, target_lon
        )
        return distance < 0.0001  # 약 10m 이내
    
    def _is_at_delivery(self, drone):
        """
        드론이 배달 지점에 도달했는지 확인 (동적 요청 지원)
        """
        request = drone['current_request']
        
        if 'customer_location' in request:
            target_lat = request['customer_location']['latitude']
            target_lon = request['customer_location']['longitude']
        else:
            target_lat = request.get('delivery_lat', drone['current_lat'])
            target_lon = request.get('delivery_lon', drone['current_lon'])
        
        distance = self._calculate_distance(
            drone['current_lat'], drone['current_lon'],
            target_lat, target_lon
        )
        return distance < 0.0001  # 약 10m 이내
    
    def _set_next_target(self, drone):
        """
        드론의 다음 목표 설정
        """
        if drone['status'] == 'pickup':
            drone['status'] = 'flying'  # 배달 지점으로 이동
            print(f"시간 {self.current_time}s: 드론 {drone['id']} 배달 지점으로 이동 시작")
        elif drone['status'] == 'delivery':
            drone['status'] = 'flying'  # depot로 복귀
            print(f"시간 {self.current_time}s: 드론 {drone['id']} depot로 복귀 시작")
        elif drone['status'] == 'flying':
            # depot에 도달
            drone['status'] = 'idle'
            drone['current_request'] = None
            drone['current_route'] = None
            print(f"시간 {self.current_time}s: 드론 {drone['id']} depot 도달, 대기 상태로 전환")
    
    def _return_to_depot(self, drone):
        """
        드론을 depot로 복귀
        """
        depot = next(d for d in self.depots if d['id'] == drone['depot_id'])
        
        # depot로 이동
        distance = self._calculate_distance(
            drone['current_lat'], drone['current_lon'],
            depot['latitude'], depot['longitude']
        )
        
        move_distance = drone['speed'] / 111000
        
        if distance <= move_distance:
            # depot 도달
            drone['current_lat'] = depot['latitude']
            drone['current_lon'] = depot['longitude']
            drone['status'] = 'idle'
            drone['current_request'] = None
            drone['current_route'] = None
            drone['battery'] = 100  # 배터리 충전
        else:
            # depot로 이동
            direction_lat = (depot['latitude'] - drone['current_lat']) / distance
            direction_lon = (depot['longitude'] - drone['current_lon']) / distance
            
            drone['current_lat'] += direction_lat * move_distance
            drone['current_lon'] += direction_lon * move_distance
    
    def _calculate_distance(self, lat1, lon1, lat2, lon2):
        """
        두 지점 간 거리 계산 (km) - UTM 좌표계 고려
        """
        # UTM 좌표계에서는 1도 = 약 111km가 아닐 수 있음
        # 더 정확한 거리 계산을 위해 피타고라스 정리 사용
        lat_diff = lat1 - lat2
        lon_diff = lon1 - lon2
        distance = np.sqrt(lat_diff**2 + lon_diff**2)
        
        # UTM 좌표계에서 대략적인 변환 (1 단위 ≈ 1m)
        return distance / 1000  # m를 km로 변환
    
    def _update_visualization(self):
        """
        시각화 업데이트
        """
        # 기존 드론 마커와 텍스트 제거
        for artist in self.ax.collections[1:]:  # 첫 번째 컬렉션(건물)은 유지
            artist.remove()
        
        # 기존 텍스트 제거
        for artist in self.ax.texts:
            artist.remove()
        
        # Depot ID 다시 표시
        for depot in self.depots:
            self.ax.text(depot['longitude'], depot['latitude'], f"D{depot['id'].split('_')[-1]}", 
                        fontsize=10, ha='center', va='center', weight='bold', color='white')
        
        # 요청 위치 다시 표시
        self._plot_request_locations()
        
        # 드론 위치 업데이트 (더 크고 눈에 띄게)
        for drone in self.drones:
            if drone['status'] != 'idle':
                color = self._get_drone_color(drone)
                # 드론 마커를 더 크게
                self.ax.scatter(
                    drone['current_lon'], drone['current_lat'],
                    c=color, s=200, marker='^', alpha=0.9, edgecolors='black', linewidth=2
                )
                # 드론 ID 표시
                drone_id_short = drone['id'].split('_')[-1]  # 예: drone_depot_1_1 -> 1_1
                self.ax.text(drone['current_lon'], drone['current_lat'], drone_id_short, 
                           fontsize=9, ha='center', va='bottom', weight='bold', color='white',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.8))
                
                # 드론의 현재 목표 표시
                if drone['current_request']:
                    target_lat, target_lon = self._get_target_position(drone)
                    # 목표 지점까지의 선 그리기
                    self.ax.plot([drone['current_lon'], target_lon], [drone['current_lat'], target_lat], 
                               '--', color=color, alpha=0.6, linewidth=1)
        
        # 제목 업데이트
        self.ax.set_title(f'Real-time Dynamic Drone Delivery Simulation - Time: {self.current_time}s', 
                         fontsize=14, weight='bold')
        
        # 통계 정보 표시
        active_drones = len([d for d in self.drones if d["status"] != "idle"])
        stats_text = f'Completed: {len(self.completed_deliveries)} | Total: {len(self.delivery_requests)} | Active Drones: {active_drones}'
        self.stats_text = self.ax.text(0.02, 0.98, stats_text, 
                                     transform=self.ax.transAxes, 
                                     verticalalignment='top', fontsize=10, weight='bold',
                                     bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.9))
        
        # 동적 요청 위치 표시
        for request in self.dynamic_requests:
            if request['status'] == 'pending':
                customer_lat = request['customer_location']['latitude']
                customer_lon = request['customer_location']['longitude']
                self.ax.scatter(customer_lon, customer_lat, c='red', s=120, marker='*', 
                              alpha=0.8, edgecolors='black', linewidth=2)
                self.ax.text(customer_lon, customer_lat, 'NEW', fontsize=8, ha='center', va='bottom', 
                           weight='bold', color='red')
        
        # 캔버스 업데이트 강제 실행
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    
    def _get_drone_color(self, drone):
        """
        드론 상태에 따른 색상 반환
        """
        if drone['status'] == 'flying':
            return 'darkblue'  # 더 진한 파란색
        elif drone['status'] == 'pickup':
            return 'darkorange'  # 더 진한 주황색
        elif drone['status'] == 'delivery':
            return 'darkgreen'  # 더 진한 초록색
        else:
            return 'gray'
    
    def _get_simulation_results(self):
        """
        시뮬레이션 결과 반환 (동적 요청 통계 포함)
        """
        total_requests = len(self.delivery_requests)
        dynamic_requests = len(self.dynamic_requests)
        completed = len(self.completed_deliveries)
        success_rate = completed / total_requests if total_requests > 0 else 0
        
        avg_delivery_time = self._calculate_avg_delivery_time()
        drone_utilization = self._calculate_drone_utilization()
        
        # 동적 요청 처리율 계산
        dynamic_completed = len([d for d in self.completed_deliveries if d.get('is_dynamic', False)])
        dynamic_request_rate = dynamic_completed / dynamic_requests if dynamic_requests > 0 else 0
        
        return {
            'total_time': self.current_time,
            'total_requests': total_requests,
            'dynamic_requests': dynamic_requests,
            'completed_deliveries': completed,
            'success_rate': success_rate,
            'avg_delivery_time': avg_delivery_time,
            'drone_utilization': drone_utilization,
            'dynamic_request_rate': dynamic_request_rate
        }
    
    def _calculate_avg_delivery_time(self):
        """
        평균 배달 시간 계산
        """
        if not self.completed_deliveries:
            return 0
        
        # 간단한 추정 (실제로는 각 배달의 시작/완료 시간을 추적해야 함)
        return self.current_time / len(self.completed_deliveries)
    
    def _calculate_drone_utilization(self):
        """
        드론 활용률 계산
        """
        if not self.drones:
            return 0
        
        idle_drones = sum(1 for drone in self.drones if drone['status'] == 'idle')
        return (len(self.drones) - idle_drones) / len(self.drones) 