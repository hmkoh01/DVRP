"""
3D 경로 최적화 및 장애물 회피 알고리즘
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

class PathOptimizer3D:
    """
    3D 경로 최적화 및 장애물 회피 클래스
    """
    
    def __init__(self, building_data, safety_height=20.0, min_flight_height=30.0):
        """
        초기화
        """
        self.building_data = building_data
        self.safety_height = safety_height
        self.min_flight_height = min_flight_height
        
        # 건물 높이 정보 처리
        self._process_building_heights()
        
        # 공간 인덱스 생성
        self._build_spatial_index()
    
    def _process_building_heights(self):
        """
        건물 높이 정보 처리
        """
        if 'height_m' in self.building_data.columns:
            self.building_heights = self.building_data['height_m'].fillna(3.0)
            self.max_building_height = self.building_heights.max()
        else:
            self.building_heights = np.full(len(self.building_data), 3.0)
            self.max_building_height = 3.0
        
        # 안전 비행 고도 설정
        self.safe_flight_height = max(self.min_flight_height, self.max_building_height + self.safety_height)
        
        print(f"3D 경로 최적화 초기화:")
        print(f"  - 최대 건물 높이: {self.max_building_height:.1f}m")
        print(f"  - 안전 비행 고도: {self.safe_flight_height:.1f}m")
    
    def _build_spatial_index(self):
        """
        공간 인덱스 구축 (빠른 장애물 검색용)
        """
        self.building_positions = np.column_stack([
            self.building_data['longitude'].values,
            self.building_data['latitude'].values,
            self.building_heights.values
        ])
        
        print(f"공간 인덱스 구축 완료: {len(self.building_positions)}개 건물")
    
    def optimize_3d_route(self, start_point, end_point, intermediate_points=None):
        """
        3D 경로 최적화 (장애물 회피 포함)
        """
        # 모든 경유점 수집
        all_points = [start_point]
        if intermediate_points:
            all_points.extend(intermediate_points)
        all_points.append(end_point)
        
        # 3D 경로 생성
        optimized_path = []
        
        for i in range(len(all_points) - 1):
            current_point = all_points[i]
            next_point = all_points[i + 1]
            
            # 두 지점 간의 3D 경로 생성
            segment_path = self._generate_3d_segment_path(current_point, next_point)
            optimized_path.extend(segment_path)
        
        return optimized_path
    
    def _generate_3d_segment_path(self, start_point, end_point):
        """
        두 지점 간의 3D 세그먼트 경로 생성
        """
        path_3d = []
        
        # 시작점 추가
        path_3d.append(start_point)
        
        # 1단계: 상승 (안전 고도까지)
        if start_point[2] < self.safe_flight_height:
            climb_point = (start_point[0], start_point[1], self.safe_flight_height)
            path_3d.append(climb_point)
        
        # 2단계: 수평 이동 (장애물 회피)
        horizontal_path = self._calculate_obstacle_avoidance_path(
            (start_point[0], start_point[1]),
            (end_point[0], end_point[1]),
            self.safe_flight_height
        )
        
        for point_2d in horizontal_path:
            path_3d.append((point_2d[0], point_2d[1], self.safe_flight_height))
        
        # 3단계: 하강 (목표 지점까지)
        if end_point[2] < self.safe_flight_height:
            descent_point = (end_point[0], end_point[1], self.safe_flight_height)
            path_3d.append(descent_point)
        
        # 끝점 추가
        path_3d.append(end_point)
        
        return path_3d
    
    def _calculate_obstacle_avoidance_path(self, start_2d, end_2d, flight_height):
        """
        장애물 회피 경로 계산
        """
        # 직선 경로상의 장애물 확인
        obstacles = self._find_obstacles_in_path(start_2d, end_2d, flight_height)
        
        if not obstacles:
            # 장애물이 없으면 직선 경로
            return [start_2d, end_2d]
        
        # 장애물이 있으면 우회 경로 생성
        return self._generate_detour_path(start_2d, end_2d, obstacles, flight_height)
    
    def _find_obstacles_in_path(self, start_2d, end_2d, flight_height):
        """
        경로상의 장애물 찾기
        """
        obstacles = []
        
        # 경로를 여러 점으로 나누어 검사
        num_check_points = 20
        for i in range(num_check_points + 1):
            t = i / num_check_points
            check_lon = start_2d[0] + t * (end_2d[0] - start_2d[0])
            check_lat = start_2d[1] + t * (end_2d[1] - start_2d[1])
            
            # 이 지점 근처의 건물들 확인
            nearby_buildings = self._find_nearby_buildings(check_lon, check_lat, radius=0.001)
            
            for building in nearby_buildings:
                building_height = building['height']
                
                # 건물이 비행 고도보다 높으면 장애물
                if building_height + self.safety_height > flight_height:
                    obstacles.append(building)
        
        return obstacles
    
    def _find_nearby_buildings(self, lon, lat, radius=0.001):
        """
        주변 건물 찾기
        """
        nearby_buildings = []
        
        for idx, building in self.building_data.iterrows():
            building_lon = building['longitude']
            building_lat = building['latitude']
            building_height = self.building_heights[idx]
            
            # 거리 계산
            distance = np.sqrt((lon - building_lon)**2 + (lat - building_lat)**2)
            
            if distance <= radius:
                nearby_buildings.append({
                    'lon': building_lon,
                    'lat': building_lat,
                    'height': building_height,
                    'distance': distance
                })
        
        return nearby_buildings
    
    def _generate_detour_path(self, start_2d, end_2d, obstacles, flight_height):
        """
        우회 경로 생성
        """
        # 장애물의 최대 높이 확인
        max_obstacle_height = max(obs['height'] for obs in obstacles)
        detour_height = max_obstacle_height + self.safety_height
        
        # 우회 경로 생성 (여러 방법)
        detour_path = self._generate_vertical_detour(start_2d, end_2d, detour_height)
        
        return detour_path
    
    def _generate_vertical_detour(self, start_2d, end_2d, detour_height):
        """
        수직 우회 경로 생성 (장애물 위로 올라가기)
        """
        # 중간점 계산
        mid_lon = (start_2d[0] + end_2d[0]) / 2
        mid_lat = (start_2d[1] + end_2d[1]) / 2
        
        # 우회 경로: 시작점 -> 중간점 -> 끝점
        path = [start_2d, (mid_lon, mid_lat), end_2d]
        
        return path
    
    def _generate_lateral_detour(self, start_2d, end_2d, obstacles, flight_height):
        """
        측면 우회 경로 생성 (장애물을 옆으로 피하기)
        """
        # 장애물들의 중심점 계산
        obstacle_center_lon = np.mean([obs['lon'] for obs in obstacles])
        obstacle_center_lat = np.mean([obs['lat'] for obs in obstacles])
        
        # 우회 방향 결정 (장애물 중심에서 경로까지의 수직 방향)
        path_vector = np.array([end_2d[0] - start_2d[0], end_2d[1] - start_2d[1]])
        path_vector = path_vector / np.linalg.norm(path_vector)
        
        # 수직 벡터 (시계방향 90도 회전)
        perpendicular_vector = np.array([-path_vector[1], path_vector[0]])
        
        # 우회 거리 계산
        detour_distance = 0.002  # 200m
        
        # 우회점 계산
        detour_point1 = np.array(start_2d) + perpendicular_vector * detour_distance
        detour_point2 = np.array(end_2d) + perpendicular_vector * detour_distance
        
        # 우회 경로
        path = [
            start_2d,
            (detour_point1[0], detour_point1[1]),
            (detour_point2[0], detour_point2[1]),
            end_2d
        ]
        
        return path
    
    def calculate_3d_distance(self, point1, point2):
        """
        3D 거리 계산
        """
        dx = (point2[0] - point1[0]) * 111000  # 경도 차이를 미터로 변환
        dy = (point2[1] - point1[1]) * 111000  # 위도 차이를 미터로 변환
        dz = point2[2] - point1[2]  # 높이 차이
        
        return np.sqrt(dx**2 + dy**2 + dz**2)
    
    def calculate_path_cost(self, path_3d, speed=15.0):
        """
        3D 경로의 비용 계산
        """
        if len(path_3d) < 2:
            return 0, 0, 0
        
        total_distance = 0
        total_time = 0
        total_cost = 0
        
        for i in range(len(path_3d) - 1):
            distance = self.calculate_3d_distance(path_3d[i], path_3d[i + 1])
            time = distance / speed  # 초 단위
            
            # 비용 계산 (거리당 연료비 + 시간당 운영비)
            fuel_cost = distance * 0.5  # km당 0.5원
            operation_cost = time / 3600 * 100  # 시간당 100원
            
            total_distance += distance
            total_time += time
            total_cost += fuel_cost + operation_cost
        
        return total_cost, total_time, total_distance
    
    def optimize_delivery_route_3d(self, depot, delivery_requests):
        """
        배달 요청들을 위한 3D 경로 최적화
        """
        if not delivery_requests:
            return []
        
        # 모든 배달 지점 수집
        all_points = []
        
        # Depot에서 시작
        depot_point = (depot['longitude'], depot['latitude'], 0)
        all_points.append(depot_point)
        
        # 각 배달 요청의 픽업/배달 지점
        for request in delivery_requests:
            # 픽업 지점
            pickup_lon = request.get('pickup_lon', request.get('restaurant_location', {}).get('longitude', 0))
            pickup_lat = request.get('pickup_lat', request.get('restaurant_location', {}).get('latitude', 0))
            pickup_point = (pickup_lon, pickup_lat, 0)
            all_points.append(pickup_point)
            
            # 배달 지점
            delivery_lon = request.get('delivery_lon', request.get('customer_location', {}).get('longitude', 0))
            delivery_lat = request.get('delivery_lat', request.get('customer_location', {}).get('latitude', 0))
            delivery_point = (delivery_lon, delivery_lat, 0)
            all_points.append(delivery_point)
        
        # Depot로 복귀
        all_points.append(depot_point)
        
        # 3D 경로 최적화
        optimized_path = self.optimize_3d_route(all_points[0], all_points[-1], all_points[1:-1])
        
        # 경로 비용 계산
        total_cost, total_time, total_distance = self.calculate_path_cost(optimized_path)
        
        return {
            'path_3d': optimized_path,
            'total_cost': total_cost,
            'total_time': total_time,
            'total_distance': total_distance,
            'requests': delivery_requests
        }
    
    def generate_safety_zones(self):
        """
        안전 비행 구역 생성
        """
        safety_zones = []
        
        for idx, building in self.building_data.iterrows():
            building_height = self.building_heights[idx]
            
            if building_height > 20:  # 높은 건물만 고려
                safety_zone = {
                    'center_lon': building['longitude'],
                    'center_lat': building['latitude'],
                    'radius': 0.001,  # 100m 반경
                    'min_height': building_height + self.safety_height,
                    'max_height': self.safe_flight_height + 50
                }
                safety_zones.append(safety_zone)
        
        return safety_zones
    
    def check_collision_risk(self, path_3d, safety_margin=10.0):
        """
        경로의 충돌 위험도 검사
        """
        collision_risks = []
        
        for i in range(len(path_3d) - 1):
            segment_start = path_3d[i]
            segment_end = path_3d[i + 1]
            
            # 세그먼트 상의 장애물 확인
            obstacles = self._find_obstacles_in_segment(segment_start, segment_end, safety_margin)
            
            if obstacles:
                collision_risks.append({
                    'segment_index': i,
                    'start_point': segment_start,
                    'end_point': segment_end,
                    'obstacles': obstacles,
                    'risk_level': 'high' if len(obstacles) > 3 else 'medium'
                })
        
        return collision_risks
    
    def _find_obstacles_in_segment(self, start_3d, end_3d, safety_margin):
        """
        세그먼트 상의 장애물 찾기
        """
        obstacles = []
        
        # 세그먼트를 여러 점으로 나누어 검사
        num_points = 10
        for i in range(num_points + 1):
            t = i / num_points
            check_lon = start_3d[0] + t * (end_3d[0] - start_3d[0])
            check_lat = start_3d[1] + t * (end_3d[1] - start_3d[1])
            check_height = start_3d[2] + t * (end_3d[2] - start_3d[2])
            
            # 이 지점 근처의 건물들 확인
            nearby_buildings = self._find_nearby_buildings(check_lon, check_lat, radius=0.001)
            
            for building in nearby_buildings:
                building_height = building['height']
                
                # 건물이 비행 경로와 충돌할 위험이 있는지 확인
                if (building_height + safety_margin > check_height and 
                    abs(building['lon'] - check_lon) < 0.0005 and 
                    abs(building['lat'] - check_lat) < 0.0005):
                    obstacles.append(building)
        
        return obstacles 