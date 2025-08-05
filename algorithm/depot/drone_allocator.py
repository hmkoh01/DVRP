"""
드론 할당 및 최적화 모듈 (의존성 제거 버전)
"""

import numpy as np
import pandas as pd

class DroneAllocator:
    """
    드론 할당 및 최적화 클래스
    """
    
    def __init__(self, depots, building_data):
        """
        초기화
        """
        self.depots = depots
        self.building_data = building_data
        self.drone_assignments = {}
        
    def calculate_optimal_drone_count(self, method='workload_based'):
        """
        최적 드론 수 계산
        """
        print("=== 최적 드론 수 계산 ===")
        
        if method == 'workload_based':
            return self._calculate_workload_based()
        elif method == 'coverage_based':
            return self._calculate_coverage_based()
        elif method == 'capacity_based':
            return self._calculate_capacity_based()
        else:
            print(f"지원하지 않는 방법: {method}")
            return None
    
    def _calculate_workload_based(self):
        """
        작업량 기반 드론 수 계산
        """
        total_buildings = len(self.building_data)
        total_area = self._calculate_total_area()
        
        # 예상 배달 요청 수 (건물 수와 면적 기반)
        estimated_requests = total_buildings * 0.1  # 건물당 10% 배달 요청
        area_factor = total_area * 0.5  # 면적당 0.5 요청/km²
        
        total_estimated_requests = estimated_requests + area_factor
        
        # 드론당 처리 가능한 요청 수 (시간당)
        requests_per_drone_per_hour = 4  # 드론당 시간당 4개 요청 처리
        
        # 필요한 드론 수 계산
        required_drones = int(np.ceil(total_estimated_requests / requests_per_drone_per_hour))
        
        # 최소/최대 제한
        min_drones = max(1, len(self.depots))
        max_drones = min(50, total_buildings // 10)
        
        optimal_drones = np.clip(required_drones, min_drones, max_drones)
        
        print(f"예상 배달 요청: {total_estimated_requests:.1f}/시간")
        print(f"필요한 드론 수: {optimal_drones}")
        
        return optimal_drones
    
    def _calculate_coverage_based(self):
        """
        커버리지 기반 드론 수 계산
        """
        total_area = self._calculate_total_area()
        
        # 드론당 커버 가능한 면적 (반경 2km 기준)
        drone_coverage_area = np.pi * 2**2  # 약 12.57 km²
        
        # 필요한 드론 수
        required_drones = int(np.ceil(total_area / drone_coverage_area))
        
        # Depot 수 고려
        min_drones = len(self.depots)
        max_drones = required_drones * 2
        
        optimal_drones = np.clip(required_drones, min_drones, max_drones)
        
        print(f"총 면적: {total_area:.2f} km²")
        print(f"드론당 커버 면적: {drone_coverage_area:.2f} km²")
        print(f"필요한 드론 수: {optimal_drones}")
        
        return optimal_drones
    
    def _calculate_capacity_based(self):
        """
        용량 기반 드론 수 계산
        """
        # 각 depot의 용량 계산
        depot_capacities = []
        
        for depot in self.depots:
            # depot 주변 건물 수 기반 용량 계산
            nearby_buildings = self._get_nearby_buildings(
                depot['latitude'], depot['longitude'], radius=0.003
            )
            
            # 건물 수에 따른 용량 계산
            capacity = max(1, len(nearby_buildings) // 5)  # 5개 건물당 1개 드론
            depot_capacities.append(capacity)
        
        total_capacity = sum(depot_capacities)
        
        # 최소/최대 제한
        min_drones = max(1, len(self.depots))
        max_drones = min(50, total_capacity * 2)
        
        optimal_drones = np.clip(total_capacity, min_drones, max_drones)
        
        print(f"총 용량: {total_capacity}")
        print(f"최적 드론 수: {optimal_drones}")
        
        return optimal_drones
    
    def _calculate_total_area(self):
        """
        총 면적 계산
        """
        if len(self.building_data) == 0:
            return 0.0
        
        lats = self.building_data['latitude']
        lons = self.building_data['longitude']
        
        lat_range = (lats.max() - lats.min()) * 111
        lon_range = (lons.max() - lons.min()) * 111 * np.cos(np.radians(lats.mean()))
        
        return lat_range * lon_range
    
    def _get_nearby_buildings(self, lat, lon, radius=0.003):
        """
        주변 건물 조회
        """
        distances = np.sqrt(
            (self.building_data['latitude'] - lat)**2 +
            (self.building_data['longitude'] - lon)**2
        )
        return self.building_data[distances <= radius]
    
    def allocate_drones_to_depots(self, total_drones, method='balanced'):
        """
        드론을 depot에 할당
        """
        print(f"=== {total_drones}개 드론을 {len(self.depots)}개 depot에 할당 ===")
        
        if method == 'balanced':
            return self._allocate_balanced(total_drones)
        elif method == 'workload_based':
            return self._allocate_workload_based(total_drones)
        elif method == 'optimal':
            return self._allocate_simple_optimal(total_drones)
        else:
            print(f"지원하지 않는 방법: {method}")
            return None
    
    def _allocate_balanced(self, total_drones):
        """
        균등 할당
        """
        allocation = {}
        base_drones = total_drones // len(self.depots)
        remaining = total_drones % len(self.depots)
        
        for i, depot in enumerate(self.depots):
            depot_drones = base_drones + (1 if i < remaining else 0)
            allocation[depot['id']] = depot_drones
        
        print(f"드론 할당: {allocation}")
        return allocation
    
    def _allocate_workload_based(self, total_drones):
        """
        작업량 기반 할당
        """
        # 각 depot의 예상 작업량 계산
        workloads = []
        for depot in self.depots:
            nearby_buildings = self._get_nearby_buildings(
                depot['latitude'], depot['longitude'], radius=0.003
            )
            workloads.append(len(nearby_buildings))
        
        # 작업량 비율에 따른 할당
        total_workload = sum(workloads)
        if total_workload == 0:
            return self._allocate_balanced(total_drones)
        
        allocation = {}
        for i, depot in enumerate(self.depots):
            ratio = workloads[i] / total_workload
            depot_drones = max(1, int(total_drones * ratio))
            allocation[depot['id']] = depot_drones
        
        # 남은 드론 균등 분배
        allocated = sum(allocation.values())
        remaining = total_drones - allocated
        
        if remaining > 0:
            for i in range(remaining):
                depot_id = list(allocation.keys())[i % len(allocation)]
                allocation[depot_id] += 1
        
        return allocation
    
    def _allocate_simple_optimal(self, total_drones):
        """
        간단한 최적 할당 (scipy 없이)
        """
        # 각 depot의 효율성 계산
        efficiencies = []
        for depot in self.depots:
            nearby_buildings = self._get_nearby_buildings(
                depot['latitude'], depot['longitude'], radius=0.003
            )
            
            # 효율성 = 건물 수 / depot 간 평균 거리
            if len(self.depots) > 1:
                distances = []
                for other_depot in self.depots:
                    if other_depot['id'] != depot['id']:
                        dist = np.sqrt(
                            (depot['latitude'] - other_depot['latitude'])**2 +
                            (depot['longitude'] - other_depot['longitude'])**2
                        )
                        distances.append(dist)
                avg_distance = np.mean(distances)
                efficiency = len(nearby_buildings) / (avg_distance + 0.001)
            else:
                efficiency = len(nearby_buildings)
            
            efficiencies.append(efficiency)
        
        # 효율성 기반 할당
        total_efficiency = sum(efficiencies)
        if total_efficiency == 0:
            return self._allocate_balanced(total_drones)
        
        allocation = {}
        for i, depot in enumerate(self.depots):
            ratio = efficiencies[i] / total_efficiency
            depot_drones = max(1, int(total_drones * ratio))
            allocation[depot['id']] = depot_drones
        
        # 남은 드론 분배
        allocated = sum(allocation.values())
        remaining = total_drones - allocated
        
        if remaining > 0:
            for i in range(remaining):
                depot_id = list(allocation.keys())[i % len(allocation)]
                allocation[depot_id] += 1
        
        return allocation
    
    def get_allocation_statistics(self, allocation):
        """
        할당 통계 반환
        """
        if not allocation:
            return {}
        
        depot_drones = list(allocation.values())
        
        return {
            'total_drones': sum(depot_drones),
            'avg_drones_per_depot': np.mean(depot_drones),
            'std_drones_per_depot': np.std(depot_drones),
            'min_drones_per_depot': min(depot_drones),
            'max_drones_per_depot': max(depot_drones)
        }