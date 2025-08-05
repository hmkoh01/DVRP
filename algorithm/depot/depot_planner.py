"""
Depot 위치 최적화 및 계획 모듈 (의존성 제거 버전)
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

class DepotPlanner:
    """
    Depot 위치 최적화 및 계획 클래스
    """
    
    def __init__(self, building_data):
        """
        초기화
        """
        self.building_data = building_data
        self.depots = []
        self.optimal_k = None
        self.depot_coverage = {}
        
    def analyze_map_characteristics(self):
        """
        지도 특성 분석
        """
        print("=== 지도 특성 분석 ===")
        
        # 기본 통계
        total_buildings = len(self.building_data)
        area_coverage = self._calculate_area_coverage()
        
        # 배달 요청 예상 지역 분석
        delivery_hotspots = self._identify_delivery_hotspots()
        
        # 건물 밀도 분석
        building_density = self._analyze_building_density()
        
        # 지형 특성 분석
        terrain_characteristics = self._analyze_terrain()
        
        characteristics = {
            'total_buildings': total_buildings,
            'area_coverage': area_coverage,
            'delivery_hotspots': delivery_hotspots,
            'building_density': building_density,
            'terrain_characteristics': terrain_characteristics
        }
        
        print(f"총 건물 수: {total_buildings}")
        print(f"면적 커버리지: {area_coverage:.2f} km²")
        print(f"배달 핫스팟 수: {len(delivery_hotspots)}")
        print(f"건물 밀도: {building_density:.2f} 건물/km²")
        
        return characteristics
    
    def _calculate_area_coverage(self):
        """
        지도 커버리지 면적 계산
        """
        if len(self.building_data) == 0:
            return 0.0
            
        lats = self.building_data['latitude']
        lons = self.building_data['longitude']
        
        # 대략적인 면적 계산 (위도/경도 차이를 km로 변환)
        lat_range = (lats.max() - lats.min()) * 111  # 1도 위도 ≈ 111km
        lon_range = (lons.max() - lons.min()) * 111 * np.cos(np.radians(lats.mean()))
        
        return lat_range * lon_range
    
    def _identify_delivery_hotspots(self):
        """
        배달 요청이 많이 발생할 것으로 예상되는 지역 식별
        """
        hotspots = []
        
        # 식당, 상업시설 등 배달 출발점
        restaurant_types = ['근린생활시설', '기타시설', '상업시설']
        restaurants = self.building_data[
            self.building_data['용도'].isin(restaurant_types)
        ]
        
        # 주거지역 등 배달 도착점
        residential_types = ['주택', '근린생활시설']
        residential = self.building_data[
            self.building_data['용도'].isin(residential_types)
        ]
        
        # 핫스팟으로 추가 (식당은 모두, 주거지는 샘플링)
        for _, building in restaurants.iterrows():
            hotspots.append({
                'type': 'restaurant',
                'latitude': building['latitude'],
                'longitude': building['longitude'],
                'weight': 2.0  # 식당은 높은 가중치
            })
        
        # 주거지역은 샘플링하여 추가
        residential_sample = residential.sample(n=min(len(residential), 50))
        for _, building in residential_sample.iterrows():
            hotspots.append({
                'type': 'residential',
                'latitude': building['latitude'],
                'longitude': building['longitude'],
                'weight': 1.0  # 주거지역은 낮은 가중치
            })
        
        return hotspots
    
    def _analyze_building_density(self):
        """
        건물 밀도 분석
        """
        area = self._calculate_area_coverage()
        if area == 0:
            return 0.0
        return len(self.building_data) / area
    
    def _analyze_terrain(self):
        """
        지형 특성 분석
        """
        if 'height' not in self.building_data.columns:
            return {'avg_height': 0, 'height_variance': 0}
        
        heights = self.building_data['height'].dropna()
        return {
            'avg_height': heights.mean(),
            'height_variance': heights.var(),
            'max_height': heights.max(),
            'min_height': heights.min()
        }
    
    def optimize_depot_locations(self, method='auto'):
        """
        Depot 위치 최적화
        """
        print("=== Depot 위치 최적화 ===")
        
        # 지도 특성 분석
        characteristics = self.analyze_map_characteristics()
        
        # 자동으로 최적 방법 선택
        if method == 'auto':
            method = self._select_optimal_method(characteristics)
        
        print(f"선택된 최적화 방법: {method}")
        
        if method == 'kmeans':
            return self._optimize_with_simple_clustering(characteristics)
        elif method == 'density_based':
            return self._optimize_with_density_based(characteristics)
        elif method == 'grid_based':
            return self._optimize_with_grid_based(characteristics)
        else:
            print(f"지원하지 않는 방법: {method}")
            return None
    
    def _select_optimal_method(self, characteristics):
        """
        지도 특성에 따른 최적 방법 선택
        """
        building_density = characteristics['building_density']
        area_coverage = characteristics['area_coverage']
        
        if building_density > 100:  # 고밀도 지역
            return 'density_based'
        elif area_coverage > 10:  # 넓은 지역
            return 'grid_based'
        else:
            return 'kmeans'
    
    def _optimize_with_simple_clustering(self, characteristics):
        """
        간단한 클러스터링을 사용한 Depot 최적화 (sklearn 없이)
        """
        hotspots = characteristics['delivery_hotspots']
        
        if len(hotspots) == 0:
            print("배달 핫스팟이 없습니다.")
            return None
        
        # 간단한 그리드 기반 클러스터링
        coordinates = np.array([[h['latitude'], h['longitude']] for h in hotspots])
        
        # 좌표 범위 계산
        lat_min, lat_max = coordinates[:, 0].min(), coordinates[:, 0].max()
        lon_min, lon_max = coordinates[:, 1].min(), coordinates[:, 1].max()
        
        # 그리드 크기 결정 (핫스팟 수에 따라)
        grid_size = min(5, max(2, int(np.sqrt(len(hotspots) / 10))))
        
        # 그리드 생성
        lat_grid = np.linspace(lat_min, lat_max, grid_size + 1)
        lon_grid = np.linspace(lon_min, lon_max, grid_size + 1)
        
        self.depots = []
        depot_id = 1
        
        for i in range(grid_size):
            for j in range(grid_size):
                center_lat = (lat_grid[i] + lat_grid[i+1]) / 2
                center_lon = (lon_grid[j] + lon_grid[j+1]) / 2
                
                # 해당 그리드 내 핫스팟 수 계산
                hotspots_in_grid = [
                    h for h in hotspots
                    if (lat_grid[i] <= h['latitude'] <= lat_grid[i+1] and
                        lon_grid[j] <= h['longitude'] <= lon_grid[j+1])
                ]
                
                if len(hotspots_in_grid) > 0:
                    depot = {
                        'id': f'depot_{depot_id}',
                        'latitude': center_lat,
                        'longitude': center_lon,
                        'cluster_size': len(hotspots_in_grid),
                        'coverage_radius': 0.002  # 약 200m 반경
                    }
                    self.depots.append(depot)
                    depot_id += 1
        
        self.optimal_k = len(self.depots)
        return self.depots
    
    def _optimize_with_density_based(self, characteristics):
        """
        밀도 기반 Depot 최적화 (간단한 버전)
        """
        hotspots = characteristics['delivery_hotspots']
        
        if len(hotspots) == 0:
            return None
        
        # 간단한 밀도 기반 클러스터링
        coordinates = np.array([[h['latitude'], h['longitude']] for h in hotspots])
        
        # 좌표를 정렬하여 밀도 높은 지역 찾기
        sorted_indices = np.argsort(coordinates[:, 0])  # 위도 기준 정렬
        
        # 밀도 높은 지역을 중심으로 depot 배치
        self.depots = []
        depot_id = 1
        
        # 간격을 두고 depot 배치
        step = max(1, len(sorted_indices) // 5)  # 최대 5개 depot
        
        for i in range(0, len(sorted_indices), step):
            if depot_id > 5:  # 최대 5개 depot
                break
                
            idx = sorted_indices[i]
            center_lat = coordinates[idx, 0]
            center_lon = coordinates[idx, 1]
            
            depot = {
                'id': f'depot_{depot_id}',
                'latitude': center_lat,
                'longitude': center_lon,
                'cluster_size': step,
                'coverage_radius': 0.003  # 약 300m 반경
            }
            self.depots.append(depot)
            depot_id += 1
        
        self.optimal_k = len(self.depots)
        return self.depots
    
    def _optimize_with_grid_based(self, characteristics):
        """
        그리드 기반 Depot 최적화
        """
        area_coverage = characteristics['area_coverage']
        
        # 면적에 따른 그리드 크기 결정
        grid_size = max(2, min(5, int(np.sqrt(area_coverage / 4))))  # 4km²당 1개 depot
        
        # 건물 데이터의 경계
        lats = self.building_data['latitude']
        lons = self.building_data['longitude']
        
        lat_min, lat_max = lats.min(), lats.max()
        lon_min, lon_max = lons.min(), lons.max()
        
        # 그리드 생성
        lat_grid = np.linspace(lat_min, lat_max, grid_size + 1)
        lon_grid = np.linspace(lon_min, lon_max, grid_size + 1)
        
        self.depots = []
        depot_id = 1
        
        for i in range(grid_size):
            for j in range(grid_size):
                center_lat = (lat_grid[i] + lat_grid[i+1]) / 2
                center_lon = (lon_grid[j] + lon_grid[j+1]) / 2
                
                # 해당 그리드 내 건물 수 계산
                buildings_in_grid = self.building_data[
                    (self.building_data['latitude'] >= lat_grid[i]) &
                    (self.building_data['latitude'] <= lat_grid[i+1]) &
                    (self.building_data['longitude'] >= lon_grid[j]) &
                    (self.building_data['longitude'] <= lon_grid[j+1])
                ]
                
                if len(buildings_in_grid) > 0:
                    depot = {
                        'id': f'depot_{depot_id}',
                        'latitude': center_lat,
                        'longitude': center_lon,
                        'cluster_size': len(buildings_in_grid),
                        'coverage_radius': 0.002  # 약 200m 반경
                    }
                    self.depots.append(depot)
                    depot_id += 1
        
        self.optimal_k = len(self.depots)
        return self.depots
    
    def _calculate_coverage_radius(self, center, points):
        """
        Depot의 커버리지 반경 계산
        """
        if len(points) == 0:
            return 0.001  # 기본 100m
        
        distances = np.sqrt(np.sum((points - center) ** 2, axis=1))
        return np.percentile(distances, 90)  # 90% 포인트까지 커버
    
    def get_depot_statistics(self):
        """
        Depot 통계 반환
        """
        if not self.depots:
            return {}
        
        total_coverage = sum(d['cluster_size'] for d in self.depots)
        avg_coverage_radius = np.mean([d['coverage_radius'] for d in self.depots])
        
        return {
            'total_depots': len(self.depots),
            'total_coverage': total_coverage,
            'avg_coverage_radius': avg_coverage_radius,
            'optimal_k': self.optimal_k
        }