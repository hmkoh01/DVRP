"""
건물 데이터 로더
실제 건물 정보를 로드하고 처리하는 유틸리티
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class BuildingDataLoader:
    """
    건물 데이터 로드 및 전처리 클래스
    """
    
    def __init__(self, data_path=None):
        """
        초기화
        """
        self.data_path = data_path
        self.buildings = None
        self.restaurants = None
        self.residential_buildings = None
        
    def load_building_data(self, data_path=None):
        """
        건물 데이터 로드
        """
        if data_path:
            self.data_path = data_path
            
        if not self.data_path:
            raise ValueError("데이터 경로가 설정되지 않았습니다.")
            
        try:
            self.buildings = pd.read_csv(self.data_path)
            print(f"건물 데이터 로드 완료: {len(self.buildings)}개 건물")
            
            # 데이터 검증
            self._validate_data()
            
            # 건물 분류
            self._classify_buildings()
            
            return self.buildings
            
        except Exception as e:
            print(f"데이터 로드 실패: {e}")
            return None
    
    def _validate_data(self):
        """
        데이터 유효성 검증
        """
        required_columns = ['longitude', 'latitude', 'height_m', '용도', '종류']
        missing_columns = [col for col in required_columns if col not in self.buildings.columns]
        
        if missing_columns:
            raise ValueError(f"필수 컬럼이 누락되었습니다: {missing_columns}")
        
        # 좌표 범위 검증
        lon_range = self.buildings['longitude'].describe()
        lat_range = self.buildings['latitude'].describe()
        
        print(f"좌표 범위:")
        print(f"  경도: {lon_range['min']:.2f} ~ {lon_range['max']:.2f}")
        print(f"  위도: {lat_range['min']:.2f} ~ {lat_range['max']:.2f}")
        
        # 높이 정보 검증
        if 'height_m' in self.buildings.columns:
            height_stats = self.buildings['height_m'].describe()
            print(f"높이 통계:")
            print(f"  최소: {height_stats['min']:.1f}m")
            print(f"  최대: {height_stats['max']:.1f}m")
            print(f"  평균: {height_stats['mean']:.1f}m")
    
    def _classify_buildings(self):
        """
        건물을 용도별로 분류
        """
        from ..config import DELIVERY_CONFIG
        
        # 식당 건물 분류
        restaurant_types = DELIVERY_CONFIG['restaurant_types']
        self.restaurants = self.buildings[
            self.buildings['용도'].isin(restaurant_types)
        ].copy()
        
        # 주거용 건물 분류
        residential_types = DELIVERY_CONFIG['residential_types']
        self.residential_buildings = self.buildings[
            self.buildings['용도'].isin(residential_types)
        ].copy()
        
        print(f"건물 분류 완료:")
        print(f"  식당: {len(self.restaurants)}개")
        print(f"  주거용: {len(self.residential_buildings)}개")
    
    def get_building_by_id(self, building_id):
        """
        건물 ID로 건물 정보 조회
        """
        if self.buildings is None:
            return None
            
        building = self.buildings[self.buildings['building_id'] == building_id]
        return building.iloc[0] if len(building) > 0 else None
    
    def get_random_restaurant(self):
        """
        무작위 식당 선택
        """
        if self.restaurants is None or len(self.restaurants) == 0:
            return None
            
        return self.restaurants.sample(n=1).iloc[0]
    
    def get_random_residential(self):
        """
        무작위 주거용 건물 선택
        """
        if self.residential_buildings is None or len(self.residential_buildings) == 0:
            return None
            
        return self.residential_buildings.sample(n=1).iloc[0]
    
    def get_buildings_in_area(self, center_lon, center_lat, radius_meters):
        """
        특정 지역 내 건물들 조회
        """
        if self.buildings is None:
            return pd.DataFrame()
        
        # 간단한 유클리드 거리 계산 (미터 단위로 변환)
        # 1도 ≈ 111,000m (대략적)
        buildings_copy = self.buildings.copy()
        buildings_copy['distance'] = np.sqrt(
            ((buildings_copy['longitude'] - center_lon) * 111000) ** 2 +
            ((buildings_copy['latitude'] - center_lat) * 111000) ** 2
        )
        
        return buildings_copy[buildings_copy['distance'] <= radius_meters]
    
    def get_buildings_by_height_range(self, min_height, max_height):
        """
        높이 범위별 건물 조회
        """
        if self.buildings is None:
            return pd.DataFrame()
            
        return self.buildings[
            (self.buildings['height_m'] >= min_height) & 
            (self.buildings['height_m'] <= max_height)
        ]
    
    def get_buildings_by_type(self, building_type):
        """
        건물 종류별 조회
        """
        if self.buildings is None:
            return pd.DataFrame()
            
        return self.buildings[self.buildings['종류'] == building_type]
    
    def get_buildings_by_usage(self, usage_type):
        """
        건물 용도별 조회
        """
        if self.buildings is None:
            return pd.DataFrame()
            
        return self.buildings[self.buildings['용도'] == usage_type]
    
    def calculate_3d_distance(self, building1, building2):
        """
        두 건물 간 3D 거리 계산
        """
        # 2D 거리 (미터 단위)
        lon1, lat1 = building1['longitude'], building1['latitude']
        lon2, lat2 = building2['longitude'], building2['latitude']
        
        # 대략적인 미터 단위 변환
        dx = (lon2 - lon1) * 111000  # 경도 차이를 미터로
        dy = (lat2 - lat1) * 111000  # 위도 차이를 미터로
        
        # 높이 차이
        dz = building2['height_m'] - building1['height_m']
        
        # 3D 유클리드 거리
        distance_3d = np.sqrt(dx**2 + dy**2 + dz**2)
        
        return distance_3d
    
    def get_building_statistics(self):
        """
        건물 통계 정보 반환
        """
        if self.buildings is None:
            return {}
        
        stats = {
            'total_buildings': len(self.buildings),
            'restaurants': len(self.restaurants) if self.restaurants is not None else 0,
            'residential': len(self.residential_buildings) if self.residential_buildings is not None else 0,
            'height_stats': self.buildings['height_m'].describe().to_dict(),
            'usage_distribution': self.buildings['용도'].value_counts().to_dict(),
            'type_distribution': self.buildings['종류'].value_counts().to_dict()
        }
        
        return stats
    
    def save_processed_data(self, output_path):
        """
        전처리된 데이터 저장
        """
        if self.buildings is None:
            print("저장할 데이터가 없습니다.")
            return
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 전체 건물 데이터
        self.buildings.to_csv(output_path / 'processed_buildings.csv', index=False)
        
        # 식당 데이터
        if self.restaurants is not None:
            self.restaurants.to_csv(output_path / 'restaurants.csv', index=False)
        
        # 주거용 건물 데이터
        if self.residential_buildings is not None:
            self.residential_buildings.to_csv(output_path / 'residential_buildings.csv', index=False)
        
        print(f"전처리된 데이터 저장 완료: {output_path}") 