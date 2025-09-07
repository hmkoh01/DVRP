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
        required_columns = ['longitude', 'latitude', 'HEIGHT', 'USABILITY']
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
        if 'HEIGHT' in self.buildings.columns:
            height_stats = self.buildings['HEIGHT'].describe()
            print(f"높이 통계:")
            print(f"  최소: {height_stats['min']:.1f}m")
            print(f"  최대: {height_stats['max']:.1f}m")
            print(f"  평균: {height_stats['mean']:.1f}m")
        
        # 절대 높이 정보 검증
        if 'ABSOLUTE_HEIGHT' in self.buildings.columns:
            abs_height_stats = self.buildings['ABSOLUTE_HEIGHT'].describe()
            print(f"절대 높이 통계:")
            print(f"  최소: {abs_height_stats['min']:.1f}m")
            print(f"  최대: {abs_height_stats['max']:.1f}m")
            print(f"  평균: {abs_height_stats['mean']:.1f}m")
    
    def _classify_buildings(self):
        """
        건물을 용도별로 분류 (market_info_pohang.csv의 식당 정보 활용)
        """
        try:
            from ..config import DELIVERY_CONFIG
        except ImportError:
            # 상대 import 실패 시 절대 import 시도
            import sys
            import os
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from config import DELIVERY_CONFIG
        
        # 식당 정보 로드
        restaurant_coordinates = self._load_restaurant_coordinates()
        
        # 식당 건물 분류 (좌표 매칭)
        self.restaurants = self._match_restaurants_by_coordinates(restaurant_coordinates)
        
        # 주거용 건물 분류
        residential_types = ['공동주택']
        self.residential_buildings = self.buildings[
            self.buildings['USABILITY'].isin(residential_types)
        ].copy()
        
        print(f"건물 분류 완료:")
        print(f"  식당(좌표 매칭): {len(self.restaurants)}개")
        print(f"  주거용(공동주택): {len(self.residential_buildings)}개")
        
        # 용도별 분포 출력
        usage_dist = self.buildings['USABILITY'].value_counts()
        print(f"용도별 분포:")
        for usage, count in usage_dist.items():
            print(f"  {usage}: {count}개")
        
        # 층별 포인트 생성
        self._generate_floor_based_points()
        
        # # 건물 분류 시각화 자동 실행
        # print(f"\n🗺️ 건물 분류 시각화 시작...")
        # self.visualize_classified_buildings()
    
    def _generate_floor_based_points(self):
        """
        각 건물의 층수에 따라 다른 높이의 포인트들을 생성
        """
        print("🏢 층별 포인트 생성 시작...")
        
        # 층수 정보가 없으면 높이로 추정
        if 'GRND_FLR' not in self.buildings.columns:
            print("⚠️ 층수 정보(GRND_FLR)가 없습니다. 높이로 층수를 추정합니다.")
            # 높이를 3m로 나누어 층수 추정
            self.buildings['GRND_FLR'] = (self.buildings['HEIGHT'] / 3).round().astype(int)
            self.buildings['GRND_FLR'] = self.buildings['GRND_FLR'].clip(lower=1)  # 최소 1층
        
        # 층별 포인트 생성
        floor_points = []
        building_id_counter = 0
        
        for idx, building in self.buildings.iterrows():
            building_id = building_id_counter
            building_id_counter += 1
            
            # 건물 기본 정보
            base_lon = building['longitude']
            base_lat = building['latitude']
            base_height = building.get('CONT', 0)  # 지면 고도
            building_height = building['HEIGHT']
            floors = building['GRND_FLR']
            
            # 각 층별로 포인트 생성
            for floor_num in range(1, floors + 1):
                # 층별 높이 계산 (지면 + 건물 높이의 비례)
                floor_height = base_height + (building_height * floor_num / floors)
                
                # 층별 포인트 생성
                floor_point = {
                    'building_id': building_id,
                    'floor_number': floor_num,
                    'longitude': base_lon,
                    'latitude': base_lat,
                    'height': floor_height,
                    'absolute_height': floor_height,
                    'building_height': building_height,
                    'total_floors': floors,
                    'usage': building['USABILITY'],
                    'is_restaurant': idx in self.restaurants.index if self.restaurants is not None else False,
                    'is_residential': idx in self.residential_buildings.index if self.residential_buildings is not None else False,
                    'original_building_idx': idx
                }
                
                floor_points.append(floor_point)
        
        # 층별 포인트를 DataFrame으로 변환
        self.floor_points = pd.DataFrame(floor_points)
        
        print(f"층별 포인트 생성 완료:")
        print(f"  원본 건물 수: {len(self.buildings)}개")
        print(f"  생성된 포인트 수: {len(self.floor_points)}개")
        print(f"  평균 층수: {self.floor_points['total_floors'].mean():.1f}층")
        
        # 층수별 통계
        floor_stats = self.floor_points['total_floors'].value_counts().sort_index()
        print(f"  층수별 건물 분포:")
        for floors, count in floor_stats.items():
            print(f"    {floors}층: {count}개")
    
    def get_floor_points(self, usage_type=None, min_floors=None, max_floors=None):
        """
        층별 포인트 반환 (필터링 옵션 포함)
        
        Args:
            usage_type: 건물 용도 필터
            min_floors: 최소 층수
            max_floors: 최대 층수
        """
        if not hasattr(self, 'floor_points'):
            self._generate_floor_based_points()
        
        filtered_points = self.floor_points.copy()
        
        if usage_type:
            filtered_points = filtered_points[filtered_points['usage'] == usage_type]
        
        if min_floors:
            filtered_points = filtered_points[filtered_points['total_floors'] >= min_floors]
        
        if max_floors:
            filtered_points = filtered_points[filtered_points['total_floors'] <= max_floors]
        
        return filtered_points
    
    def get_restaurant_floor_points(self):
        """
        식당 건물의 층별 포인트 반환
        """
        if not hasattr(self, 'floor_points'):
            self._generate_floor_based_points()
        
        return self.floor_points[self.floor_points['is_restaurant'] == True]
    
    def get_residential_floor_points(self):
        """
        주거용 건물의 층별 포인트 반환
        """
        if not hasattr(self, 'floor_points'):
            self._generate_floor_based_points()
        
        return self.floor_points[self.floor_points['is_residential'] == True]
    
    def get_random_restaurant_floor_point(self):
        """
        식당 건물의 층별 포인트를 무작위로 선택
        """
        restaurant_points = self.get_restaurant_floor_points()
        if len(restaurant_points) == 0:
            return None
        
        return restaurant_points.sample(n=1).iloc[0]
    
    def get_random_residential_floor_point(self):
        """
        주거용 건물의 층별 포인트를 무작위로 선택
        """
        residential_points = self.get_residential_floor_points()
        if len(residential_points) == 0:
            return None
        
        return residential_points.sample(n=1).iloc[0]
    
    def get_random_restaurant(self):
        """
        식당 건물을 무작위로 선택 (기존 메서드, 호환성 유지)
        """
        if self.restaurants is None or len(self.restaurants) == 0:
            return None
        
        return self.restaurants.sample(n=1).iloc[0]
    
    def get_random_residential(self):
        """
        주거용 건물을 무작위로 선택 (기존 메서드, 호환성 유지)
        """
        if self.residential_buildings is None or len(self.residential_buildings) == 0:
            return None
        
        return self.residential_buildings.sample(n=1).iloc[0]
    
    def _load_restaurant_coordinates(self):
        """
        market_info_pohang.csv에서 식당 좌표 정보 로드
        """
        try:
            import pandas as pd
            from pathlib import Path
            
            # market_info_pohang.csv 파일 경로
            # 현재 파일 위치에서 상대 경로 계산
            current_dir = Path(__file__).parent  # utils/
            project_root = current_dir.parent.parent  # algorithm/의 상위 (프로젝트 루트)
            market_file = project_root / "map" / "postech_map" / "market_info_pohang.csv"
            
            if not market_file.exists():
                print("⚠️ market_info_pohang.csv 파일을 찾을 수 없습니다. 기본 분류를 사용합니다.")
                return pd.DataFrame()
            
            # 식당 데이터 로드
            market_data = pd.read_csv(market_file, encoding='cp949', low_memory=False)
            restaurants = market_data[market_data['상권업종대분류명'] == '음식'].copy()
            
            # 필요한 컬럼만 선택
            restaurant_coords = restaurants[['상호명', '상권업종중분류명', '경도', '위도']].copy()
            restaurant_coords.columns = ['name', 'category', 'longitude', 'latitude']
            
            print(f"식당 데이터 로드 완료: {len(restaurant_coords)}개")
            return restaurant_coords
            
        except Exception as e:
            print(f"⚠️ 식당 데이터 로드 실패: {e}. 기본 분류를 사용합니다.")
            return pd.DataFrame()
    
    def _match_restaurants_by_coordinates(self, restaurant_coordinates, tolerance=0.01):
        """
        좌표 매칭을 통해 식당 건물 찾기
        """
        if restaurant_coordinates.empty:
            # 기본 분류 사용
            restaurant_types = ['교육연구시설']
            return self.buildings[
                self.buildings['USABILITY'].isin(restaurant_types)
            ].copy()
        
        # 건물 데이터 범위 내의 식당만 필터링
        lon_min, lon_max = self.buildings['longitude'].min(), self.buildings['longitude'].max()
        lat_min, lat_max = self.buildings['latitude'].min(), self.buildings['latitude'].max()
        
        nearby_restaurants = restaurant_coordinates[
            (restaurant_coordinates['longitude'] >= lon_min) & 
            (restaurant_coordinates['longitude'] <= lon_max) & 
            (restaurant_coordinates['latitude'] >= lat_min) & 
            (restaurant_coordinates['latitude'] <= lat_max)
        ]
        
        print(f"건물 데이터 범위 내 식당: {len(nearby_restaurants)}개")
        
        # 좌표 매칭을 위한 빈 DataFrame 생성
        matched_buildings = pd.DataFrame()
        
        # 각 식당에 대해 가장 가까운 건물 찾기 (모든 식당이 매핑되도록)
        for idx, restaurant in nearby_restaurants.iterrows():
            restaurant_lon = restaurant['longitude']
            restaurant_lat = restaurant['latitude']
            
            # 모든 건물과의 거리 계산
            distances = np.sqrt(
                (self.buildings['longitude'] - restaurant_lon)**2 + 
                (self.buildings['latitude'] - restaurant_lat)**2
            )
            
            # 가장 가까운 건물 찾기
            min_distance = distances.min()
            nearest_building_idx = distances.idxmin()
            nearest_building = self.buildings.loc[nearest_building_idx]
            
            print(f"식당 '{restaurant['name']}' → 건물 {nearest_building_idx}와의 거리: {min_distance:.6f}")
            
            if min_distance <= tolerance:
                # 매칭된 건물에 식당 정보 추가
                building_copy = nearest_building.copy()
                building_copy['restaurant_name'] = restaurant['name']
                building_copy['restaurant_category'] = restaurant['category']
                building_copy['distance_to_restaurant'] = min_distance
                building_copy['restaurant_id'] = idx  # 식당 고유 ID
                
                # 건물 하나에 여러 식당이 매핑될 수 있음 (중복 건물 허용)
                matched_buildings = pd.concat([matched_buildings, pd.DataFrame([building_copy])], ignore_index=True)
            else:
                print(f"  ❌ 거리 {min_distance:.6f} > tolerance {tolerance} - 매핑 실패")
        
        print(f"좌표 매칭으로 찾은 식당 건물: {len(matched_buildings)}개")
        
        # 매칭된 건물이 없으면 기본 분류 사용
        if matched_buildings.empty:
            print("⚠️ 좌표 매칭으로 식당을 찾지 못했습니다. 기본 분류를 사용합니다.")
            restaurant_types = ['교육연구시설']
            return self.buildings[
                self.buildings['USABILITY'].isin(restaurant_types)
            ].copy()
        
        return matched_buildings
    
    def get_building_by_id(self, building_id):
        """
        건물 ID로 건물 정보 조회
        """
        if self.buildings is None:
            return None
            
        building = self.buildings[self.buildings['UFID'] == building_id]
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
            (self.buildings['HEIGHT'] >= min_height) & 
            (self.buildings['HEIGHT'] <= max_height)
        ]
    
    def get_buildings_by_type(self, building_type):
        """
        건물 종류별 조회
        """
        if self.buildings is None:
            return pd.DataFrame()
            
        return self.buildings[self.buildings['USABILITY'] == building_type]
    
    def get_buildings_by_usage(self, usage_type):
        """
        건물 용도별 조회
        """
        if self.buildings is None:
            return pd.DataFrame()
            
        return self.buildings[self.buildings['USABILITY'] == usage_type]
    
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
        dz = building2['HEIGHT'] - building1['HEIGHT']
        
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
            'height_stats': self.buildings['HEIGHT'].describe().to_dict(),
            'absolute_height_stats': self.buildings['ABSOLUTE_HEIGHT'].describe().to_dict(),
            'usage_distribution': self.buildings['USABILITY'].value_counts().to_dict()
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
    
    def visualize_classified_buildings(self, output_path="classified_buildings_map.png"):
        """
        분류된 건물들을 지도에 태그를 붙여서 시각화
        """
        try:
            import plotly.graph_objects as go
            import plotly.express as px
            from pathlib import Path
            
            print("건물 분류 시각화 시작...")
            
            # 출력 경로 설정
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # 시각화용 데이터 준비
            viz_data = []
            
            # 일반 건물 (회색)
            for idx, building in self.buildings.iterrows():
                # 이미 식당이나 주거용으로 분류된 건물은 건너뛰기
                if hasattr(self, 'restaurants') and self.restaurants is not None:
                    if not self.restaurants.empty:
                        restaurant_buildings = self.restaurants['UFID'].values
                        if building['UFID'] in restaurant_buildings:
                            continue
                
                if hasattr(self, 'residential_buildings') and self.residential_buildings is not None:
                    if not self.residential_buildings.empty:
                        residential_buildings = self.residential_buildings['UFID'].values
                        if building['UFID'] in residential_buildings:
                            continue
                
                viz_data.append({
                    'longitude': building['longitude'],
                    'latitude': building['latitude'],
                    'height': building['HEIGHT'],
                    'name': building.get('BLD_NM', 'Unknown'),
                    'usage': building['USABILITY'],
                    'type': '일반건물',
                    'color': 'lightgray',
                    'size': 3
                })
            
            # 식당 건물 (빨간색)
            if hasattr(self, 'restaurants') and self.restaurants is not None and not self.restaurants.empty:
                for idx, restaurant in self.restaurants.iterrows():
                    viz_data.append({
                        'longitude': restaurant['longitude'],
                        'latitude': restaurant['latitude'],
                        'height': restaurant['HEIGHT'],
                        'name': restaurant.get('restaurant_name', 'Unknown'),
                        'usage': restaurant.get('restaurant_category', 'Unknown'),
                        'type': '식당',
                        'color': 'red',
                        'size': 8
                    })
            
            # 주거용 건물 (파란색)
            if hasattr(self, 'residential_buildings') and self.residential_buildings is not None and not self.residential_buildings.empty:
                for idx, residential in self.residential_buildings.iterrows():
                    viz_data.append({
                        'longitude': residential['longitude'],
                        'latitude': residential['latitude'],
                        'height': residential['HEIGHT'],
                        'name': residential.get('BLD_NM', 'Unknown'),
                        'usage': residential['USABILITY'],
                        'type': '주거용',
                        'color': 'blue',
                        'size': 6
                    })
            
            # DataFrame으로 변환
            viz_df = pd.DataFrame(viz_data)
            
            if viz_df.empty:
                print("⚠️ 시각화할 데이터가 없습니다.")
                return
            
            # Plotly 지도 생성
            fig = go.Figure()
            
            # 건물 타입별로 마커 추가
            for building_type in viz_df['type'].unique():
                type_data = viz_df[viz_df['type'] == building_type]
                
                # 색상 매핑
                color_map = {
                    '식당': 'red',
                    '주거용': 'blue',
                    '일반건물': 'lightgray'
                }
                
                fig.add_trace(go.Scattermapbox(
                    lon=type_data['longitude'],
                    lat=type_data['latitude'],
                    mode='markers',
                    marker=go.scattermapbox.Marker(
                        size=type_data['size'],
                        color=color_map.get(building_type, 'gray'),
                        opacity=0.8
                    ),
                    text=type_data['name'] + '<br>' + type_data['usage'] + '<br>높이: ' + type_data['height'].astype(str) + 'm',
                    hoverinfo='text',
                    name=building_type,
                    showlegend=True
                ))
            
            # 지도 레이아웃 설정
            fig.update_layout(
                title={
                    'text': f'포스텍 건물 분류 지도 (총 {len(viz_df)}개 건물)',
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 20}
                },
                mapbox=dict(
                    style="open-street-map",
                    center=dict(
                        lat=viz_df['latitude'].mean(),
                        lon=viz_df['longitude'].mean()
                    ),
                    zoom=16
                ),
                width=1200,
                height=800,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01
                )
            )
            
            # PNG 파일로 저장
            fig.write_image(str(output_file), width=1200, height=800, scale=2)
            print(f"✅ 건물 분류 지도 저장 완료: {output_file}")
            
            # 통계 정보 출력
            print(f"\n📊 건물 분류 통계:")
            type_counts = viz_df['type'].value_counts()
            for building_type, count in type_counts.items():
                print(f"  {building_type}: {count}개")
            
            # 식당 상세 정보
            if hasattr(self, 'restaurants') and self.restaurants is not None and not self.restaurants.empty:
                print(f"\n🍽️ 식당 상세 정보:")
                for idx, restaurant in self.restaurants.iterrows():
                    print(f"  {restaurant.get('restaurant_name', 'Unknown')} ({restaurant.get('restaurant_category', 'Unknown')})")
            
        except Exception as e:
            print(f"⚠️ 시각화 실패: {e}")
            print("plotly와 kaleido 라이브러리가 설치되어 있는지 확인해주세요.")
            print("설치 명령어: pip install plotly kaleido")