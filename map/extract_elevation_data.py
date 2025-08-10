import geopandas as gpd
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def extract_elevation_data():
    """
    수치지도에서 지형 고도 정보 추출
    """
    base_path = Path("postech_map/raw_maps")
    
    # 두 수치지도 폴더 경로
    map1_path = base_path / "(B010)수치지도_36914093_2024_00000634112839"
    map2_path = base_path / "(B010)수치지도_36914094_2024_0000056145927"
    
    print("수치지도에서 지형 고도 정보 추출 중...")
    
    all_elevations = []
    
    # 지도 1에서 지형 정보 추출
    try:
        # N3L_F0010000.shp 파일 (가장 많은 지형 정보 포함)
        elevation_gdf1 = gpd.read_file(map1_path / "N3L_F0010000.shp")
        print(f"지도 1에서 {len(elevation_gdf1)}개 지형 요소 로드")
        
        # 컬럼 정보 출력
        print(f"지형 데이터 컬럼: {elevation_gdf1.columns.tolist()}")
        
        # 중심점 계산
        elevation_gdf1['centroid'] = elevation_gdf1.geometry.centroid
        elevation_gdf1['longitude'] = elevation_gdf1['centroid'].x
        elevation_gdf1['latitude'] = elevation_gdf1['centroid'].y
        elevation_gdf1['source'] = 'map1'
        
        # 고도 정보 추출 (실제 컬럼명 확인 후 조정)
        elevation_gdf1['elevation_m'] = 0.0  # 기본값
        
        # 가능한 고도 컬럼들 확인
        possible_elevation_columns = ['등고수치', '고도', 'ELEV', 'HEIGHT', 'ALT', 'Z', '높이']
        for col in possible_elevation_columns:
            if col in elevation_gdf1.columns:
                print(f"고도 정보 컬럼 발견: {col}")
                elevation_gdf1['elevation_m'] = elevation_gdf1[col].fillna(0.0)
                break
        
        # 필요한 컬럼만 선택
        elevation_data1 = elevation_gdf1[['source', 'longitude', 'latitude', 'elevation_m']].copy()
        all_elevations.append(elevation_data1)
        
    except Exception as e:
        print(f"지도 1 지형 처리 실패: {e}")
    
    # 지도 2에서 지형 정보 추출
    try:
        elevation_gdf2 = gpd.read_file(map2_path / "N3L_F0010000.shp")
        print(f"지도 2에서 {len(elevation_gdf2)}개 지형 요소 로드")
        
        # 중심점 계산
        elevation_gdf2['centroid'] = elevation_gdf2.geometry.centroid
        elevation_gdf2['longitude'] = elevation_gdf2['centroid'].x
        elevation_gdf2['latitude'] = elevation_gdf2['centroid'].y
        elevation_gdf2['source'] = 'map2'
        
        # 고도 정보 추출
        elevation_gdf2['elevation_m'] = 0.0  # 기본값
        
        for col in possible_elevation_columns:
            if col in elevation_gdf2.columns:
                elevation_gdf2['elevation_m'] = elevation_gdf2[col].fillna(0.0)
                break
        
        # 필요한 컬럼만 선택
        elevation_data2 = elevation_gdf2[['source', 'longitude', 'latitude', 'elevation_m']].copy()
        all_elevations.append(elevation_data2)
        
    except Exception as e:
        print(f"지도 2 지형 처리 실패: {e}")
    
    if not all_elevations:
        print("추출된 지형 데이터가 없습니다.")
        return None
    
    # 모든 지형 데이터 병합
    merged_elevations = pd.concat(all_elevations, ignore_index=True)
    
    print(f"\n총 추출된 지형 요소 수: {len(merged_elevations)}")
    print(f"출처별 분포:")
    print(merged_elevations['source'].value_counts())
    
    # 고도 통계
    print(f"\n고도 통계:")
    print(merged_elevations['elevation_m'].describe())
    
    # 출력 폴더 생성
    output_path = Path("postech_map/merged_maps")
    output_path.mkdir(exist_ok=True)
    
    # 지형 고도 정보 저장
    elevation_csv = output_path / "postech_elevation_data.csv"
    merged_elevations.to_csv(elevation_csv, index=False, encoding='utf-8')
    print(f"\n지형 고도 정보 저장: {elevation_csv}")
    
    return merged_elevations

def interpolate_elevation_for_buildings(building_data, elevation_data):
    """
    건물 위치에서의 지형 고도를 보간하여 계산
    """
    print("건물 위치에서 지형 고도 보간 중...")
    
    # 건물 데이터에 고도 정보 추가
    building_data['ground_elevation_m'] = 0.0
    
    # 각 건물에 대해 가장 가까운 지형 요소의 고도 사용
    for idx, building in building_data.iterrows():
        building_lon = building['longitude']
        building_lat = building['latitude']
        
        # 모든 지형 요소와의 거리 계산
        distances = np.sqrt(
            (elevation_data['longitude'] - building_lon)**2 + 
            (elevation_data['latitude'] - building_lat)**2
        )
        
        # 가장 가까운 지형 요소 찾기
        nearest_idx = distances.idxmin()
        nearest_elevation = elevation_data.loc[nearest_idx, 'elevation_m']
        
        building_data.loc[idx, 'ground_elevation_m'] = nearest_elevation
    
    # 절대 고도 계산 (지면 고도 + 건물 높이)
    building_data['absolute_height_m'] = building_data['ground_elevation_m'] + building_data['height_m']
    
    print(f"지형 고도 보간 완료:")
    print(f"  - 지면 고도 범위: {building_data['ground_elevation_m'].min():.1f}m ~ {building_data['ground_elevation_m'].max():.1f}m")
    print(f"  - 절대 고도 범위: {building_data['absolute_height_m'].min():.1f}m ~ {building_data['absolute_height_m'].max():.1f}m")
    
    return building_data

if __name__ == "__main__":
    # 지형 고도 정보 추출
    elevation_data = extract_elevation_data()
    
    if elevation_data is not None:
        print("\n지형 고도 정보 추출 완료!")
        
        # 기존 건물 데이터 로드
        from extract_real_buildings import extract_real_building_data
        building_data = extract_real_building_data()
        
        if building_data is not None:
            # 건물 데이터에 지형 고도 정보 추가
            building_data_with_elevation = interpolate_elevation_for_buildings(building_data, elevation_data)
            
            # 지형 고도가 포함된 건물 데이터 저장
            output_path = Path("postech_map/merged_maps")
            elevation_building_csv = output_path / "postech_buildings_with_elevation.csv"
            building_data_with_elevation.to_csv(elevation_building_csv, index=False, encoding='utf-8')
            print(f"\n지형 고도가 포함된 건물 데이터 저장: {elevation_building_csv}")
        else:
            print("건물 데이터를 찾을 수 없습니다.")
    else:
        print("지형 고도 정보 추출에 실패했습니다.")
