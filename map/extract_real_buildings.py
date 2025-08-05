import geopandas as gpd
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def extract_real_building_data():
    """
    원본 수치지도에서 실제 건물 정보 추출
    """
    base_path = Path("map/postech_map/raw_maps")
    
    # 두 수치지도 폴더 경로
    map1_path = base_path / "(B010)수치지도_36914093_2024_00000634112839"
    map2_path = base_path / "(B010)수치지도_36914094_2024_0000056145927"
    
    print("원본 수치지도에서 실제 건물 정보 추출 중...")
    
    all_buildings = []
    
    # 지도 1에서 건물 정보 추출
    try:
        # N3A_B0010000.shp 파일 (가장 많은 건물 정보 포함)
        gdf1 = gpd.read_file(map1_path / "N3A_B0010000.shp")
        print(f"지도 1에서 {len(gdf1)}개 건물 로드")
        
        # 중심점 계산
        gdf1['centroid'] = gdf1.geometry.centroid
        gdf1['longitude'] = gdf1['centroid'].x
        gdf1['latitude'] = gdf1['centroid'].y
        gdf1['source'] = 'map1'
        gdf1['building_id'] = range(len(gdf1))
        
        # 높이 계산 (층수 기반)
        gdf1['height_m'] = gdf1['층수'] * 3.0  # 층당 3미터
        
        # 필요한 컬럼만 선택
        building_data1 = gdf1[['source', 'building_id', 'longitude', 'latitude', 'height_m', '층수', '용도', '종류']].copy()
        all_buildings.append(building_data1)
        
    except Exception as e:
        print(f"지도 1 처리 실패: {e}")
    
    # 지도 2에서 건물 정보 추출
    try:
        # 지도 2에서도 동일한 파일 찾기
        gdf2 = gpd.read_file(map2_path / "N3A_B0010000.shp")
        print(f"지도 2에서 {len(gdf2)}개 건물 로드")
        
        # 중심점 계산
        gdf2['centroid'] = gdf2.geometry.centroid
        gdf2['longitude'] = gdf2['centroid'].x
        gdf2['latitude'] = gdf2['centroid'].y
        gdf2['source'] = 'map2'
        gdf2['building_id'] = range(len(gdf2))
        
        # 높이 계산 (층수 기반)
        gdf2['height_m'] = gdf2['층수'] * 3.0  # 층당 3미터
        
        # 필요한 컬럼만 선택
        building_data2 = gdf2[['source', 'building_id', 'longitude', 'latitude', 'height_m', '층수', '용도', '종류']].copy()
        all_buildings.append(building_data2)
        
    except Exception as e:
        print(f"지도 2 처리 실패: {e}")
    
    if not all_buildings:
        print("추출된 건물 데이터가 없습니다.")
        return None
    
    # 모든 건물 데이터 병합
    merged_buildings = pd.concat(all_buildings, ignore_index=True)
    
    # 건물 ID 재할당
    merged_buildings['building_id'] = range(len(merged_buildings))
    
    print(f"\n총 추출된 건물 수: {len(merged_buildings)}")
    print(f"출처별 분포:")
    print(merged_buildings['source'].value_counts())
    
    # 높이 통계
    print(f"\n높이 통계:")
    print(merged_buildings['height_m'].describe())
    
    # 용도별 분포
    print(f"\n용도별 분포:")
    print(merged_buildings['용도'].value_counts())
    
    # 종류별 분포
    print(f"\n종류별 분포:")
    print(merged_buildings['종류'].value_counts())
    
    # 출력 폴더 생성
    output_path = Path("map/postech_map/merged_maps")
    output_path.mkdir(exist_ok=True)
    
    # 실제 높이 정보가 포함된 CSV 저장
    real_height_csv = output_path / "postech_buildings_real_height.csv"
    merged_buildings.to_csv(real_height_csv, index=False, encoding='utf-8')
    print(f"\n실제 높이 정보 저장: {real_height_csv}")
    
    # 라우팅용 CSV도 업데이트 (기존 파일 대체)
    routing_csv = output_path / "postech_buildings_routing.csv"
    routing_data = merged_buildings[['source', 'building_id', 'longitude', 'latitude']].copy()
    routing_data.to_csv(routing_csv, index=False, encoding='utf-8')
    print(f"라우팅용 데이터 업데이트: {routing_csv}")
    
    # GeoJSON으로도 저장
    geojson_file = output_path / "postech_buildings_real_height.geojson"
    
    # GeoDataFrame으로 변환
    gdf_output = gpd.GeoDataFrame(
        merged_buildings,
        geometry=gpd.points_from_xy(merged_buildings['longitude'], merged_buildings['latitude']),
        crs="EPSG:5187"
    )
    gdf_output.to_file(geojson_file, driver='GeoJSON')
    print(f"GeoJSON 저장: {geojson_file}")
    
    return merged_buildings

def update_visualization_code():
    """
    시각화 코드를 실제 높이 정보를 사용하도록 업데이트
    """
    print("\n시각화 코드 업데이트 중...")
    
    # visualize_buildings.py 파일 읽기
    viz_file = Path("map/visualize_buildings.py")
    
    if viz_file.exists():
        with open(viz_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 실제 높이 정보를 사용하도록 수정
        new_content = content.replace(
            "def load_building_data():",
            """def load_building_data():
    \"\"\"
    merge_maps 폴더에서 건물 데이터를 로드 (실제 높이 정보 포함)
    \"\"\"
    base_path = Path("map/postech_map/merged_maps")
    
    # 실제 높이 정보가 포함된 CSV 파일 로드
    real_height_csv = base_path / "postech_buildings_real_height.csv"
    if real_height_csv.exists():
        df = pd.read_csv(real_height_csv)
        print(f"실제 높이 정보 포함 CSV 파일에서 {len(df)}개 건물 로드 완료")
        return df
    
    # 기존 CSV 파일 로드 (fallback)
    csv_file = base_path / "postech_buildings_routing.csv"
    if csv_file.exists():
        df = pd.read_csv(csv_file)
        print(f"CSV 파일에서 {len(df)}개 건물 로드 완료")
        return df
    
    # GeoJSON 파일 로드 (fallback)
    geojson_file = base_path / "postech_buildings_routing.geojson"
    if geojson_file.exists():
        gdf = gpd.read_file(geojson_file)
        print(f"GeoJSON 파일에서 {len(gdf)}개 건물 로드 완료")
        return gdf
    
    print("건물 데이터 파일을 찾을 수 없습니다.")
    return None"""
        )
        
        # 실제 높이 정보를 사용하도록 generate_realistic_heights 함수 수정
        new_content = new_content.replace(
            "def generate_realistic_heights(building_ids, num_buildings):",
            """def generate_realistic_heights(data):
    \"\"\"
    실제 건물 높이 정보 사용 (있는 경우) 또는 현실적인 높이 생성
    \"\"\"
    # 실제 높이 정보가 있는지 확인
    if 'height_m' in data.columns and data['height_m'].notna().sum() > 0:
        print(f"실제 높이 정보 사용: {data['height_m'].notna().sum()}개 건물")
        return data['height_m'].fillna(15.0)  # 없는 경우 15m로 기본값
    
    # 실제 높이 정보가 없는 경우 기존 로직 사용
    print("실제 높이 정보 없음 - 가상 높이 생성")
    building_ids = data['building_id']
    num_buildings = len(data)
    
    np.random.seed(42)  # 재현 가능성을 위한 시드 설정
    
    # 건물 유형별 높이 분포 (미터 단위)
    building_types = {
        'residential': {'min': 10, 'max': 30, 'weight': 0.4},      # 주거용 건물
        'commercial': {'min': 20, 'max': 50, 'weight': 0.3},       # 상업용 건물
        'institutional': {'min': 15, 'max': 40, 'weight': 0.2},    # 기관용 건물
        'tall': {'min': 40, 'max': 80, 'weight': 0.1}             # 고층 건물
    }
    
    heights = []
    for i in range(num_buildings):
        # 건물 ID를 기반으로 건물 유형 결정
        building_id = building_ids.iloc[i] if hasattr(building_ids, 'iloc') else building_ids[i]
        
        # 건물 ID의 패턴에 따라 유형 결정
        if building_id % 20 == 0:  # 20의 배수는 고층 건물
            btype = 'tall'
        elif building_id % 7 == 0:  # 7의 배수는 상업용
            btype = 'commercial'
        elif building_id % 5 == 0:  # 5의 배수는 기관용
            btype = 'institutional'
        else:  # 나머지는 주거용
            btype = 'residential'
        
        # 해당 유형의 높이 범위에서 랜덤 선택
        min_h = building_types[btype]['min']
        max_h = building_types[btype]['max']
        height = np.random.uniform(min_h, max_h)
        
        # 약간의 노이즈 추가로 자연스러운 분포 생성
        noise = np.random.normal(0, 2)
        height = max(5, height + noise)  # 최소 5미터 보장
        
        heights.append(height)
    
    return np.array(heights)"""
        )
        
        # 함수 호출 부분 수정
        new_content = new_content.replace(
            "building_heights = generate_realistic_heights(data['building_id'], len(data))",
            "building_heights = generate_realistic_heights(data)"
        )
        
        # 파일 저장
        with open(viz_file, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print("시각화 코드 업데이트 완료!")

if __name__ == "__main__":
    print("실제 건물 정보 추출 및 시각화 코드 업데이트를 시작합니다...")
    
    # 1단계: 실제 건물 정보 추출
    building_data = extract_real_building_data()
    
    if building_data is not None:
        # 2단계: 시각화 코드 업데이트
        update_visualization_code()
        
        print("\n완료!")
        print("이제 visualize_buildings.py를 실행하면 실제 건물 높이 정보를 사용한 3D 시각화를 볼 수 있습니다.")
    else:
        print("\n건물 정보 추출에 실패했습니다.") 