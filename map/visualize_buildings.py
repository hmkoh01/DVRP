import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from pathlib import Path
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# 한글 폰트가 없는 경우를 위한 대체 설정
try:
    plt.rcParams['font.family'] = ['Malgun Gothic', 'DejaVu Sans']
except:
    plt.rcParams['font.family'] = 'DejaVu Sans'

def load_building_data():
    """
    merge_maps 폴더에서 건물 데이터를 로드 (실제 높이 정보 포함)
    """
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
    return None
    """
    merge_maps 폴더에서 건물 데이터를 로드 (실제 높이 정보 포함)
    """
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
    return None
    """
    merge_maps 폴더에서 건물 데이터를 로드
    """
    base_path = Path("map/postech_map/merged_maps")
    
    # CSV 파일 로드
    csv_file = base_path / "postech_buildings_routing.csv"
    if csv_file.exists():
        df = pd.read_csv(csv_file)
        print(f"CSV 파일에서 {len(df)}개 건물 로드 완료")
        return df
    
    # GeoJSON 파일 로드 (CSV가 없는 경우)
    geojson_file = base_path / "postech_buildings_routing.geojson"
    if geojson_file.exists():
        gdf = gpd.read_file(geojson_file)
        print(f"GeoJSON 파일에서 {len(gdf)}개 건물 로드 완료")
        return gdf
    
    print("건물 데이터 파일을 찾을 수 없습니다.")
    return None

def generate_realistic_heights(data):
    """
    실제 건물 높이 정보 사용 (있는 경우) 또는 현실적인 높이 생성
    """
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
    
    return np.array(heights)
    """
    건물 ID를 기반으로 현실적인 높이 분포 생성
    """
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
    
    return np.array(heights)

def create_3d_box(x, y, z, width, height, depth):
    """
    3D 박스의 정점들을 생성
    """
    # 박스의 8개 정점
    vertices = [
        [x - width/2, y - height/2, z],           # 0: 바닥 왼쪽 앞
        [x + width/2, y - height/2, z],           # 1: 바닥 오른쪽 앞
        [x + width/2, y + height/2, z],           # 2: 바닥 오른쪽 뒤
        [x - width/2, y + height/2, z],           # 3: 바닥 왼쪽 뒤
        [x - width/2, y - height/2, z + depth],   # 4: 천장 왼쪽 앞
        [x + width/2, y - height/2, z + depth],   # 5: 천장 오른쪽 앞
        [x + width/2, y + height/2, z + depth],   # 6: 천장 오른쪽 뒤
        [x - width/2, y + height/2, z + depth]    # 7: 천장 왼쪽 뒤
    ]
    
    # 박스의 6개 면
    faces = [
        [vertices[0], vertices[1], vertices[2], vertices[3]],  # 바닥면
        [vertices[4], vertices[5], vertices[6], vertices[7]],  # 천장면
        [vertices[0], vertices[1], vertices[5], vertices[4]],  # 앞면
        [vertices[2], vertices[3], vertices[7], vertices[6]],  # 뒷면
        [vertices[0], vertices[3], vertices[7], vertices[4]],  # 왼쪽면
        [vertices[1], vertices[2], vertices[6], vertices[5]]   # 오른쪽면
    ]
    
    return faces

def create_3d_buildings_plot(data, output_path):
    """
    3D 입체 건물 시각화 생성 (실제 높이 반영)
    """
    print("3D 입체 건물 시각화 생성 중...")
    
    # 3D 플롯
    fig = plt.figure(figsize=(15, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    if isinstance(data, pd.DataFrame):
        x_coords = data['longitude']
        y_coords = data['latitude']
    else:
        x_coords = data.geometry.x
        y_coords = data.geometry.y
    
    # 현실적인 건물 높이 생성
    building_heights = generate_realistic_heights(data)
    
    # 건물 크기 설정 (데이터 범위에 따라 조정)
    x_range = x_coords.max() - x_coords.min()
    y_range = y_coords.max() - y_coords.min()
    
    # 건물 크기 계산 (전체 범위의 일정 비율)
    building_width = x_range * 0.0005   # 경도 방향 크기 (더 작게)
    building_height = y_range * 0.0005  # 위도 방향 크기 (더 작게)
    
    # 건물들을 3D 박스로 그리기
    for i in range(len(data)):
        x, y = x_coords.iloc[i], y_coords.iloc[i]
        z = 0  # 지면에서 시작
        depth = building_heights[i]  # 실제 높이
        
        # 3D 박스 생성
        box_faces = create_3d_box(x, y, z, building_width, building_height, depth)
        
        # 박스 그리기 (투명한 회색, 검은색 테두리)
        poly3d = Poly3DCollection(box_faces, 
                                 facecolor='lightgray', 
                                 edgecolor='black', 
                                 alpha=0.7, 
                                 linewidth=0.5)
        ax.add_collection3d(poly3d)
    
    ax.set_title('POSTECH Buildings 3D Distribution (Realistic Heights)', fontsize=16, fontweight='bold')
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_zlabel('Height (meters)', fontsize=12)
    
    # 축 범위 설정
    ax.set_xlim(x_coords.min() - x_range*0.05, x_coords.max() + x_range*0.05)
    ax.set_ylim(y_coords.min() - y_range*0.05, y_coords.max() + y_range*0.05)
    ax.set_zlim(0, building_heights.max() + 10)  # 최대 높이 + 여유분
    
    plt.tight_layout()
    
    # 저장
    output_file = output_path / "postech_buildings_3d.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"3D 입체 건물 시각화 저장: {output_file}")
    plt.show()

def create_3d_buildings_plotly(data, output_path):
    """
    Plotly를 사용한 3D 입체 건물 인터랙티브 시각화 (실제 높이 반영)
    """
    print("Plotly 3D 입체 건물 인터랙티브 시각화 생성 중...")
    
    if isinstance(data, pd.DataFrame):
        x_coords = data['longitude']
        y_coords = data['latitude']
    else:
        x_coords = data.geometry.x
        y_coords = data.geometry.y
    
    # 현실적인 건물 높이 생성
    building_heights = generate_realistic_heights(data)
    
    # 건물 크기 설정
    x_range = x_coords.max() - x_coords.min()
    y_range = y_coords.max() - y_coords.min()
    
    building_width = x_range * 0.0005
    building_height = y_range * 0.0005
    
    # 3D 박스들을 생성
    fig = go.Figure()
    
    for i in range(len(data)):
        x, y = x_coords.iloc[i], y_coords.iloc[i]
        z = 0  # 지면에서 시작
        depth = building_heights[i]  # 실제 높이
        
        # 3D 박스의 정점들
        vertices = [
            [x - building_width/2, y - building_height/2, z],           # 0
            [x + building_width/2, y - building_height/2, z],           # 1
            [x + building_width/2, y + building_height/2, z],           # 2
            [x - building_width/2, y + building_height/2, z],           # 3
            [x - building_width/2, y - building_height/2, z + depth],   # 4
            [x + building_width/2, y - building_height/2, z + depth],   # 5
            [x + building_width/2, y + building_height/2, z + depth],   # 6
            [x - building_width/2, y + building_height/2, z + depth]    # 7
        ]
        
        # 박스의 면들
        faces = [
            [0, 1, 2, 3],  # 바닥면
            [4, 5, 6, 7],  # 천장면
            [0, 1, 5, 4],  # 앞면
            [2, 3, 7, 6],  # 뒷면
            [0, 3, 7, 4],  # 왼쪽면
            [1, 2, 6, 5]   # 오른쪽면
        ]
        
        # 3D 메시 추가
        fig.add_trace(go.Mesh3d(
            x=[v[0] for v in vertices],
            y=[v[1] for v in vertices],
            z=[v[2] for v in vertices],
            i=[face[0] for face in faces],
            j=[face[1] for face in faces],
            k=[face[2] for face in faces],
            opacity=0.7,
            color='lightgray',
            flatshading=True,
            hoverinfo='skip'
        ))
    
    fig.update_layout(
        title='POSTECH Buildings 3D Distribution (Realistic Heights)',
        scene=dict(
            xaxis_title='Longitude',
            yaxis_title='Latitude',
            zaxis_title='Height (meters)'
        ),
        width=1200,
        height=800
    )
    
    # 저장
    output_file = output_path / "postech_buildings_3d_plotly.html"
    fig.write_html(str(output_file))
    print(f"Plotly 3D 입체 건물 시각화 저장: {output_file}")
    
    # 이미지로도 저장
    try:
        img_file = output_path / "postech_buildings_3d_plotly.png"
        fig.write_image(str(img_file))
        print(f"Plotly 3D 입체 건물 이미지 저장: {img_file}")
    except Exception as e:
        print(f"이미지 저장 실패 (kaleido 패키지 필요): {e}")

def main():
    """
    메인 함수
    """
    print("POSTECH 건물 데이터 3D 입체 시각화를 시작합니다...")
    
    # 데이터 로드
    data = load_building_data()
    if data is None:
        return
    
    # 출력 폴더 생성
    output_path = Path("map/postech_map/visualizations")
    output_path.mkdir(exist_ok=True)
    
    print(f"데이터 정보:")
    print(f"- 총 건물 수: {len(data)}")
    if 'source' in data.columns:
        print(f"- 출처별 분포:")
        print(data['source'].value_counts())
    
    # 3D 입체 건물 시각화 생성
    try:
        create_3d_buildings_plot(data, output_path)
    except Exception as e:
        print(f"3D 입체 건물 시각화 오류: {e}")
    
    try:
        create_3d_buildings_plotly(data, output_path)
    except Exception as e:
        print(f"Plotly 3D 입체 건물 시각화 오류: {e}")
    
    print(f"\n3D 입체 건물 시각화 완료!")
    print(f"결과는 '{output_path}' 폴더에서 확인할 수 있습니다.")
    print("\n생성된 파일들:")
    print("- postech_buildings_3d.png: 3D 입체 건물 분포도 (실제 높이 반영)")
    print("- postech_buildings_3d_plotly.html: 3D 입체 건물 인터랙티브 시각화 (실제 높이 반영)")
    print("- postech_buildings_3d_plotly.png: 3D 입체 건물 이미지 (실제 높이 반영)")

if __name__ == "__main__":
    main() 