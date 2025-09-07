# 필요한 라이브러리들을 불러옵니다.
import geopandas as gpd  # 지리공간 데이터를 다루기 위한 라이브러리 (Shapefile 등)
import matplotlib.pyplot as plt  # 데이터 시각화 라이브러리
from mpl_toolkits.mplot3d import Axes3D  # 3D 시각화를 위한 도구
import numpy as np  # 수치 계산을 위한 라이브러리
import pandas as pd  # 데이터 분석 및 조작을 위한 라이브러리 (GeoPandas의 기반)
import matplotlib.cm as cm  # 컬러맵 사용을 위한 라이브러리
import matplotlib.colors as colors  # 색상 정규화를 위한 라이브러리
from scipy.interpolate import griddata  # 공간 보간법을 위한 라이브러리

# --- 1. CONFIGURATION BLOCK (설정 블록) ---
# 이 블록에서 파일 경로, 출력 파일 이름, 해상도 등 스크립트의 주요 설정을 관리합니다.
CONFIG = {
    # 등고선 데이터 파일 경로 목록 (지형의 높이 정보)
    "terrain_contour_paths": [
        "./국가기본공간정보_포스텍/NF_L_F01000_L_F01000_000000.shp",
    ],
    # 표고점 데이터 파일 경로 목록 (산 정상 등 특정 지점의 높이, 정확도 향상용)
    "spot_elevation_paths": [
        "./국가기본공간정보_포스텍/NF_P_F02000_P_F02000_000000.shp",
    ],
    # 건물 데이터 파일 경로 목록 (여러 파일로 나뉘어 있을 경우 모두 추가)
    "building_paths": [
        "./경상북도_GIS건물통합정보/AL_D010_47_20250804.shp",
        "./경상북도_GIS건물통합정보/AL_D010_47_20250804(2).shp",
        "./경상북도_GIS건물통합정보/AL_D010_47_20250804(3).shp"
    ],
    # 결과물로 저장될 건물 목록 CSV 파일 이름
    "output_csv_filename": "postech_building_list.csv",
    # 결과물로 저장될 2D 지도 이미지 파일 이름
    "output_2d_filename": "postech_2d_map.png",
    # 결과물로 저장될 3D 지도 이미지 파일 이름
    "output_3d_filename": "postech_3d_map.png",
    # 2D 이미지 저장 시 해상도 (DPI: Dots Per Inch)
    "dpi_2d": 300,
    # 3D 이미지 저장 시 해상도
    "dpi_3d": 300
}

def load_data(config):
    """모든 Shapefile을 불러오고, 데이터를 병합하며, 건물 데이터의 컬럼 이름을 변경합니다."""
    print("✅ 1. 데이터 로딩을 시작합니다...")

    # 내부 헬퍼 함수: 지정된 경로의 Shapefile들을 읽어 하나의 GeoDataFrame으로 합칩니다.
    def _load_and_concat_shp(paths, encoding):
        gdf_list = []  # 개별 GeoDataFrame을 담을 리스트
        if not paths or not paths[0]: return gpd.GeoDataFrame() # 경로가 비어있으면 빈 프레임 반환
        for path in paths:
            try:
                # GeoPandas를 이용해 Shapefile을 읽습니다. 한글 깨짐 방지를 위해 인코딩 지정.
                gdf_list.append(gpd.read_file(path, encoding=encoding))
            except Exception as e:
                print(f"    - 🚨 오류: '{path}' 파일 로딩 실패: {e}")
        # 리스트에 담긴 모든 GeoDataFrame을 하나로 합칩니다.
        return pd.concat(gdf_list, ignore_index=True) if gdf_list else gpd.GeoDataFrame()

    # 설정값(config)에 따라 각 데이터 로딩
    terrain_gdf = _load_and_concat_shp(config["terrain_contour_paths"], 'UTF-8') # 등고선 데이터
    building_gdf = _load_and_concat_shp(config["building_paths"], 'EUC-KR')   # 건물 데이터 (주로 EUC-KR 인코딩)
    spot_gdf = _load_and_concat_shp(config["spot_elevation_paths"], 'UTF-8')   # 표고점 데이터

    # 건물 데이터의 컬럼 이름(A1, A2 등)을 이해하기 쉬운 이름으로 변경합니다.
    column_rename_map = {
        'A1': 'UFID',      # GIS건물통합식별번호
        'A2': 'PNU',       # 고유번호 (필지)
        'A9': 'USABILITY', # 건축물용도명
        'A16': 'HEIGHT',   # 높이(m)
        'A24': 'BLD_NM',   # 건물명
        'A26': 'GRND_FLR', # 지상층수
    }
    building_gdf.rename(columns=column_rename_map, inplace=True)
    print("    - 건물 데이터의 컬럼 이름을 의미있는 이름으로 변경했습니다.")
    
    # 로딩된 데이터의 개수를 출력합니다.
    print(f"    - 총 {len(building_gdf)}개의 건물, {len(terrain_gdf)}개의 등고선, {len(spot_gdf)}개의 표고점 데이터를 로딩 및 병합했습니다.")
    
    # 필수 데이터(지형, 건물)가 없으면 처리를 중단합니다.
    if terrain_gdf.empty or building_gdf.empty:
        print("🚨 [중요] 등고선 또는 건물 데이터가 비어있어 처리를 중단합니다.")
        return gpd.GeoDataFrame(), gpd.GeoDataFrame(), gpd.GeoDataFrame()
    
    # 로딩된 데이터들을 반환합니다.
    return terrain_gdf, building_gdf, spot_gdf

def preprocess_data(terrain_gdf, building_gdf, spot_elevation_gdf):
    """모든 전처리 과정(좌표계 통일, 필터링, 고도 계산 등)을 수행합니다."""
    print("✅ 2. 데이터 전처리를 시작합니다...")
    
    # 1. 좌표계 통일(CRS Unification): 모든 데이터를 하나의 좌표계로 맞춰야 위치를 정확히 비교할 수 있습니다.
    # 기준 좌표계는 등고선 데이터의 좌표계로 설정합니다.
    target_crs = terrain_gdf.crs
    if building_gdf.crs != target_crs:
        building_gdf = building_gdf.to_crs(target_crs)
    if not spot_elevation_gdf.empty and spot_elevation_gdf.crs != target_crs:
        spot_elevation_gdf = spot_elevation_gdf.to_crs(target_crs)

    # 2. 공간 필터링(Spatial Filtering): 등고선 데이터가 포함하는 전체 영역을 계산합니다.
    minx, miny, maxx, maxy = terrain_gdf.total_bounds
    # 이 영역 내에 있는 건물들만 필터링하여 처리 효율을 높입니다.
    filtered_building_gdf = building_gdf.cx[minx:maxx, miny:maxy].copy()
    print(f"    - 지형 범위 내 필터링 후 건물 수: {len(filtered_building_gdf)}")
    
    # 지형과 표고점 데이터도 동일한 범위로 필터링합니다.
    filtered_terrain_gdf = terrain_gdf.cx[minx:maxx, miny:maxy].copy()
    filtered_spot_elevation_gdf = spot_elevation_gdf.cx[minx:maxx, miny:maxy].copy() if not spot_elevation_gdf.empty else gpd.GeoDataFrame()
    
    # 나중에 데이터를 합칠 때 기준이 될 고유 ID를 각 건물에 부여합니다.
    filtered_building_gdf = filtered_building_gdf.reset_index(drop=True)
    filtered_building_gdf['unique_id'] = filtered_building_gdf.index


    # 3. 건물 높이 계산: 3D 시각화를 위해 각 건물의 높이를 결정합니다.
    height_col, floor_col = 'HEIGHT', 'GRND_FLR'
    print(f"    - 건물 높이를 계산합니다: '{height_col}' 값 사용, 0일 경우 '{floor_col}' * 3으로 추정")

    # 'HEIGHT' 컬럼을 숫자형으로 변환합니다. 존재하지 않거나 변환 실패 시 0으로 채웁니다.
    if height_col in filtered_building_gdf.columns:
        filtered_building_gdf[height_col] = pd.to_numeric(filtered_building_gdf[height_col], errors='coerce').fillna(0)
    else:
        print(f"    - 🚨 경고: '{height_col}' 컬럼이 없어 높이를 0으로 간주하고 시작합니다.")
        filtered_building_gdf[height_col] = 0

    # 'GRND_FLR' 컬럼을 숫자형으로 변환합니다. 존재하지 않거나 변환 실패 시 0으로 채웁니다.
    if floor_col in filtered_building_gdf.columns:
        filtered_building_gdf[floor_col] = pd.to_numeric(filtered_building_gdf[floor_col], errors='coerce').fillna(0)
    else:
        print(f"    - 🚨 경고: '{floor_col}' 컬럼이 없어 높이 추정이 불가능합니다.")
        filtered_building_gdf[floor_col] = 0
        
    # 'HEIGHT'가 0인 건물의 인덱스를 찾습니다.
    indices_to_estimate = filtered_building_gdf[height_col] == 0
    
    # 해당 인덱스의 건물들에 대해 높이를 재계산합니다: 지상층수 * 3
    # .loc[indices, column]을 사용하여 특정 행과 열을 선택해 값을 변경합니다.
    estimated_heights = filtered_building_gdf.loc[indices_to_estimate, floor_col] * 3
    filtered_building_gdf.loc[indices_to_estimate, height_col] = estimated_heights

    # 최종 결과 보고
    num_estimated = indices_to_estimate.sum()
    if num_estimated > 0:
        print(f"    - '{height_col}'가 0이었던 {num_estimated}개 건물에 대해 '{floor_col}'를 이용해 높이를 추정했습니다.")
    
    # 4. 건물 바닥 고도 계산 (1단계 - Spatial Join)
    print("    - 1단계: Spatial Join으로 고도를 할당합니다...")
    # sjoin: 공간 정보를 기준으로 두 데이터를 합칩니다. 'intersects'는 '서로 만나는' 경우를 의미합니다.
    # 각 건물이 어떤 등고선(CONT)과 만나는지 찾아 고도를 할당합니다.
    buildings_with_terrain = gpd.sjoin(filtered_building_gdf, filtered_terrain_gdf[['CONT', 'geometry']], how="left", predicate='intersects')
    
    # 한 건물이 여러 등고선과 만날 수 있으므로, 고유 ID로 그룹화하여 평균 고도를 계산합니다.
    # 동시에 다른 중요 정보(건물명, 높이 등)는 그대로 유지합니다.
    agg_df = buildings_with_terrain.groupby('unique_id').agg({
        'CONT': 'mean', 'HEIGHT': 'first', 'geometry': 'first', 
        'UFID': 'first', 'PNU': 'first', 'BLD_NM': 'first', 
        'GRND_FLR': 'first', 'USABILITY': 'first' 
    }).reset_index()
    processed_buildings = gpd.GeoDataFrame(agg_df, geometry='geometry', crs=target_crs)
    print(f"    - Spatial Join 후 건물 수: {len(processed_buildings)}")

    # 5. 건물 바닥 고도 계산 (2단계 - 공간 보간법)
    # sjoin으로 고도를 찾지 못한 건물들(등고선 사이에 위치)을 대상으로 보간법을 수행합니다.
    buildings_to_interpolate = processed_buildings[processed_buildings['CONT'].isna()].copy()
    if not buildings_to_interpolate.empty:
        print(f"    - 2단계: {len(buildings_to_interpolate)}개 건물에 대해 공간 보간법을 수행합니다...")
        # 주변의 알려진 고도 지점을 모두 수집합니다. (등고선 + 표고점)
        # 1. 등고선 위의 모든 점들의 좌표(x, y)와 고도값(CONT)을 추출합니다.
        contour_points_gdf = filtered_terrain_gdf[pd.notna(filtered_terrain_gdf['CONT'])].explode(index_parts=False).get_coordinates()
        known_points_list = [contour_points_gdf[['x', 'y']].values]
        known_values_list = [filtered_terrain_gdf.loc[contour_points_gdf.index, 'CONT'].values]
        
        # 2. 표고점 데이터가 있으면, 표고점의 좌표(x, y)와 고도값(NUME)도 추가합니다.
        if not filtered_spot_elevation_gdf.empty and 'NUME' in filtered_spot_elevation_gdf.columns:
            spot_points_gdf = filtered_spot_elevation_gdf[pd.notna(filtered_spot_elevation_gdf['NUME'])].get_coordinates()
            if not spot_points_gdf.empty:
                known_points_list.append(spot_points_gdf[['x', 'y']].values)
                known_values_list.append(filtered_spot_elevation_gdf.loc[spot_points_gdf.index, 'NUME'].values)
        
        # 수집된 점들을 하나의 배열로 합칩니다.
        known_points = np.vstack(known_points_list)
        known_values = np.concatenate(known_values_list)
        
        if len(known_points) > 0:
            # 고도를 추정할 건물들의 중심점 좌표를 가져옵니다.
            centroids = buildings_to_interpolate.geometry.centroid
            target_points = np.vstack((centroids.x, centroids.y)).T
            # griddata: 주변의 알려진 점들(known_points, known_values)을 이용해,
            # 목표 지점(target_points)의 값을 선형으로 추정(method='linear')합니다.
            interpolated_values = griddata(known_points, known_values, target_points, method='linear', fill_value=0)
            # 추정된 고도값을 원래 데이터프레임에 채워넣습니다.
            processed_buildings.loc[buildings_to_interpolate.index, 'CONT'] = interpolated_values

    # 6. 건물의 절대 높이 계산: 건물의 최종 높이 = 바닥의 해발고도(CONT) + 건물 자체 높이(HEIGHT)
    processed_buildings['ABSOLUTE_HEIGHT'] = processed_buildings['CONT'] + processed_buildings['HEIGHT']
    print("✅ 데이터 전처리 및 계산 완료.")
    return filtered_terrain_gdf, processed_buildings

def export_building_list_with_coords(buildings_gdf, config):
    """건물의 경위도 좌표 및 주요 정보를 추출하여 CSV 파일로 저장합니다."""
    if buildings_gdf.empty: return # 처리할 건물이 없으면 함수 종료
    print("\n✅ 3. 시각화된 건물 목록과 좌표를 파일로 저장합니다...")
    
    buildings_to_export = buildings_gdf.copy()
    # 1. 각 건물의 중심점 좌표를 계산합니다. (현재는 미터 기반 좌표계)
    centroids_projected = buildings_to_export.geometry.centroid
    # 2. 중심점 좌표를 전 세계 표준인 위도/경도(WGS84, EPSG:4326)로 변환합니다.
    centroids_wgs84 = centroids_projected.to_crs("EPSG:4326")
    # 3. 변환된 위도(latitude)와 경도(longitude)를 새 컬럼으로 추가합니다.
    buildings_to_export['longitude'] = centroids_wgs84.x
    buildings_to_export['latitude'] = centroids_wgs84.y
    
    # CSV 파일로 저장할 경로를 설정에서 가져옵니다.
    output_path = config["output_csv_filename"]
    
    # CSV에 저장할 컬럼 목록을 지정합니다.
    columns_to_save = ['UFID', 'PNU', 'BLD_NM', 'USABILITY', 'GRND_FLR', 'HEIGHT', 'CONT', 'ABSOLUTE_HEIGHT', 'latitude', 'longitude']
    # 데이터에 존재하는 컬럼만 최종적으로 선택합니다.
    final_columns = [col for col in columns_to_save if col in buildings_to_export.columns]
    
    # 지정된 컬럼만 CSV 파일로 저장합니다. index=False는 불필요한 인덱스 저장을 방지합니다.
    # encoding='utf-8-sig'는 Excel에서 한글이 깨지지 않도록 보장합니다.
    buildings_to_export[final_columns].to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"    - 건물 목록을 '{output_path}' 파일로 저장했습니다.")

def visualize_2d(terrain, buildings, config):
    """2D 시각화 결과물을 생성하고 저장합니다."""
    print("\n✅ 4. 2D 시각화를 생성합니다...")

    # 1. 2D 지도 축을 위도/경도로 표시하기 위해 데이터를 WGS84 좌표계로 변환합니다.
    print("    - 2D 지도용 좌표를 위도/경도로 변환합니다...")
    terrain_wgs84 = terrain.to_crs("EPSG:4326")
    buildings_wgs84 = buildings.to_crs("EPSG:4326")

    # 2. 시각화를 위한 그림판(fig)과 좌표축(ax)을 생성합니다.
    fig, ax = plt.subplots(1, 1, figsize=(15, 15))
    # 3. 배경으로 지형(등고선)을 연한 회색으로 그립니다.
    terrain_wgs84.plot(ax=ax, color='gainsboro', linewidth=0.5)
    # 4. 건물들을 그립니다. 이때 'ABSOLUTE_HEIGHT' 값에 따라 색상을 다르게 표현합니다.
    buildings_wgs84.plot(column='ABSOLUTE_HEIGHT', # 색상 기준이 될 컬럼
                         cmap='plasma',           # 사용할 컬러맵 (낮으면 보라, 높으면 노랑)
                         ax=ax,
                         legend=True,             # 색상 범례 표시
                         legend_kwds={'label': "Absolute Height (m)", 'orientation': "vertical", 'shrink': 0.5, 'aspect': 30})
    
    # 5. 그래프의 x, y축 레이블과 제목을 설정합니다.
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title('Pohang 2D Map (Latitude/Longitude Axes)')
    
    # 6. 완성된 그래프를 이미지 파일로 저장합니다.
    plt.savefig(config["output_2d_filename"], dpi=config["dpi_2d"], bbox_inches='tight')
    print(f"    - 2D 지도를 '{config['output_2d_filename']}' 파일로 저장했습니다.")
   
def visualize_3d(terrain, buildings, config):
    """3D 시각화 결과물을 생성하고 저장합니다."""
    print("\n✅ 5. 3D 시각화를 생성합니다...")
    fig = plt.figure(figsize=(18, 15))
    # 3D 그래프를 그릴 수 있는 축(ax)을 생성합니다.
    ax = fig.add_subplot(111, projection='3d')

    # 1. 지형 표면 시각화
    if not terrain.empty and 'CONT' in terrain.columns:
        # 등고선 데이터에서 좌표(x, y)와 고도(z)를 추출합니다.
        points = terrain[pd.notna(terrain['CONT'])].explode(index_parts=False).get_coordinates()
        if not points.empty:
            points['Z'] = terrain.loc[points.index, 'CONT']
            # plot_trisurf: 점들을 삼각형으로 연결하여 3D 표면을 만듭니다.
            ax.plot_trisurf(points['x'], points['y'], points['Z'], cmap='Greens', alpha=0.5, zorder=1)

    # 2. 건물 시각화
    if not buildings.empty:
        # 건물의 절대 높이에 따라 색상을 매핑하기 위한 준비
        min_h, max_h = buildings['ABSOLUTE_HEIGHT'].min(), buildings['ABSOLUTE_HEIGHT'].max()
        cmap, norm = plt.get_cmap('plasma'), colors.Normalize(vmin=min_h, vmax=max_h if max_h > min_h else min_h + 1)
        
        # 각 건물을 하나씩 순회하며 3D로 그립니다.
        for _, row in buildings.iterrows():
            if row['HEIGHT'] > 0 and row.geometry and hasattr(row.geometry, 'exterior'):
                # 높이에 맞는 색상 지정
                color = cmap(norm(row['ABSOLUTE_HEIGHT']))
                # 건물의 바닥 높이(z_bottom)와 꼭대기 높이(z_top)
                z_bottom, z_top = row['CONT'], row['ABSOLUTE_HEIGHT']
                # 건물의 2D 외곽선 좌표(x, y)
                x, y = row.geometry.exterior.xy
                # 건물 밑면 그리기 (z_bottom 높이에 외곽선)
                ax.plot(x, y, z_bottom, color=color, linewidth=0.5, zorder=2)
                # 건물 윗면 그리기 (z_top 높이에 외곽선)
                ax.plot(x, y, z_top, color=color, linewidth=1, zorder=3)
                # 건물 벽(기둥) 그리기: 각 꼭짓점에서 밑면과 윗면을 잇는 수직선
                for i in range(len(x)):
                    ax.plot([x[i], x[i]], [y[i], y[i]], [z_bottom, z_top], color=color, linewidth=1, zorder=2)
        
        # 3. 컬러바 추가: 색상이 어떤 높이 값을 의미하는지 보여주는 범례
        mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
        fig.colorbar(mappable, ax=ax, shrink=0.6, aspect=10, label='Absolute Height (m)')
        
    # 4. 3D 그래프의 제목과 축 레이블 설정
    ax.set_title('Pohang 3D Map (Meter-based Axes for True Scale)')
    ax.set_xlabel("X Coordinate (meters)")
    ax.set_ylabel("Y Coordinate (meters)")
    ax.set_zlabel("Z Coordinate (meters, Elevation)")
    # 5. 3D 뷰의 시점(카메라 각도) 설정: elev는 높이, azim은 방위각
    ax.view_init(elev=30, azim=-45)
    # 6. 완성된 3D 뷰를 이미지 파일로 저장
    plt.savefig(config["output_3d_filename"], dpi=config["dpi_3d"], bbox_inches='tight')
    print(f"    - 3D 지도를 '{config['output_3d_filename']}' 파일로 저장했습니다.")

def main():
    """전체 데이터 처리 및 시각화 파이프라인을 실행하는 메인 함수입니다."""
    try:
        # 1단계: 데이터 로딩
        terrain_data, building_data, spot_data = load_data(CONFIG)
        # 필수 데이터가 없으면 종료
        if terrain_data.empty or building_data.empty:
            return
        # 2단계: 데이터 전처리 및 계산
        final_terrain, final_buildings = preprocess_data(terrain_data, building_data, spot_data)
        
        # 3단계: 처리된 건물 목록을 CSV로 저장
        export_building_list_with_coords(final_buildings, CONFIG)
        
        # 4단계: 2D 지도 생성 및 저장
        visualize_2d(final_terrain, final_buildings, CONFIG)
        # 5단계: 3D 지도 생성 및 저장
        visualize_3d(final_terrain, final_buildings, CONFIG)
        
    # 예외 처리: 스크립트 실행 중 오류 발생 시 메시지를 출력하고 종료합니다.
    except FileNotFoundError as e:
        print(f"🚨 [오류] 파일을 찾을 수 없습니다. CONFIG 블록의 파일 경로를 확인하세요. 상세 정보: {e}")
    except Exception as e:
        print(f"🚨 [오류] 예상치 못한 오류가 발생했습니다: {e}")

# 이 스크립트 파일이 직접 실행될 때만 main() 함수를 호출합니다.
if __name__ == "__main__":
    main()