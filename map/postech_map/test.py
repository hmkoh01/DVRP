import geopandas as gpd

# 사용법:
# 1. 새로 주신 .shp 파일과 관련 파일들(.dbf, .shx 등)을 하나의 폴더에 넣어주세요.
# 2. 아래 file_path 변수에 해당 .shp 파일의 전체 경로를 입력해주세요.
#    예: "C:/Users/MyUser/Documents/Pohang_Data/AL_D010_47_20250804(3).shp"

file_path = "F_FAC_BUILDING_경북_포항시_남구/F_FAC_BUILDING_47111_202507.shp"

# EUC-KR 인코딩으로 파일 로드
try:
    gdf = gpd.read_file(file_path, encoding='EUC-KR')
    print("파일의 실제 컬럼 목록:")
    print(list(gdf.columns))
    print("\n데이터 샘플:")
    print(gdf.head())
except Exception as e:
    print(f"파일 로드 오류: {e}")