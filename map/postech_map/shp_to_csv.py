import geopandas as gpd

def convert_shp_to_csv(shp_path, csv_path):
    """
    Shapefile을 읽어 CSV 파일로 변환하는 함수입니다.

    Args:
        shp_path (str): 읽을 Shapefile의 경로
        csv_path (str): 저장할 CSV 파일의 경로
    """
    try:
        # EUC-KR 인코딩으로 Shapefile을 읽어옵니다.
        # .cpg 파일에 명시된 인코딩을 사용합니다.
        gdf = gpd.read_file(shp_path, encoding='euc-kr')

        # geometry 컬럼은 CSV로 표현하기 부적합하므로 제외합니다.
        df = gdf.drop(columns='geometry')

        # UTF-8 인코딩으로 CSV 파일을 저장합니다.
        # 'utf-8-sig'로 저장해야 Excel 등에서 한글 깨짐 없이 열립니다.
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')

        print(f"성공적으로 '{csv_path}' 파일을 생성하였습니다.")

    except Exception as e:
        print(f"오류가 발생했습니다: {e}")

if __name__ == '__main__':
    # Shapefile 경로와 저장할 CSV 파일 경로를 지정합니다.
    # 사용자의 환경에 맞게 경로를 수정해야 할 수 있습니다.
    shapefile_path = 'F_FAC_BUILDING_경북_포항시_북구/F_FAC_BUILDING_47113_202507.shp'
    output_csv_path = 'F_FAC_BUILDING_경북_포항시_북구/buk-gu_building_info.csv'

    # 함수를 호출하여 변환을 실행합니다.
    convert_shp_to_csv(shapefile_path, output_csv_path)