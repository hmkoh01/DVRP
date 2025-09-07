"""
드론 음식 배달 시스템 메인 실행 파일 (AirSim 3D 시각화 + RouteManager 연동)
"""

import sys
from pathlib import Path
import warnings
import time
warnings.filterwarnings('ignore')

# 프로젝트 루트 경로 추가
sys.path.append(str(Path(__file__).parent.parent))

from algorithm.config import *
from algorithm.utils.data_loader import BuildingDataLoader
from algorithm.simulation.airsim_visualizer import AirSimVisualizer
from algorithm.utils.coordinate_converter import CoordinateConverter
from algorithm.simulation.order_generator import OrderGenerator
from algorithm.routing.realtime_route_manager import RealtimeRouteManager


class DroneDeliverySystem:
    def __init__(self, simulation_mode='realtime'):
        self.simulation_mode = simulation_mode
        self.data_loader = BuildingDataLoader()

        self.buildings = None
        self.depots = []
        self.drones = []

        self.airsim_visualizer = None

        print("🚀 AirSim 3D 드론 배달 시스템 초기화 완료")
        print(f"시뮬레이션 모드: {simulation_mode}")

    def load_data(self):
        print("\n=== 1단계: 건물 데이터 로드 ===")
        try:
            # JSON 형태의 건물 데이터 로드
            import json
            from pathlib import Path
            
            building_data_path = Path(__file__).parent.parent / "building_data.json"
            if building_data_path.exists():
                with open(building_data_path, 'r', encoding='utf-8') as f:
                    self.buildings = json.load(f)
                print(f"✅ 건물 데이터 로드 완료: {len(self.buildings)}개")
                return True
            else:
                print("❌ building_data.json 파일을 찾을 수 없습니다.")
                return False
        except Exception as e:
            print(f"❌ 데이터 로드 중 오류 발생: {e}")
            return False

    def setup_depots_and_drones(self):
        print("\n=== 2단계: Depot 및 드론 설정 ===")
        try:
            # JSON에서 depot 데이터 로드
            import json
            from pathlib import Path
            
            depot_data_path = Path(__file__).parent.parent / "depot_data.json"
            if depot_data_path.exists():
                with open(depot_data_path, 'r', encoding='utf-8') as f:
                    self.depots = json.load(f)
            else:
                # 기본 depot 설정 (fallback)
                self.depots = [
                    {'id': 'depot_1', 'longitude': 129.3250, 'latitude': 36.0145, 'height': 10.0},
                    {'id': 'depot_2', 'longitude': 129.3270, 'latitude': 36.0130, 'height': 10.0}
                ]
                print("⚠️ depot_data.json을 찾을 수 없어 기본 depot 설정을 사용합니다.")

            # JSON에서 드론 데이터 로드
            drone_data_path = Path(__file__).parent.parent / "drone_data.json"
            if drone_data_path.exists():
                with open(drone_data_path, 'r', encoding='utf-8') as f:
                    drone_data = json.load(f)
                # JSON 데이터를 시스템 형식으로 변환
                self.drones = []
                for drone in drone_data:
                    self.drones.append({
                        'id': drone['id'],
                        'depot_id': f"depot_{drone['id'].split('_')[2]}",  # drone_depot_1_1 -> depot_1
                        'current_lon': drone['longitude'],
                        'current_lat': drone['latitude'],
                        'current_height': drone['height'],
                        'battery': 100.0,
                        'current_payload': 0.0
                    })
            else:
                # 기본 드론 설정 (fallback)
                self.drones = []
                for depot in self.depots:
                    for i in range(3):
                        self.drones.append({
                            'id': f"drone_{depot['id']}_{i+1}",
                            'depot_id': depot['id'],
                            'current_lon': depot['longitude'],
                            'current_lat': depot['latitude'],
                            'current_height': 50.0,
                            'battery': 100.0,
                            'current_payload': 0.0
                        })
                print("⚠️ drone_data.json을 찾을 수 없어 기본 드론 설정을 사용합니다.")
            
            print(f"✅ Depot {len(self.depots)}개, 드론 {len(self.drones)}개 설정 완료")
            return True
        except Exception as e:
            print(f"❌ Depot 및 드론 설정 중 오류 발생: {e}")
            return False

    def run_airsim_simulation(self, duration_hours=1):
        """
        AirSim 3D 시각화 시뮬레이션 실행
        - 기존에 안정적으로 동작하는 핵심(드론별 전용 클라이언트/워커/큐, 이륙만 join, 이동 폴링)은 유지
        - RealtimeRouteManager + CoordinateConverter + OrderGenerator 를 주입하여
          임의 목적지 대신 '최적화된 경로'를 AirSim에서 실행
        """
        print(f"\n=== 3단계: AirSim 3D 시각화 시뮬레이션 실행 ({duration_hours}시간) ===")
        try:
            self.airsim_visualizer = AirSimVisualizer(
                depots=self.depots,
                drones=self.drones,
                building_data=self.buildings
            )

            if not self.airsim_visualizer.connect_to_airsim():
                print("❌ AirSim 서버 연결 실패. 프로그램을 종료합니다.")
                return

            if not self.airsim_visualizer.setup_simulation_environment():
                print("❌ AirSim 환경 설정 실패. 프로그램을 종료합니다.")
                return

            # 🔗 RouteManager 파이프라인 주입 (좌표 변환기 + 경로 관리자 + 주문 생성기)
            #    - settings.json의 OriginGeopoint와 동일해야 함
            converter = CoordinateConverter(base_lat=36.0139, base_lon=129.3261)

            route_manager = RealtimeRouteManager(
                depots=self.depots,
                drones=self.drones,
                optimization_target='cost',
                speed_mps=15.0,
                converter=converter
            )

            order_generator = OrderGenerator(
                building_data=self.buildings,
                prob_per_tick=0.25,     # tick당 주문 생성 확률 (필요시 조절)
                rng_seed=42,
                default_height_m=30.0
            )

            self.airsim_visualizer.attach_route_stack(
                route_manager=route_manager,
                converter=converter,
                order_generator=order_generator
            )
            print("🔗 RouteManager/OrderGenerator/CoordinateConverter 연동 완료 (랜덤 이동 → 경로 최적화)")

            # 시뮬레이션 시작
            if self.airsim_visualizer.start_simulation(duration_hours):
                print(f"🎬 AirSim 3D 시뮬레이션이 {duration_hours}시간 동안 자동으로 실행됩니다.")
                print("💡 강제로 중지하려면 터미널에서 Ctrl+C 를 누르세요.")
                try:
                    simulation_seconds = int(duration_hours * 3600)
                    time.sleep(simulation_seconds)
                    print("\n✅ 예정된 시뮬레이션 시간이 종료되었습니다.")
                except KeyboardInterrupt:
                    print("\n⏹️ 사용자에 의해 시뮬레이션이 중단되었습니다.")
                finally:
                    self.airsim_visualizer.stop_simulation()

                final_status = self.airsim_visualizer.get_simulation_status()
                viz_status = self.airsim_visualizer.get_visualization_status()
                print(f"\n🏁 최종 시뮬레이션 상태: {final_status}")
                print(f"🎨 시각화 상태: {viz_status}")
                return final_status
            else:
                print("❌ AirSim 시뮬레이션 시작 실패. 프로그램을 종료합니다.")
                return

        except Exception as e:
            print(f"❌ AirSim 시뮬레이션 실행 중 오류 발생: {e}")
            return None

    def get_system_status(self):
        return {
            'buildings_loaded': self.buildings is not None,
            'total_buildings': len(self.buildings) if self.buildings is not None else 0,
            'total_depots': len(self.depots),
            'total_drones': len(self.drones),
            'simulation_mode': self.simulation_mode
        }

    def print_system_info(self):
        print("\n📊 시스템 정보")
        print("=" * 40)

        st = self.get_system_status()
        print(f"🏢 건물 데이터: {'✅ 로드됨' if st['buildings_loaded'] else '❌ 로드 안됨'}")
        print(f"   총 건물 수: {st['total_buildings']}개")
        print(f"🏭 Depot 수: {st['total_depots']}개")
        print(f"🚁 드론 수: {st['total_drones']}개")
        print(f"🎮 시뮬레이션 모드: {st['simulation_mode']}")

        if self.depots:
            print("\n🏢 Depot 정보:")
            for depot in self.depots:
                print(f"  {depot['id']}: ({depot['longitude']:.4f}, {depot['latitude']:.4f})")

        if self.drones:
            print("\n🚁 드론 정보:")
            depot_drones = {}
            for d in self.drones:
                depot_drones[d['depot_id']] = depot_drones.get(d['depot_id'], 0) + 1
            for depot, count in depot_drones.items():
                print(f"  {depot}: {count}개 드론")

    def cleanup(self):
        try:
            if self.airsim_visualizer:
                self.airsim_visualizer.cleanup()
            print("🧹 리소스 정리 완료")
        except Exception as e:
            print(f"⚠️ 정리 중 오류: {e}")

    def __del__(self):
        self.cleanup()


def main():
    print("🚀 AirSim 3D 드론 배달 시스템 시작")
    print("=" * 60)

    system = DroneDeliverySystem(simulation_mode='realtime')

    if not system.load_data():
        print("❌ 데이터 로드 실패로 시스템을 종료합니다.")
        return

    if not system.setup_depots_and_drones():
        print("❌ Depot 및 드론 설정 실패로 시스템을 종료합니다.")
        return

    system.print_system_info()

    print("\n🎬 AirSim 3D 시각화 시뮬레이션을 선택하세요:")
    print("1. AirSim 3D 시각화 시뮬레이션 (1시간)")
    print("2. AirSim 3D 시각화 시뮬레이션 (2시간)")
    print("3. 종료")

    try:
        while True:
            choice = input("\n선택 (1-3): ").strip()
            if choice == '1':
                print("\n" + "=" * 60)
                system.run_airsim_simulation(duration_hours=1)
                print("\n🎉 AirSim 3D 시각화 시뮬레이션 완료!")
                break
            elif choice == '2':
                print("\n" + "=" * 60)
                system.run_airsim_simulation(duration_hours=2)
                print("\n🎉 AirSim 3D 시각화 시뮬레이션 완료!")
                break
            elif choice == '3':
                print("👋 프로그램을 종료합니다.")
                break
            else:
                print("❌ 잘못된 선택입니다. 1-3 중에서 선택해주세요.")
    except KeyboardInterrupt:
        print("\n\n⏹️ 사용자에 의해 프로그램이 중단되었습니다.")
    finally:
        system.cleanup()


if __name__ == "__main__":
    main()
