"""
드론 배달 시스템 테스트 스크립트
"""

import sys
import os
from pathlib import Path

# 프로젝트 루트 경로 추가
sys.path.append(str(Path(__file__).parent))

from algorithm.main import DroneDeliverySystem

def test_cost_optimization():
    """
    비용 최적화 테스트
    """
    print("=== 비용 최적화 테스트 ===")
    
    try:
        # 비용 최적화로 시스템 실행
        system = DroneDeliverySystem(optimization_target='cost')
        success = system.run()
        
        if success:
            print("✅ 비용 최적화 테스트 성공!")
            return True
        else:
            print("❌ 비용 최적화 테스트 실패!")
            return False
            
    except Exception as e:
        print(f"❌ 비용 최적화 테스트 중 오류 발생: {e}")
        return False

def test_time_optimization():
    """
    시간 최적화 테스트
    """
    print("=== 시간 최적화 테스트 ===")
    
    try:
        # 시간 최적화로 시스템 실행
        system = DroneDeliverySystem(optimization_target='time')
        success = system.run()
        
        if success:
            print("✅ 시간 최적화 테스트 성공!")
            return True
        else:
            print("❌ 시간 최적화 테스트 실패!")
            return False
            
    except Exception as e:
        print(f"❌ 시간 최적화 테스트 중 오류 발생: {e}")
        return False

def test_depot_planner():
    """
    Depot Planner 테스트
    """
    print("=== Depot Planner 테스트 ===")
    
    try:
        from algorithm.depot.depot_planner import DepotPlanner
        from algorithm.utils.data_loader import BuildingDataLoader
        
        # 데이터 로드
        data_loader = BuildingDataLoader()
        buildings = data_loader.load_building_data('map/postech_map/merged_maps/postech_buildings_real_height.csv')
        
        if buildings is None:
            print("❌ 건물 데이터 로드 실패!")
            return False
        
        # Depot Planner 테스트
        planner = DepotPlanner(buildings)
        
        # 지도 특성 분석
        characteristics = planner.analyze_map_characteristics()
        print(f"지도 특성: {characteristics}")
        
        # Depot 최적화
        depots = planner.optimize_depot_locations(method='auto')
        
        if depots:
            print(f"✅ Depot Planner 테스트 성공! {len(depots)}개 depot 생성")
            return True
        else:
            print("❌ Depot Planner 테스트 실패!")
            return False
            
    except Exception as e:
        print(f"❌ Depot Planner 테스트 중 오류 발생: {e}")
        return False

def test_drone_allocator():
    """
    Drone Allocator 테스트
    """
    print("=== Drone Allocator 테스트 ===")
    
    try:
        from algorithm.depot.drone_allocator import DroneAllocator
        from algorithm.utils.data_loader import BuildingDataLoader
        
        # 데이터 로드
        data_loader = BuildingDataLoader()
        buildings = data_loader.load_building_data('map/postech_map/merged_maps/postech_buildings_real_height.csv')
        
        if buildings is None:
            print("❌ 건물 데이터 로드 실패!")
            return False
        
        # 임시 depot 생성
        depots = [
            {'id': 'depot_1', 'latitude': 36.0123, 'longitude': 127.3456},
            {'id': 'depot_2', 'latitude': 36.0234, 'longitude': 127.3567}
        ]
        
        # Drone Allocator 테스트
        allocator = DroneAllocator(depots, buildings)
        
        # 최적 드론 수 계산
        optimal_drones = allocator.calculate_optimal_drone_count(method='workload_based')
        print(f"최적 드론 수: {optimal_drones}")
        
        # 드론 할당
        allocation = allocator.allocate_drones_to_depots(optimal_drones, method='workload_based')
        print(f"드론 할당: {allocation}")
        
        print("✅ Drone Allocator 테스트 성공!")
        return True
        
    except Exception as e:
        print(f"❌ Drone Allocator 테스트 중 오류 발생: {e}")
        return False

def test_routing_algorithms():
    """
    라우팅 알고리즘 테스트
    """
    print("=== 라우팅 알고리즘 테스트 ===")
    
    try:
        from algorithm.routing.matheuristic.clarke_wright import ClarkeWrightSolver
        from algorithm.config import DRONE_CONFIG
        
        # 테스트 데이터 (실제 좌표 사용)
        depots = [
            {'id': 'depot_1', 'latitude': 380000.0, 'longitude': 230000.0}
        ]
        
        delivery_requests = [
            {
                'id': 'req_1',
                'restaurant_location': {'latitude': 380100.0, 'longitude': 230100.0},
                'customer_location': {'latitude': 380200.0, 'longitude': 230200.0},
                'weight': 1.0
            },
            {
                'id': 'req_2',
                'restaurant_location': {'latitude': 380300.0, 'longitude': 230300.0},
                'customer_location': {'latitude': 380400.0, 'longitude': 230400.0},
                'weight': 1.5
            }
        ]
        
        # Clarke-Wright 알고리즘 테스트
        solver = ClarkeWrightSolver(depots, delivery_requests, DRONE_CONFIG)
        routes = solver.solve()
        
        if routes:
            print(f"✅ 라우팅 알고리즘 테스트 성공! {len(routes)}개 경로 생성")
            return True
        else:
            print("❌ 라우팅 알고리즘 테스트 실패!")
            return False
            
    except Exception as e:
        print(f"❌ 라우팅 알고리즘 테스트 중 오류 발생: {e}")
        return False

def test_realtime_simulation():
    """
    실시간 시뮬레이션 테스트
    """
    print("=== 실시간 시뮬레이션 테스트 ===")
    
    try:
        from algorithm.simulation.realtime_simulator import RealtimeDroneSimulator
        from algorithm.config import SIMULATION_CONFIG
        
        # 테스트 데이터
        depots = [
            {'id': 'depot_1', 'latitude': 380000.0, 'longitude': 230000.0}
        ]
        
        routes = [
            {
                'depot_id': 'depot_1',
                'requests': [
                    {
                        'id': 'req_1',
                        'restaurant_location': {'latitude': 380100.0, 'longitude': 230100.0},
                        'customer_location': {'latitude': 380200.0, 'longitude': 230200.0},
                        'weight': 1.0
                    }
                ]
            }
        ]
        
        # 간단한 건물 데이터
        building_data = {
            'latitude': [380000.0, 380100.0, 380200.0],
            'longitude': [230000.0, 230100.0, 230200.0]
        }
        
        # 실시간 시뮬레이터 테스트 (애니메이션 없이)
        simulator = RealtimeDroneSimulator(depots, routes, building_data, SIMULATION_CONFIG)
        simulator.setup_dynamic_requests(generation_rate=0.1, max_requests_per_batch=2)
        
        # 짧은 시뮬레이션 실행 (1분, 애니메이션 없이)
        results = simulator.run_simulation(duration_minutes=1, realtime_factor=10.0, enable_animation=False)
        
        if results:
            print(f"✅ 실시간 시뮬레이션 테스트 성공!")
            print(f"  - 총 요청: {results['total_requests']}개")
            print(f"  - 동적 요청: {results['dynamic_requests']}개")
            print(f"  - 완료된 배달: {results['completed_deliveries']}개")
            return True
        else:
            print("❌ 실시간 시뮬레이션 테스트 실패!")
            return False
            
    except Exception as e:
        print(f"❌ 실시간 시뮬레이션 테스트 중 오류 발생: {e}")
        return False

def main():
    """
    메인 테스트 함수
    """
    print("드론 배달 시스템 테스트 시작")
    print("=" * 50)
    
    test_results = []
    
    # 개별 모듈 테스트
    test_results.append(("Depot Planner", test_depot_planner()))
    test_results.append(("Drone Allocator", test_drone_allocator()))
    test_results.append(("Routing Algorithms", test_routing_algorithms()))
    test_results.append(("Realtime Simulation", test_realtime_simulation()))
    
    # 최적화 목표별 테스트
    test_results.append(("Cost Optimization", test_cost_optimization()))
    test_results.append(("Time Optimization", test_time_optimization()))
    
    # 테스트 결과 출력
    print("\n=== 개별 모듈 테스트 결과 ===")
    
    for test_name, result in test_results:
        status = "✅ 성공" if result else "❌ 실패"
        print(f"{test_name}: {status}")
    
    success_count = sum(1 for _, result in test_results if result)
    total_count = len(test_results)
    
    print(f"\n테스트 결과: {success_count}/{total_count} 성공")
    
    if success_count == total_count:
        print("🎉 모든 테스트가 성공했습니다!")
    else:
        print("⚠️ 일부 테스트가 실패했습니다.")
    
    # 실시간 시뮬레이션 안내
    print("\n" + "=" * 50)
    print("실시간 동적 시뮬레이션을 실행하려면:")
    print("python algorithm/main.py")
    print("를 실행하고 최적화 목표를 선택하세요.")
    print("애니메이션과 함께 실시간으로 배달 요청이 생성되는 것을 확인할 수 있습니다.")

if __name__ == "__main__":
    main() 