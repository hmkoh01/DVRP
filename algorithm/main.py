"""
드론 음식 배달 시스템 메인 실행 파일
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 프로젝트 루트 경로 추가
sys.path.append(str(Path(__file__).parent.parent))

from algorithm.config import *
from algorithm.utils.data_loader import BuildingDataLoader

from algorithm.utils.delivery_generator import DeliveryRequestGenerator
from algorithm.utils.visualization import VisualizationTool
from algorithm.depot.depot_planner import DepotPlanner
from algorithm.depot.drone_allocator import DroneAllocator
from algorithm.simulation.realtime_simulator import RealtimeDroneSimulator

class DroneDeliverySystem:
    """
    드론 음식 배달 시스템 메인 클래스
    """
    
    def __init__(self, auto_mode=True, optimization_target='auto', algorithm='auto'):
        """
        초기화
        """
        self.auto_mode = auto_mode
        self.optimization_target = optimization_target
        self.algorithm = algorithm
        
        # 데이터 로더 초기화
        self.data_loader = BuildingDataLoader()
        self.delivery_generator = None
        self.visualization_tool = None
        
        # 새로운 모듈들
        self.depot_planner = None
        self.drone_allocator = None
        
        # 시스템 상태
        self.buildings = None
        self.depots = []
        self.delivery_requests = []
        self.drones = []
        self.routes = []
        self.map_characteristics = {}
        
        print(f"드론 배달 시스템 초기화 완료")
        print(f"자동 모드: {auto_mode}")
        if not auto_mode:
            print(f"최적화 목표: {optimization_target}")
            print(f"선택된 알고리즘: {algorithm}")
    
    def load_data(self):
        """
        건물 데이터 로드
        """
        print("\n=== 1단계: 건물 데이터 로드 ===")
        
        try:
            # 설정된 경로에서 데이터 로드
            data_path = SYSTEM_CONFIG['building_data_path']
            self.buildings = self.data_loader.load_building_data(data_path)
            
            if self.buildings is None:
                print("건물 데이터 로드에 실패했습니다.")
                return False
            
            # 유틸리티 객체들 초기화
            self.delivery_generator = DeliveryRequestGenerator(self.data_loader)
            self.visualization_tool = VisualizationTool(self.data_loader)
            
            print("건물 데이터 로드 완료!")
            return True
            
        except Exception as e:
            print(f"데이터 로드 중 오류 발생: {e}")
            return False
    
    def analyze_map_and_select_algorithm(self):
        """
        지도 특성 분석 및 최적 알고리즘 선택
        """
        print("\n=== 2단계: 지도 특성 분석 및 알고리즘 선택 ===")
        
        # 새로운 depot planner 사용
        self.depot_planner = DepotPlanner(self.buildings)
        
        # 지도 특성 분석
        self.map_characteristics = self.depot_planner.analyze_map_characteristics()
        
        if self.auto_mode:
            # 자동으로 최적화 목표 선택
            self.optimization_target = self._select_optimal_target()
            
            # 자동으로 알고리즘 선택
            self.algorithm = self._select_optimal_algorithm()
            
            print(f"자동 선택된 최적화 목표: {self.optimization_target}")
            print(f"자동 선택된 알고리즘: {self.algorithm}")
        
        return True
    
    def _select_optimal_target(self):
        """
        지도 특성에 따른 최적화 목표 선택
        """
        building_density = self.map_characteristics['building_density']
        area_coverage = self.map_characteristics['area_coverage']
        
        # 고밀도 지역이거나 넓은 지역이면 시간 최적화
        if building_density > 50 or area_coverage > 5:
            return 'time'
        else:
            return 'cost'
    
    def _select_optimal_algorithm(self):
        """
        지도 특성에 따른 최적 알고리즘 선택
        """
        building_density = self.map_characteristics['building_density']
        area_coverage = self.map_characteristics['area_coverage']
        total_buildings = self.map_characteristics['total_buildings']
        
        # 건물 수가 많고 복잡한 지역
        if total_buildings > 1000:
            return 'genetic'
        # 중간 규모 지역
        elif total_buildings > 500:
            return 'ant_colony'
        # 작은 규모 지역
        elif total_buildings > 100:
            return 'particle_swarm'
        # 매우 작은 지역
        else:
            return 'clarke_wright'
    
    def optimize_depots(self):
        """
        Depot 최적화 (새로운 방식)
        """
        print("\n=== 3단계: Depot 최적화 ===")
        
        try:
            # 새로운 depot planner 사용
            self.depots = self.depot_planner.optimize_depot_locations(method='auto')
            
            if self.depots:
                print("Depot 최적화 완료!")
                
                # Depot 통계 출력
                stats = self.depot_planner.get_depot_statistics()
                print(f"총 Depot 수: {stats['total_depots']}")
                print(f"총 커버리지: {stats['total_coverage']}개 건물")
                print(f"평균 커버리지 반경: {stats['avg_coverage_radius']:.4f} km")
                
                return True
            else:
                print("Depot 최적화에 실패했습니다.")
                return False
                
        except Exception as e:
            print(f"Depot 최적화 중 오류 발생: {e}")
            return False
    
    def calculate_optimal_drone_count(self):
        """
        최적 드론 수 계산
        """
        print("\n=== 4단계: 최적 드론 수 계산 ===")
        
        try:
            # 새로운 drone allocator 사용
            self.drone_allocator = DroneAllocator(self.depots, self.buildings)
            
            # 최적 드론 수 계산
            optimal_drones = self.drone_allocator.calculate_optimal_drone_count(method='workload_based')
            
            if optimal_drones:
                print(f"계산된 최적 드론 수: {optimal_drones}")
                
                # 드론 할당
                drone_allocation = self.drone_allocator.allocate_drones_to_depots(
                    optimal_drones, method='workload_based'
                )
                
                # 할당 통계
                allocation_stats = self.drone_allocator.get_allocation_statistics(drone_allocation)
                print(f"Depot별 드론 할당: {drone_allocation}")
                print(f"평균 드론/Depot: {allocation_stats['avg_drones_per_depot']:.1f}")
                
                return optimal_drones
            else:
                print("드론 수 계산에 실패했습니다.")
                return 10  # 기본값
                
        except Exception as e:
            print(f"드론 수 계산 중 오류 발생: {e}")
            return 10  # 기본값

    def generate_delivery_requests(self, num_requests=50):
        """
        배달 요청 생성
        """
        print("\n=== 5단계: 배달 요청 생성 ===")
        
        try:
            # 지도 특성에 따른 요청 수 조정
            if self.map_characteristics:
                building_density = self.map_characteristics['building_density']
                if building_density > 100:
                    num_requests = min(100, num_requests * 2)
                elif building_density < 10:
                    num_requests = max(20, num_requests // 2)
            
            self.delivery_requests = self.delivery_generator.generate_requests(
                num_requests=num_requests
            )
            
            if self.delivery_requests:
                print(f"배달 요청 생성 완료: {len(self.delivery_requests)}개")
                return True
            else:
                print("배달 요청 생성에 실패했습니다.")
                return False
                
        except Exception as e:
            print(f"배달 요청 생성 중 오류 발생: {e}")
            return False
    
    def create_drones(self, num_drones=None):
        """
        드론 생성
        """
        print("\n=== 6단계: 드론 생성 ===")
        
        try:
            if num_drones is None:
                num_drones = self.calculate_optimal_drone_count()
            
            self.drones = []
            
            # 각 depot별로 드론 생성
            for depot in self.depots:
                depot_drones = max(1, num_drones // len(self.depots))
                
                for i in range(depot_drones):
                    drone = {
                        'id': f"drone_{depot['id']}_{i+1}",
                        'depot_id': depot['id'],
                        'current_lat': depot['latitude'],
                        'current_lon': depot['longitude'],
                        'status': 'idle',
                        'battery': 100,
                        'max_speed': DRONE_CONFIG['max_speed'],
                        'max_payload': DRONE_CONFIG['max_payload'],
                        'battery_capacity': DRONE_CONFIG['battery_capacity']
                    }
                    self.drones.append(drone)
            
            print(f"드론 생성 완료: {len(self.drones)}개")
            return True
            
        except Exception as e:
            print(f"드론 생성 중 오류 발생: {e}")
            return False
    
    def optimize_routes(self):
        """
        경로 최적화 (알고리즘 자동 선택)
        """
        print("\n=== 7단계: 경로 최적화 ===")
        
        try:
            # 선택된 알고리즘에 따른 경로 최적화
            if self.algorithm in ['genetic', 'ant_colony', 'particle_swarm', 'simulated_annealing']:
                # Metaheuristic 알고리즘
                if self.algorithm == 'genetic':
                    from algorithm.routing.metaheuristic.genetic_algorithm import GeneticAlgorithm
                    solver = GeneticAlgorithm(optimization_target=self.optimization_target)
                    solver.delivery_requests = self.delivery_requests
                    solver.depots = self.depots
                    solver.drones = self.drones
                elif self.algorithm == 'ant_colony':
                    from algorithm.routing.metaheuristic.ant_colony import AntColonyOptimizer
                    solver = AntColonyOptimizer(self.depots, self.delivery_requests, DRONE_CONFIG)
                elif self.algorithm == 'particle_swarm':
                    from algorithm.routing.metaheuristic.particle_swarm import ParticleSwarmOptimizer
                    solver = ParticleSwarmOptimizer(self.depots, self.delivery_requests, DRONE_CONFIG)
                elif self.algorithm == 'simulated_annealing':
                    from algorithm.routing.metaheuristic.simulated_annealing import SimulatedAnnealingOptimizer
                    solver = SimulatedAnnealingOptimizer(self.depots, self.delivery_requests, DRONE_CONFIG)
                
            elif self.algorithm in ['q_learning', 'dqn', 'actor_critic']:
                # Reinforcement Learning 알고리즘
                if self.algorithm == 'q_learning':
                    from algorithm.routing.RL.q_learning import QLearningOptimizer
                    solver = QLearningOptimizer(self.depots, self.delivery_requests, DRONE_CONFIG)
                elif self.algorithm == 'dqn':
                    from algorithm.routing.RL.deep_q_network import DQNOptimizer
                    solver = DQNOptimizer(self.depots, self.delivery_requests, DRONE_CONFIG)
                elif self.algorithm == 'actor_critic':
                    from algorithm.routing.RL.actor_critic import ActorCriticOptimizer
                    solver = ActorCriticOptimizer(self.depots, self.delivery_requests, DRONE_CONFIG)
            
            elif self.algorithm in ['clarke_wright', 'savings', 'sweep']:
                # Matheuristic 알고리즘
                if self.algorithm == 'clarke_wright':
                    from algorithm.routing.matheuristic.clarke_wright import ClarkeWrightSolver
                    solver = ClarkeWrightSolver(self.depots, self.delivery_requests, DRONE_CONFIG)
                elif self.algorithm == 'savings':
                    from algorithm.routing.matheuristic.savings_algorithm import SavingsAlgorithm
                    solver = SavingsAlgorithm(self.depots, self.delivery_requests, DRONE_CONFIG)
                elif self.algorithm == 'sweep':
                    from algorithm.routing.matheuristic.sweep_algorithm import SweepAlgorithm
                    solver = SweepAlgorithm(self.depots, self.delivery_requests, DRONE_CONFIG)
            
            else:
                print(f"지원하지 않는 알고리즘: {self.algorithm}")
                return False
            
            # 경로 최적화 실행
            self.routes = solver.solve()
            
            if self.routes:
                print(f"경로 최적화 완료: {len(self.routes)}개 경로")
                return True
            else:
                print("경로 최적화에 실패했습니다.")
                return False
                
        except Exception as e:
            print(f"경로 최적화 중 오류 발생: {e}")
            return False
    
    def run_simulation(self):
        """
        시뮬레이션 실행 (실시간 동적 요청 + 애니메이션)
        """
        print("\n=== 8단계: 실시간 동적 시뮬레이션 실행 ===")
        
        try:
            # 실시간 시뮬레이터 사용
            simulator = RealtimeDroneSimulator(
                self.depots, 
                self.routes, 
                self.buildings, 
                SIMULATION_CONFIG
            )
            
            # 동적 요청 생성을 위한 설정
            simulator.setup_dynamic_requests(
                generation_rate=0.3,  # 분당 0.3개 요청 (18개/시간)
                max_requests_per_batch=5,  # 한 번에 최대 5개 요청
                time_window_minutes=60  # 1시간 윈도우
            )
            
            # 시뮬레이션 실행 (60분, 실시간 속도)
            simulation_results = simulator.run_simulation(
                duration_minutes=60, 
                realtime_factor=1.0,  # 실시간 속도
                enable_animation=True  # 애니메이션 활성화
            )
            
            if simulation_results:
                print("실시간 동적 시뮬레이션 완료!")
                print(f"총 생성된 요청: {simulation_results['total_requests']}개")
                print(f"완료된 배달: {simulation_results['completed_deliveries']}개")
                print(f"성공률: {simulation_results['success_rate']:.2%}")
                print(f"평균 배달 시간: {simulation_results['avg_delivery_time']:.1f}초")
                print(f"드론 활용률: {simulation_results['drone_utilization']:.2%}")
                print(f"실시간 요청 처리율: {simulation_results['dynamic_request_rate']:.2%}")
                
                return simulation_results
            else:
                print("시뮬레이션 실행에 실패했습니다.")
                return None
                
        except Exception as e:
            print(f"시뮬레이션 실행 중 오류 발생: {e}")
            return None
    
    def analyze_results(self, simulation_results):
        """
        결과 분석
        """
        print("\n=== 9단계: 결과 분석 ===")
        
        try:
            # 성능 분석기 사용
            from algorithm.simulation.performance_analyzer import PerformanceAnalyzer
            
            analyzer = PerformanceAnalyzer()
            metrics = analyzer.analyze_performance(simulation_results)
            
            print("결과 분석 완료!")
            print(f"총 비용: {metrics.get('total_cost', 0):.2f}원")
            print(f"총 거리: {metrics.get('total_distance', 0):.2f}km")
            print(f"평균 배달 시간: {metrics.get('avg_delivery_time', 0):.1f}분")
            
            return metrics
            
        except Exception as e:
            print(f"결과 분석 중 오류 발생: {e}")
            return {}
    
    def visualize_results(self, simulation_results=None):
        """
        결과 시각화
        """
        print("\n=== 10단계: 결과 시각화 ===")
        
        try:
            # 시각화 도구 사용
            if self.depots:
                self.visualization_tool.plot_depots(self.depots)
            
            if self.routes:
                self.visualization_tool.plot_drone_routes(self.routes, self.depots)
            
            if simulation_results:
                self.visualization_tool.plot_simulation_results(simulation_results)
            
            print("결과 시각화 완료!")
            
        except Exception as e:
            print(f"결과 시각화 중 오류 발생: {e}")
    
    def save_results(self, simulation_results=None):
        """
        결과 저장
        """
        print("\n=== 11단계: 결과 저장 ===")
        
        try:
            # 결과를 CSV 파일로 저장
            results_dir = Path("results")
            results_dir.mkdir(exist_ok=True)
            
            # 경로 정보 저장
            routes_data = []
            for i, route in enumerate(self.routes):
                routes_data.append({
                    'route_id': i+1,
                    'depot_id': route['depot_id'],
                    'num_requests': len(route['requests']),
                    'total_distance': route.get('total_distance', 0),
                    'total_time': route.get('total_time', 0)
                })
            
            routes_df = pd.DataFrame(routes_data)
            routes_df.to_csv(results_dir / "routes.csv", index=False)
            
            # 시뮬레이션 결과 저장
            if simulation_results:
                sim_df = pd.DataFrame([simulation_results])
                sim_df.to_csv(results_dir / "simulation_results.csv", index=False)
            
            print("결과 저장 완료!")
            
        except Exception as e:
            print(f"결과 저장 중 오류 발생: {e}")
    
    def run(self):
        """
        전체 시스템 실행
        """
        print("=" * 50)
        print("드론 음식 배달 시스템 시작")
        print("=" * 50)
        
        # 1. 데이터 로드
        if not self.load_data():
            return False
        
        # 2. 지도 분석 및 알고리즘 선택
        if not self.analyze_map_and_select_algorithm():
            return False
        
        # 3. Depot 최적화
        if not self.optimize_depots():
            return False
        
        # 4. 배달 요청 생성
        if not self.generate_delivery_requests():
            return False
        
        # 5. 드론 생성
        if not self.create_drones():
            return False
        
        # 6. 경로 최적화
        if not self.optimize_routes():
            return False
        
        # 7. 시뮬레이션 실행
        simulation_results = self.run_simulation()
        if simulation_results is None:
            return False
        
        # 8. 결과 분석
        metrics = self.analyze_results(simulation_results)
        
        # 9. 시각화
        self.visualize_results(simulation_results)
        
        # 10. 결과 저장
        self.save_results(simulation_results)
        
        print("\n" + "=" * 50)
        print("드론 음식 배달 시스템 완료!")
        print("=" * 50)
        
        return True

def main():
    """
    메인 함수
    """
    print("드론 음식 배달 시스템")
    print("-" * 30)
    
    # 자동 모드 선택
    print("실행 모드를 선택하세요:")
    print("1. 자동 모드 (지도 분석 후 최적 알고리즘 자동 선택)")
    print("2. 수동 모드 (사용자가 직접 선택)")
    
    while True:
        mode_choice = input("선택 (1 또는 2): ").strip()
        if mode_choice == '1':
            auto_mode = True
            break
        elif mode_choice == '2':
            auto_mode = False
            break
        else:
            print("잘못된 선택입니다. 1 또는 2를 입력하세요.")
    
    if auto_mode:
        # 자동 모드
        system = DroneDeliverySystem(auto_mode=True)
    else:
        # 수동 모드
        print("\n최적화 목표를 선택하세요:")
        print("1. cost (비용 최적화)")
        print("2. time (시간 최적화)")
        
        while True:
            target_choice = input("선택 (1 또는 2): ").strip()
            if target_choice == '1':
                optimization_target = 'cost'
                break
            elif target_choice == '2':
                optimization_target = 'time'
                break
            else:
                print("잘못된 선택입니다. 1 또는 2를 입력하세요.")
        
        print("\n알고리즘을 선택하세요:")
        print("Metaheuristic:")
        print("1. genetic (유전 알고리즘)")
        print("2. ant_colony (개미 군집 최적화)")
        print("3. particle_swarm (입자 군집 최적화)")
        print("4. simulated_annealing (시뮬레이티드 어닐링)")
        print("Reinforcement Learning:")
        print("5. q_learning (Q-Learning)")
        print("6. dqn (Deep Q-Network)")
        print("7. actor_critic (Actor-Critic)")
        print("Matheuristic:")
        print("8. clarke_wright (Clarke-Wright)")
        print("9. savings (Savings Algorithm)")
        print("10. sweep (Sweep Algorithm)")
        
        algorithm_map = {
            '1': 'genetic',
            '2': 'ant_colony',
            '3': 'particle_swarm',
            '4': 'simulated_annealing',
            '5': 'q_learning',
            '6': 'dqn',
            '7': 'actor_critic',
            '8': 'clarke_wright',
            '9': 'savings',
            '10': 'sweep'
        }
        
        while True:
            algo_choice = input("선택 (1-10): ").strip()
            if algo_choice in algorithm_map:
                algorithm = algorithm_map[algo_choice]
                break
            else:
                print("잘못된 선택입니다. 1-10 중에서 선택하세요.")
        
        system = DroneDeliverySystem(
            auto_mode=False,
            optimization_target=optimization_target,
            algorithm=algorithm
        )
    
    # 시스템 실행
    success = system.run()
    
    if success:
        print("\n시스템이 성공적으로 완료되었습니다!")
    else:
        print("\n시스템 실행 중 오류가 발생했습니다.")

if __name__ == "__main__":
    main() 