# 드론 음식 배달 시스템 (DVRP - Drone Vehicle Routing Problem)

POSTECH 지도 데이터를 기반으로 한 지능형 드론 음식 배달 시스템입니다. 지도 정보를 분석하여 최적의 depot 위치와 드론 수를 자동으로 설정하고, 다양한 라우팅 알고리즘을 통해 효율적인 배달 경로를 최적화합니다.

## 🚀 주요 기능

### 1. 지도 특성 분석 및 자동 알고리즘 선택
- **지도 분석**: 건물 밀도, 면적, 배달 핫스팟 등 지도 특성 자동 분석
- **알고리즘 자동 선택**: 지도 특성에 따른 최적 라우팅 알고리즘 자동 선택
- **최적화 목표 자동 설정**: 비용 최적화 vs 시간 최적화 자동 결정

### 2. Depot 최적화
- **K-means 클러스터링**: 밀도 기반 depot 위치 최적화
- **그리드 기반 최적화**: 넓은 지역을 위한 그리드 기반 depot 배치
- **자동 방법 선택**: 지도 특성에 따른 최적 depot 최적화 방법 자동 선택

### 3. 드론 할당 최적화
- **작업량 기반 계산**: 예상 배달 요청 수에 따른 최적 드론 수 계산
- **커버리지 기반 계산**: 면적 커버리지를 고려한 드론 수 계산
- **Depot별 할당**: 각 depot의 작업량에 따른 드론 균형 할당

### 4. 다양한 라우팅 알고리즘
#### Metaheuristic 알고리즘
- **유전 알고리즘 (Genetic Algorithm)**: 복잡한 대규모 지역에 적합
- **개미 군집 최적화 (Ant Colony Optimization)**: 중간 규모 지역에 적합
- **입자 군집 최적화 (Particle Swarm Optimization)**: 작은 규모 지역에 적합
- **시뮬레이티드 어닐링 (Simulated Annealing)**: 전역 최적해 탐색

#### Reinforcement Learning 알고리즘
- **Q-Learning**: 기본 강화학습 기반 경로 최적화
- **Deep Q-Network (DQN)**: 딥러닝 기반 강화학습
- **Actor-Critic**: 정책 기반 강화학습

#### Matheuristic 알고리즘
- **Clarke-Wright Savings**: 전통적인 VRP 알고리즘
- **Savings Algorithm**: 절약 기반 경로 최적화
- **Sweep Algorithm**: 각도 기반 경로 생성

### 5. 실시간 시뮬레이션
- **실시간 애니메이션**: 드론의 배달 과정을 실시간으로 시각화
- **배터리 관리**: 드론 배터리 상태 실시간 모니터링
- **충돌 회피**: 안전한 비행 경로 보장
- **성능 지표**: 배달 성공률, 평균 배달 시간, 드론 활용률 등

## 📁 프로젝트 구조

```
DVRP_UGRP/
├── algorithm/
│   ├── depot/                    # Depot 최적화 모듈
│   │   ├── depot_planner.py     # Depot 위치 최적화
│   │   └── drone_allocator.py   # 드론 할당 최적화
│   ├── routing/                  # 라우팅 알고리즘
│   │   ├── metaheuristic/       # 메타휴리스틱 알고리즘
│   │   ├── RL/                  # 강화학습 알고리즘
│   │   └── matheuristic/        # 수학적 휴리스틱 알고리즘
│   ├── simulation/              # 시뮬레이션 모듈
│   │   ├── realtime_simulator.py # 실시간 시뮬레이션
│   │   └── performance_analyzer.py # 성능 분석
│   ├── utils/                   # 유틸리티 모듈
│   ├── config.py               # 시스템 설정
│   └── main.py                 # 메인 실행 파일
├── map/                        # 지도 데이터
├── results/                    # 결과 저장 폴더
├── test_system.py             # 시스템 테스트
└── README.md                  # 프로젝트 설명서
```

## 🛠️ 설치 및 실행

### 1. 환경 설정
```bash
# 가상환경 생성 (권장)
python -m venv dvrp_env
source dvrp_env/bin/activate  # Linux/Mac
# 또는
dvrp_env\Scripts\activate     # Windows

# 필요한 패키지 설치
pip install numpy pandas matplotlib scikit-learn plotly folium
```

### 2. 시스템 실행

#### 자동 모드 (권장)
```bash
python algorithm/main.py
# 선택: 1 (자동 모드)
```

#### 수동 모드
```bash
python algorithm/main.py
# 선택: 2 (수동 모드)
# 최적화 목표 선택: 1 (비용) 또는 2 (시간)
# 알고리즘 선택: 1-10 (원하는 알고리즘)
```

### 3. 시스템 테스트
```bash
python test_system.py
```

## 🎯 사용 예시

### 자동 모드 실행
```python
from algorithm.main import DroneDeliverySystem

# 자동 모드로 시스템 실행
system = DroneDeliverySystem(auto_mode=True)
success = system.run()

if success:
    print("시스템 실행 완료!")
```

### 수동 모드 실행
```python
from algorithm.main import DroneDeliverySystem

# 수동 모드로 시스템 실행 (Clarke-Wright 알고리즘)
system = DroneDeliverySystem(
    auto_mode=False,
    optimization_target='cost',
    algorithm='clarke_wright'
)
success = system.run()
```

### 개별 모듈 사용
```python
from algorithm.depot.depot_planner import DepotPlanner
from algorithm.depot.drone_allocator import DroneAllocator
from algorithm.routing.matheuristic.clarke_wright import ClarkeWrightSolver

# Depot 최적화
planner = DepotPlanner(building_data)
depots = planner.optimize_depot_locations(method='auto')

# 드론 할당
allocator = DroneAllocator(depots, building_data)
optimal_drones = allocator.calculate_optimal_drone_count()

# 경로 최적화
solver = ClarkeWrightSolver(depots, delivery_requests, drone_config)
routes = solver.solve()
```

## 📊 성능 지표

시스템은 다음과 같은 성능 지표를 제공합니다:

- **배달 성공률**: 전체 배달 요청 중 성공한 비율
- **평균 배달 시간**: 배달 완료까지 걸린 평균 시간
- **드론 활용률**: 전체 드론 중 실제 작업 중인 비율
- **총 비용**: 연료비, 운영비, 페널티 등을 포함한 총 비용
- **에너지 효율성**: 배터리 사용량 대비 배달 성과

## 🔧 설정 옵션

`algorithm/config.py` 파일에서 다음 설정을 조정할 수 있습니다:

- **드론 설정**: 최대 속도, 배터리 용량, 최대 적재량 등
- **배달 설정**: 요청 생성률, 대기 시간, 배달 시간 창 등
- **알고리즘 설정**: 각 알고리즘의 하이퍼파라미터
- **비용 설정**: 연료비, 운영비, 페널티 등
- **시뮬레이션 설정**: 시간 단위, 시뮬레이션 기간 등

## 🎨 시각화

시스템은 다음과 같은 시각화 기능을 제공합니다:

- **3D 건물 시각화**: POSTECH 건물들의 3D 렌더링
- **실시간 드론 애니메이션**: 드론의 배달 과정 실시간 시각화
- **경로 시각화**: 최적화된 배달 경로 표시
- **성능 대시보드**: 실시간 성능 지표 표시

## 🤝 기여하기

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 `LICENSE` 파일을 참조하세요.

## 👥 팀원

- **UGRP 2025 팀**: 드론 음식 배달 시스템 개발
- **POSTECH**: 지도 데이터 제공 및 기술 지원

## 📞 문의

프로젝트에 대한 문의사항이 있으시면 이슈를 생성해 주세요.

---

**참고**: 이 시스템은 연구 목적으로 개발되었으며, 실제 드론 배달 서비스에 적용하기 전에 추가적인 안전 검증이 필요합니다. 