"""
드론 음식 배달 시스템 설정 파일
"""

# 시스템 설정
SYSTEM_CONFIG = {
    'optimization_targets': ['cost', 'time'],
    'algorithms': {
        'metaheuristic': ['genetic', 'ant_colony', 'particle_swarm', 'simulated_annealing'],
        'RL': ['q_learning', 'dqn', 'actor_critic']
    },
    'building_data_path': 'map/postech_map/merged_maps/postech_buildings_real_height.csv'
}

# 드론 설정
DRONE_CONFIG = {
    'max_speed': 15.0,        # m/s
    'max_payload': 2.0,       # kg
    'battery_capacity': 30.0, # minutes
    'safety_height': 10.0,    # m (건물 위 안전 거리)
    'takeoff_landing_time': 1.0,  # minutes
    'energy_consumption_rate': 0.1,  # kWh/km
    'max_flight_height': 100.0,  # m
    'min_flight_height': 20.0,   # m
}

# 배달 설정
DELIVERY_CONFIG = {
    'request_generation_rate': 0.1,  # requests per minute
    'max_waiting_time': 30.0,        # minutes
    'restaurant_types': ['근린생활시설', '기타시설'],  # 식당으로 사용할 건물 용도
    'residential_types': ['주택', '근린생활시설'],     # 배달 목적지로 사용할 건물 용도
    'delivery_time_window': 15.0,    # minutes (배달 완료 허용 시간)
    'order_preparation_time': 5.0,   # minutes (주문 준비 시간)
}

# Depot 설정
DEPOT_CONFIG = {
    'min_depots': 1,
    'max_depots': 5,
    'depot_capacity': 10,  # 각 depot당 최대 드론 수
    'depot_optimization_method': 'kmeans',  # 'kmeans', 'genetic', 'manual'
    'depot_height': 50.0,  # m (depot 고도)
}

# 알고리즘별 설정
ALGORITHM_CONFIG = {
    'genetic': {
        'population_size': 50,
        'generations': 100,
        'mutation_rate': 0.1,
        'crossover_rate': 0.8,
        'elite_size': 5
    },
    'ant_colony': {
        'n_ants': 30,
        'n_iterations': 50,
        'evaporation_rate': 0.1,
        'alpha': 1.0,  # 페로몬 중요도
        'beta': 2.0    # 거리 중요도
    },
    'particle_swarm': {
        'n_particles': 30,
        'n_iterations': 50,
        'w': 0.7,      # 관성 가중치
        'c1': 2.0,     # 개인 학습률
        'c2': 2.0      # 사회 학습률
    },
    'q_learning': {
        'learning_rate': 0.1,
        'discount_factor': 0.95,
        'epsilon': 0.1,
        'episodes': 1000
    },
    'dqn': {
        'learning_rate': 0.001,
        'discount_factor': 0.95,
        'epsilon': 0.1,
        'epsilon_decay': 0.995,
        'episodes': 1000,
        'batch_size': 32,
        'memory_size': 10000
    }
}

# 비용 설정
COST_CONFIG = {
    'fuel_cost_per_km': 0.5,      # 원/km
    'drone_operation_cost': 100.0, # 원/hour
    'delivery_penalty': 50.0,      # 원 (지연 배달 페널티)
    'collision_penalty': 1000.0,   # 원 (충돌 페널티)
    'battery_replacement_cost': 500.0,  # 원 (배터리 교체 비용)
}

# 시뮬레이션 설정
SIMULATION_CONFIG = {
    'time_step': 1.0,  # seconds
    'simulation_duration': 1440.0,  # minutes (24시간)
    'visualization_interval': 60.0,  # seconds
    'save_results': True,
    'output_path': './simulation_results/'
}

# 시각화 설정
VISUALIZATION_CONFIG = {
    'figure_size': (15, 12),
    'building_alpha': 0.7,
    'drone_size': 50,
    'route_linewidth': 2,
    'depot_color': 'green',
    'restaurant_color': 'red',
    'residential_color': 'blue',
    'drone_color': 'orange'
}

# 성능 지표 설정
PERFORMANCE_METRICS = {
    'total_cost': True,
    'average_delivery_time': True,
    'drone_utilization_rate': True,
    'energy_efficiency': True,
    'customer_satisfaction': True,
    'safety_score': True
} 