"""
라우팅 알고리즘 설정 파일
"""

# 알고리즘별 설정
ALGORITHM_CONFIG = {
    # Metaheuristic 알고리즘 설정
    'genetic': {
        'population_size': 50,
        'generations': 100,
        'mutation_rate': 0.1,
        'crossover_rate': 0.8,
        'elite_size': 5
    },
    
    'ant_colony': {
        'num_ants': 30,
        'iterations': 100,
        'evaporation_rate': 0.1,
        'alpha': 1.0,  # 페로몬 중요도
        'beta': 2.0    # 거리 중요도
    },
    
    'particle_swarm': {
        'num_particles': 30,
        'iterations': 100,
        'cognitive_weight': 2.0,
        'social_weight': 2.0,
        'inertia_weight': 0.7
    },
    
    'simulated_annealing': {
        'initial_temperature': 1000,
        'cooling_rate': 0.95,
        'iterations_per_temp': 10,
        'min_temperature': 1
    },
    
    # Reinforcement Learning 알고리즘 설정
    'q_learning': {
        'learning_rate': 0.1,
        'discount_factor': 0.9,
        'epsilon': 0.1,
        'episodes': 1000
    },
    
    'dqn': {
        'learning_rate': 0.001,
        'discount_factor': 0.9,
        'epsilon': 0.1,
        'epsilon_decay': 0.995,
        'episodes': 1000,
        'batch_size': 32
    },
    
    'actor_critic': {
        'actor_learning_rate': 0.001,
        'critic_learning_rate': 0.001,
        'discount_factor': 0.9,
        'episodes': 1000
    },
    
    # Matheuristic 알고리즘 설정
    'clarke_wright': {
        'max_route_length': 100,
        'max_route_time': 3600
    },
    
    'savings': {
        'max_route_length': 100,
        'max_route_time': 3600
    },
    
    'sweep': {
        'max_route_length': 100,
        'max_route_time': 3600
    }
}

# 드론 설정
DRONE_CONFIG = {
    'max_speed': 15.0,  # m/s
    'max_payload': 5.0,  # kg
    'battery_capacity': 3600,  # seconds
    'takeoff_time': 30,  # seconds
    'landing_time': 30,  # seconds
    'service_time': 120  # seconds
}

# 시스템 설정
SYSTEM_CONFIG = {
    'time_window': 3600,  # seconds
    'max_wait_time': 1800,  # seconds
    'penalty_factor': 10.0
} 