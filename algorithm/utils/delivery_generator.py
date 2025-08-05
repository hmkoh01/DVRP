"""
배달 요청 생성기
동적으로 음식 배달 요청을 생성하는 클래스
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class DeliveryRequestGenerator:
    """
    배달 요청 생성 클래스
    """
    
    def __init__(self, building_data_loader):
        """
        초기화
        """
        self.data_loader = building_data_loader
        self.requests = []
        self.request_id_counter = 0
        
    def generate_requests(self, num_requests=50, time_window_hours=24, 
                         generation_rate=None, random_seed=42):
        """
        배달 요청 생성
        """
        if self.data_loader.restaurants is None or self.data_loader.residential_buildings is None:
            print("식당 또는 주거용 건물 데이터가 없습니다.")
            return []
        
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        print(f"배달 요청 생성 시작: {num_requests}개 요청")
        
        # 시간 윈도우 설정
        start_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        end_time = start_time + timedelta(hours=time_window_hours)
        
        # 요청 생성
        for i in range(num_requests):
            request = self._generate_single_request(start_time, end_time, i)
            if request:
                self.requests.append(request)
                self.request_id_counter += 1
        
        print(f"배달 요청 생성 완료: {len(self.requests)}개 요청")
        
        # 요청 통계 출력
        self._print_request_statistics()
        
        return self.requests
    
    def _generate_single_request(self, start_time, end_time, request_index):
        """
        단일 배달 요청 생성
        """
        # 식당과 주거용 건물 무작위 선택
        restaurant = self.data_loader.get_random_restaurant()
        residential = self.data_loader.get_random_residential()
        
        if restaurant is None or residential is None:
            return None
        
        # 요청 시간 생성 (시간대별 가중치 적용)
        request_time = self._generate_request_time(start_time, end_time, request_index)
        
        # 배달 시간 윈도우 설정
        delivery_deadline = request_time + timedelta(minutes=30)  # 30분 내 배달
        
        # 음식 종류 및 가격 설정
        food_type, food_price = self._generate_food_info()
        
        # 우선순위 설정 (시간, 거리, 가격 등 고려)
        priority = self._calculate_priority(request_time, restaurant, residential, food_price)
        
        request = {
            'request_id': self.request_id_counter,
            'restaurant_id': restaurant['building_id'],
            'restaurant_name': f"Restaurant_{restaurant['building_id']}",
            'restaurant_location': {
                'longitude': restaurant['longitude'],
                'latitude': restaurant['latitude'],
                'height': restaurant['height_m']
            },
            'customer_id': f"Customer_{residential['building_id']}",
            'customer_location': {
                'longitude': residential['longitude'],
                'latitude': residential['latitude'],
                'height': residential['height_m']
            },
            'food_type': food_type,
            'food_price': food_price,
            'request_time': request_time,
            'delivery_deadline': delivery_deadline,
            'priority': priority,
            'status': 'pending',  # pending, assigned, in_progress, completed, failed
            'assigned_drone': None,
            'actual_delivery_time': None,
            'distance_3d': self.data_loader.calculate_3d_distance(restaurant, residential)
        }
        
        return request
    
    def _generate_request_time(self, start_time, end_time, request_index):
        """
        요청 시간 생성 (시간대별 가중치 적용)
        """
        # 시간대별 가중치 (점심, 저녁 시간대에 더 많은 주문)
        time_weights = {
            6: 0.1,   # 아침
            7: 0.2,
            8: 0.3,
            9: 0.4,
            10: 0.5,
            11: 0.8,
            12: 1.0,  # 점심 피크
            13: 1.0,
            14: 0.7,
            15: 0.5,
            16: 0.4,
            17: 0.6,
            18: 0.9,  # 저녁 피크
            19: 1.0,
            20: 0.8,
            21: 0.6,
            22: 0.4,
            23: 0.2,
            0: 0.1,   # 새벽
            1: 0.05,
            2: 0.05,
            3: 0.05,
            4: 0.05,
            5: 0.05
        }
        
        # 가중치 기반 시간 선택
        hours = list(time_weights.keys())
        weights = list(time_weights.values())
        
        # 요청 인덱스에 따라 시간 분산
        hour = random.choices(hours, weights=weights)[0]
        
        # 분과 초는 무작위
        minute = random.randint(0, 59)
        second = random.randint(0, 59)
        
        # 요청 인덱스를 고려한 시간 분산
        time_offset = timedelta(
            hours=hour,
            minutes=minute,
            seconds=second,
            microseconds=random.randint(0, 999999)
        )
        
        # 전체 시간 윈도우 내에서 분산
        total_seconds = (end_time - start_time).total_seconds()
        request_seconds = (request_index / 100) * total_seconds  # 요청 인덱스 기반 분산
        
        request_time = start_time + timedelta(seconds=request_seconds) + time_offset
        
        return request_time
    
    def _generate_food_info(self):
        """
        음식 종류 및 가격 생성
        """
        food_types = [
            ('한식', 8000, 15000),
            ('중식', 6000, 12000),
            ('일식', 10000, 20000),
            ('양식', 12000, 25000),
            ('분식', 3000, 8000),
            ('치킨', 15000, 25000),
            ('피자', 18000, 30000),
            ('햄버거', 5000, 12000),
            ('커피/음료', 2000, 8000),
            ('디저트', 3000, 10000)
        ]
        
        food_type, min_price, max_price = random.choice(food_types)
        food_price = random.randint(min_price, max_price)
        
        return food_type, food_price
    
    def _calculate_priority(self, request_time, restaurant, residential, food_price):
        """
        배달 우선순위 계산
        """
        priority = 0
        
        # 1. 시간 기반 우선순위 (마감 시간이 가까울수록 높음)
        time_factor = 1.0
        if request_time.hour in [12, 13, 18, 19]:  # 피크 시간대
            time_factor = 1.5
        
        # 2. 거리 기반 우선순위 (가까울수록 높음)
        distance = self.data_loader.calculate_3d_distance(restaurant, residential)
        distance_factor = max(0.5, 1.0 - (distance / 1000))  # 1km 기준
        
        # 3. 가격 기반 우선순위 (비쌀수록 높음)
        price_factor = min(2.0, food_price / 10000)  # 1만원 기준
        
        # 4. 건물 높이 기반 우선순위 (높은 건물일수록 낮음 - 배달 어려움)
        height_factor = max(0.5, 1.0 - (residential['height_m'] / 100))  # 100m 기준
        
        priority = (time_factor + distance_factor + price_factor + height_factor) / 4
        
        return round(priority, 3)
    
    def generate_dynamic_requests(self, current_time, time_window_minutes=60, 
                                 generation_rate=0.1, max_requests=10):
        """
        동적 배달 요청 생성 (실시간)
        """
        requests = []
        
        # 시간 윈도우 내에서 요청 생성
        end_time = current_time + timedelta(minutes=time_window_minutes)
        
        # 포아송 분포를 사용한 요청 수 결정
        expected_requests = generation_rate * time_window_minutes
        num_requests = min(np.random.poisson(expected_requests), max_requests)
        
        for _ in range(num_requests):
            request = self._generate_single_request(current_time, end_time, self.request_id_counter)
            if request:
                requests.append(request)
                self.requests.append(request)
                self.request_id_counter += 1
        
        return requests
    
    def get_pending_requests(self):
        """
        대기 중인 요청 조회
        """
        return [req for req in self.requests if req['status'] == 'pending']
    
    def get_requests_by_time_range(self, start_time, end_time):
        """
        시간 범위별 요청 조회
        """
        return [req for req in self.requests 
                if start_time <= req['request_time'] <= end_time]
    
    def get_requests_by_restaurant(self, restaurant_id):
        """
        특정 식당의 요청 조회
        """
        return [req for req in self.requests if req['restaurant_id'] == restaurant_id]
    
    def get_requests_by_customer(self, customer_id):
        """
        특정 고객의 요청 조회
        """
        return [req for req in self.requests if req['customer_id'] == customer_id]
    
    def update_request_status(self, request_id, status, drone_id=None, delivery_time=None):
        """
        요청 상태 업데이트
        """
        for request in self.requests:
            if request['request_id'] == request_id:
                request['status'] = status
                if drone_id:
                    request['assigned_drone'] = drone_id
                if delivery_time:
                    request['actual_delivery_time'] = delivery_time
                break
    
    def _print_request_statistics(self):
        """
        요청 통계 출력
        """
        if not self.requests:
            return
        
        # 시간대별 분포
        hour_distribution = {}
        for request in self.requests:
            hour = request['request_time'].hour
            hour_distribution[hour] = hour_distribution.get(hour, 0) + 1
        
        # 음식 종류별 분포
        food_distribution = {}
        for request in self.requests:
            food_type = request['food_type']
            food_distribution[food_type] = food_distribution.get(food_type, 0) + 1
        
        # 거리 통계
        distances = [req['distance_3d'] for req in self.requests]
        
        print(f"\n=== 배달 요청 통계 ===")
        print(f"총 요청 수: {len(self.requests)}")
        print(f"평균 거리: {np.mean(distances):.1f}m")
        print(f"최대 거리: {np.max(distances):.1f}m")
        print(f"최소 거리: {np.min(distances):.1f}m")
        
        print(f"\n시간대별 분포:")
        for hour in sorted(hour_distribution.keys()):
            print(f"  {hour:02d}시: {hour_distribution[hour]}개")
        
        print(f"\n음식 종류별 분포:")
        for food_type, count in sorted(food_distribution.items(), key=lambda x: x[1], reverse=True):
            print(f"  {food_type}: {count}개")
    
    def get_request_statistics(self):
        """
        요청 통계 정보 반환
        """
        if not self.requests:
            return {}
        
        # 시간대별 분포
        hour_distribution = {}
        for request in self.requests:
            hour = request['request_time'].hour
            hour_distribution[hour] = hour_distribution.get(hour, 0) + 1
        
        # 음식 종류별 분포
        food_distribution = {}
        for request in self.requests:
            food_type = request['food_type']
            food_distribution[food_type] = food_distribution.get(food_type, 0) + 1
        
        # 거리 통계
        distances = [req['distance_3d'] for req in self.requests]
        priorities = [req['priority'] for req in self.requests]
        prices = [req['food_price'] for req in self.requests]
        
        stats = {
            'total_requests': len(self.requests),
            'hour_distribution': hour_distribution,
            'food_distribution': food_distribution,
            'distance_stats': {
                'mean': np.mean(distances),
                'max': np.max(distances),
                'min': np.min(distances),
                'std': np.std(distances)
            },
            'priority_stats': {
                'mean': np.mean(priorities),
                'max': np.max(priorities),
                'min': np.min(priorities)
            },
            'price_stats': {
                'mean': np.mean(prices),
                'max': np.max(prices),
                'min': np.min(prices),
                'total': np.sum(prices)
            }
        }
        
        return stats
    
    def save_requests(self, output_path):
        """
        요청 데이터 저장
        """
        if not self.requests:
            print("저장할 요청이 없습니다.")
            return
        
        # DataFrame으로 변환
        df_data = []
        for request in self.requests:
            row = {
                'request_id': request['request_id'],
                'restaurant_id': request['restaurant_id'],
                'restaurant_lon': request['restaurant_location']['longitude'],
                'restaurant_lat': request['restaurant_location']['latitude'],
                'restaurant_height': request['restaurant_location']['height'],
                'customer_id': request['customer_id'],
                'customer_lon': request['customer_location']['longitude'],
                'customer_lat': request['customer_location']['latitude'],
                'customer_height': request['customer_location']['height'],
                'food_type': request['food_type'],
                'food_price': request['food_price'],
                'request_time': request['request_time'],
                'delivery_deadline': request['delivery_deadline'],
                'priority': request['priority'],
                'status': request['status'],
                'distance_3d': request['distance_3d']
            }
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        df.to_csv(output_path, index=False, encoding='utf-8')
        
        print(f"요청 데이터 저장: {output_path}")
    
    def load_requests(self, file_path):
        """
        저장된 요청 데이터 로드
        """
        try:
            df = pd.read_csv(file_path)
            
            self.requests = []
            for _, row in df.iterrows():
                request = {
                    'request_id': row['request_id'],
                    'restaurant_id': row['restaurant_id'],
                    'restaurant_name': f"Restaurant_{row['restaurant_id']}",
                    'restaurant_location': {
                        'longitude': row['restaurant_lon'],
                        'latitude': row['restaurant_lat'],
                        'height': row['restaurant_height']
                    },
                    'customer_id': row['customer_id'],
                    'customer_location': {
                        'longitude': row['customer_lon'],
                        'latitude': row['customer_lat'],
                        'height': row['customer_height']
                    },
                    'food_type': row['food_type'],
                    'food_price': row['food_price'],
                    'request_time': pd.to_datetime(row['request_time']),
                    'delivery_deadline': pd.to_datetime(row['delivery_deadline']),
                    'priority': row['priority'],
                    'status': row['status'],
                    'assigned_drone': None,
                    'actual_delivery_time': None,
                    'distance_3d': row['distance_3d']
                }
                self.requests.append(request)
            
            self.request_id_counter = max([req['request_id'] for req in self.requests]) + 1
            
            print(f"요청 데이터 로드 완료: {len(self.requests)}개 요청")
            return self.requests
            
        except Exception as e:
            print(f"요청 데이터 로드 실패: {e}")
            return [] 