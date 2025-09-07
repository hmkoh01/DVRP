# algorithm/routing/realtime_route_manager.py
# -*- coding: utf-8 -*-
import numpy as np
import copy
from datetime import datetime
from typing import List, Dict, Any
import math
import warnings
warnings.filterwarnings('ignore')


class RealtimeRouteManager:
    """
    실시간 경로 관리 클래스
    새로운 주문이 들어올 때마다 기존 경로에 효율적으로 추가
    """

    def __init__(self, depots, drones, optimization_target='cost', speed_mps=15.0, converter=None):
        self.depots = depots
        self.drones = drones
        self.optimization_target = optimization_target
        self.speed_mps = speed_mps
        self.converter = converter

        # 현재 활성 경로들 (드론별)
        self.active_routes: Dict[str, List[Dict[str, Any]]] = {}

        # 드론 상태 관리
        self.drone_states: Dict[str, Dict[str, Any]] = {}

        # 경로 업데이트 히스토리
        self.update_history: List[Dict[str, Any]] = []

        self._initialize_drones()

    def _initialize_drones(self):
        for drone in self.drones:
            drone_id = drone['id']
            self.drone_states[drone_id] = {
                'drone_id': drone_id,
                'depot_id': drone['depot_id'],
                'current_location': {
                    'longitude': drone['current_lon'],
                    'latitude': drone['current_lat'],
                    'height': drone['current_height']
                },
                'status': 'idle',  # idle, flying, pickup, delivery, returning
                'current_route': None,
                'current_route_index': 0,
                'battery': drone['battery'],
                'payload': drone['current_payload'],
                'completed_deliveries': 0,
                'total_distance': 0.0,
                'last_update_time': datetime.now()
            }
            self.active_routes[drone_id] = []

    # ---------------- 신규 요청 ----------------
    def add_new_delivery_request(self, delivery_request):
        print(f"🆕 새로운 배달 요청 추가: {delivery_request['request_id']}")
        available_drones = self._find_available_drones()
        if not available_drones:
            print("❌ 사용 가능한 드론이 없습니다.")
            return False

        insertion_costs = []
        for drone_id in available_drones:
            cost = self._calculate_insertion_cost(drone_id, delivery_request)
            insertion_costs.append((drone_id, cost))

        insertion_costs.sort(key=lambda x: x[1])
        best_drone_id, best_cost = insertion_costs[0]
        print(f"✅ 최적 드론 선택: {best_drone_id} (비용: {best_cost:.2f})")

        success = self._insert_request_to_drone_route(best_drone_id, delivery_request)
        if success:
            self.update_history.append({
                'timestamp': datetime.now(),
                'request_id': delivery_request['request_id'],
                'drone_id': best_drone_id,
                'insertion_cost': best_cost,
                'action': 'add_request'
            })
            print(f"✅ 요청 {delivery_request['request_id']}이 드론 {best_drone_id}에 성공적으로 추가됨")
        return success

    def _find_available_drones(self):
        available_drones = []
        for drone_id, st in self.drone_states.items():
            if st['battery'] > 20 and st['payload'] < 2.0:
                if st['status'] == 'idle' or self._is_route_near_completion(drone_id):
                    available_drones.append(drone_id)
        return available_drones

    def _is_route_near_completion(self, drone_id):
        if drone_id not in self.active_routes:
            return True
        route = self.active_routes[drone_id]
        if not route:
            return True
        current_index = self.drone_states[drone_id]['current_route_index']
        return current_index >= len(route) * 0.8

    def _calculate_insertion_cost(self, drone_id, delivery_request):
        if drone_id not in self.active_routes:
            return self._calculate_direct_delivery_cost(drone_id, delivery_request)
        current_route = self.active_routes[drone_id]
        if not current_route:
            return self._calculate_direct_delivery_cost(drone_id, delivery_request)

        min_cost = float('inf')
        for i in range(len(current_route) + 1):
            new_route = self._insert_request_at_position(current_route, delivery_request, i)
            total_cost = self._calculate_route_total_cost(drone_id, new_route)
            if total_cost < min_cost:
                min_cost = total_cost
        return min_cost

    # -------- 거리/비용 (고쳐진 스케일) --------
    def _calculate_distance_3d(self, p1, p2):
        """
        경위도 → 미터 근사로 3D 거리 계산
        """
        lat1 = float(p1['latitude']); lon1 = float(p1['longitude']); h1 = float(p1.get('height', 0.0))
        lat2 = float(p2['latitude']); lon2 = float(p2['longitude']); h2 = float(p2.get('height', 0.0))

        dlat = (lat2 - lat1)
        dlon = (lon2 - lon1)
        latm = math.radians((lat1 + lat2) * 0.5)

        m_per_deg_lat = 111_320.0
        m_per_deg_lon = 111_320.0 * math.cos(latm)

        dx = dlon * m_per_deg_lon
        dy = dlat * m_per_deg_lat
        dz = (h2 - h1)
        return math.sqrt(dx*dx + dy*dy + dz*dz)

    def _calculate_direct_delivery_cost(self, drone_id, delivery_request):
        st = self.drone_states[drone_id]
        depot = self._get_depot_by_id(st['depot_id'])

        d1 = self._calculate_distance_3d(depot, delivery_request['restaurant_location'])
        d2 = self._calculate_distance_3d(delivery_request['restaurant_location'], delivery_request['customer_location'])
        d3 = self._calculate_distance_3d(delivery_request['customer_location'], depot)
        total_distance = d1 + d2 + d3

        total_time = total_distance / self.speed_mps  # m / (m/s) = s
        cost = total_distance * 0.5 + (total_time / 3600.0) * 100.0
        return cost

    def _insert_request_at_position(self, current_route, delivery_request, position):
        new_route = current_route.copy()
        restaurant_point = {
            'type': 'pickup',
            'location': delivery_request['restaurant_location'],
            'order_id': delivery_request['order_id']
        }
        customer_point = {
            'type': 'delivery',
            'location': delivery_request['customer_location'],
            'order_id': delivery_request['order_id']
        }
        if position == 0:
            new_route.insert(0, customer_point)
            new_route.insert(0, restaurant_point)
        elif position == len(current_route):
            new_route.append(restaurant_point)
            new_route.append(customer_point)
        else:
            new_route.insert(position, restaurant_point)
            new_route.insert(position + 1, customer_point)
        return new_route

    def _calculate_route_total_cost(self, drone_id, route):
        if not route:
            return 0.0
        st = self.drone_states[drone_id]
        depot = self._get_depot_by_id(st['depot_id'])

        total_cost = 0.0
        cur = depot
        for pt in route:
            nxt = pt['location']
            dist = self._calculate_distance_3d(cur, nxt)
            t = dist / self.speed_mps
            cost = dist * 0.5 + (t / 3600.0) * 100.0
            total_cost += cost
            cur = nxt

        ret = self._calculate_distance_3d(cur, depot)
        ret_t = ret / self.speed_mps
        ret_cost = ret * 0.5 + (ret_t / 3600.0) * 100.0
        return total_cost + ret_cost

    def _insert_request_to_drone_route(self, drone_id, delivery_request):
        try:
            best_pos, best_cost = self._find_best_insertion_position(drone_id, delivery_request)
            if drone_id not in self.active_routes:
                self.active_routes[drone_id] = []

            restaurant_point = {
                'type': 'pickup',
                'location': delivery_request['restaurant_location'],
                'order_id': delivery_request['order_id'],
                'status': 'pending'
            }
            customer_point = {
                'type': 'delivery',
                'location': delivery_request['customer_location'],
                'order_id': delivery_request['order_id'],
                'status': 'pending'
            }

            if best_pos == 0:
                self.active_routes[drone_id].insert(0, customer_point)
                self.active_routes[drone_id].insert(0, restaurant_point)
            elif best_pos == len(self.active_routes[drone_id]):
                self.active_routes[drone_id].append(restaurant_point)
                self.active_routes[drone_id].append(customer_point)
            else:
                self.active_routes[drone_id].insert(best_pos, restaurant_point)
                self.active_routes[drone_id].insert(best_pos + 1, customer_point)

            if self.drone_states[drone_id]['status'] == 'idle':
                self.drone_states[drone_id]['status'] = 'flying'

            print(f"✅ 요청 삽입 위치: {best_pos} (비용: {best_cost:.2f})")
            return True
        except Exception as e:
            print(f"❌ 요청 삽입 실패: {e}")
            return False

    def _find_best_insertion_position(self, drone_id, delivery_request):
        if drone_id not in self.active_routes or not self.active_routes[drone_id]:
            return 0, self._calculate_direct_delivery_cost(drone_id, delivery_request)

        current_route = self.active_routes[drone_id]
        min_cost = float('inf')
        best_position = 0
        for i in range(len(current_route) + 1):
            new_route = self._insert_request_at_position(current_route, delivery_request, i)
            total_cost = self._calculate_route_total_cost(drone_id, new_route)
            if total_cost < min_cost:
                min_cost = total_cost
                best_position = i
        return best_position, min_cost

    def _get_depot_by_id(self, depot_id):
        for depot in self.depots:
            if depot['id'] == depot_id:
                return {
                    'longitude': depot['longitude'],
                    'latitude': depot['latitude'],
                    'height': depot.get('height', 50.0)
                }
        return None

    # -------- 실행 중 상태 갱신(텍스트 시뮬처럼 유지) --------
    def update_drone_positions(self, current_time):
        for drone_id in list(self.drone_states.keys()):
            drone_state = self.drone_states[drone_id]
            if drone_state['status'] == 'idle':
                continue
            self._move_drone(drone_id, current_time)
            if self._is_route_completed(drone_id):
                self._complete_drone_route(drone_id)

    def _move_drone(self, drone_id, current_time):
        drone_state = self.drone_states[drone_id]
        route = self.active_routes.get(drone_id, [])
        if not route:
            return
        idx = drone_state['current_route_index']
        if idx >= len(route):
            return

        next_point = route[idx]
        cur = drone_state['current_location']
        distance = self._calculate_distance_3d(cur, next_point['location'])

        move_time = distance / self.speed_mps
        if move_time <= 60.0:
            drone_state['current_location'] = copy.deepcopy(next_point['location'])
            drone_state['current_route_index'] += 1
            drone_state['total_distance'] += distance
            battery_consumption = distance * 0.0001
            drone_state['battery'] -= battery_consumption
            if next_point['type'] == 'pickup':
                drone_state['status'] = 'pickup'
            elif next_point['type'] == 'delivery':
                drone_state['status'] = 'delivery'
                drone_state['completed_deliveries'] += 1
        else:
            max_distance_per_min = self.speed_mps * 60.0
            if distance > max_distance_per_min:
                lon1, lat1, h1 = cur['longitude'], cur['latitude'], cur.get('height', 0.0)
                lon2, lat2, h2 = next_point['location']['longitude'], next_point['location']['latitude'], next_point['location'].get('height', 0.0)

                dlat = (lat2 - lat1)
                dlon = (lon2 - lon1)
                latm = math.radians((lat1 + lat2) * 0.5)
                m_per_deg_lat = 111_320.0
                m_per_deg_lon = 111_320.0 * math.cos(latm)

                dx = dlon * m_per_deg_lon
                dy = dlat * m_per_deg_lat
                dz = (h2 - h1)

                norm = math.sqrt(dx*dx + dy*dy + dz*dz)
                if norm > 0:
                    ratio = max_distance_per_min / distance
                    # 경위도 좌표에서 비율만큼 선형 보간
                    new_lon = lon1 + (lon2 - lon1) * ratio
                    new_lat = lat1 + (lat2 - lat1) * ratio
                    new_h   = h1   + (h2   - h1)   * ratio

                    drone_state['current_location'] = {
                        'longitude': new_lon,
                        'latitude': new_lat,
                        'height': new_h
                    }
                    battery_consumption = max_distance_per_min * 0.0001
                    drone_state['battery'] -= battery_consumption
                    drone_state['battery'] = max(0.0, drone_state['battery'])

    def _is_route_completed(self, drone_id):
        drone_state = self.drone_states[drone_id]
        route = self.active_routes.get(drone_id, [])
        if not route:
            return False
        return drone_state['current_route_index'] >= len(route)

    def _complete_drone_route(self, drone_id):
        drone_state = self.drone_states[drone_id]
        depot = self._get_depot_by_id(drone_state['depot_id'])
        return_distance = self._calculate_distance_3d(drone_state['current_location'], depot)
        drone_state['current_location'] = copy.deepcopy(depot)
        drone_state['status'] = 'idle'
        drone_state['total_distance'] += return_distance
        drone_state['current_route_index'] = 0
        if drone_id in self.active_routes:
            del self.active_routes[drone_id]
        self.update_history.append({
            'timestamp': datetime.now(),
            'action': 'route_completed',
            'drone_id': drone_id,
            'details': f'경로 완료, 총 거리: {drone_state["total_distance"]:.1f}m'
        })

    # -------- 조회 유틸 --------
    def get_drone_status(self, drone_id):
        if drone_id not in self.drone_states:
            return None
        return self.drone_states[drone_id].copy()

    def get_all_drone_statuses(self):
        return {drone_id: status.copy() for drone_id, status in self.drone_states.items()}

    def get_active_routes(self):
        return self.active_routes.copy()

    def get_route_statistics(self):
        stats = {
            'total_drones': len(self.drone_states),
            'active_drones': len([d for d in self.drone_states.values() if d['status'] != 'idle']),
            'total_requests': sum(len(route) // 2 for route in self.active_routes.values()),
            'completed_deliveries': sum(d['completed_deliveries'] for d in self.drone_states.values()),
            'total_distance': sum(d['total_distance'] for d in self.drone_states.values()),
            'update_history_count': len(self.update_history)
        }
        return stats

    def print_current_status(self):
        print("\n=== 실시간 경로 관리자 상태 ===")
        stats = self.get_route_statistics()
        print(f"총 드론 수: {stats['total_drones']}")
        print(f"활성 드론 수: {stats['active_drones']}")
        print(f"총 요청 수: {stats['total_requests']}")
        print(f"완료된 배달: {stats['completed_deliveries']}")
        print(f"총 이동 거리: {stats['total_distance']:.1f}m")
        print(f"\n드론별 상태:")
        for drone_id, status in self.drone_states.items():
            route_length = len(self.active_routes.get(drone_id, []))
            print(f"  {drone_id}: {status['status']} (경로: {route_length}개 포인트, 배터리: {status['battery']:.1f}%)")
        print(f"\n최근 업데이트:")
        for update in self.update_history[-5:]:
            print(f"  {update['timestamp'].strftime('%H:%M:%S')}: {update['action']} - 드론 {update['drone_id']}")
