# -*- coding: utf-8 -*-
import airsim
import numpy as np
import time
import threading
import queue
from typing import Dict, Any, Optional, Tuple, List, Union
import math
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

try:
    import pandas as pd
except Exception:
    pd = None

from algorithm.utils.coordinate_converter import CoordinateConverter
from algorithm.simulation.route_executor import RouteExecutor

QueueItem = Union[airsim.Vector3r, Tuple[str, List[airsim.Vector3r]]]


class AirSimVisualizer:
    """
    다중 드론 동시 제어 + 경로 최적화 연동 + 건물/Depot 오버레이
    (방법 1) StaticMesh 스폰 시 기본 제공 큐브(/Engine/BasicShapes/Cube.Cube)만 사용
    """

    def __init__(self, depots, drones, building_data=None, **kwargs):
        self.depots = depots
        self.drones_meta = drones
        self.building_data = building_data

        # 드론별 자원
        self.clients: Dict[str, airsim.MultirotorClient] = {}
        self.queues: Dict[str, "queue.Queue[Optional[QueueItem]]"] = {}
        self.workers: Dict[str, threading.Thread] = {}
        self.drone_states: Dict[str, Dict[str, Any]] = {}

        # 시뮬레이션 제어
        self.connected = False
        self.simulation_running = False
        self.simulation_thread = None

        # 좌표 기준 (POSTECH 근방)
        self.base_lat = 36.0139
        self.base_lon = 129.3261

        # 비행 파라미터
        self.order_prob = 0.08
        self.cruise_speed = 12.0
        self.confirm_dist = 1.0
        self.poll_interval = 0.2
        self.move_timeout = 120.0
        self.cruise_alt_m = 35.0

        # RouteManager 연동
        self.use_route_manager = False
        self.converter: Optional[CoordinateConverter] = None
        self.order_generator = None
        self.route_executor: Optional[RouteExecutor] = None
        self.logical_to_vehicle: Dict[str, str] = {}

        # 오버레이 옵션
        self.plot_buildings = True
        self.plot_buildings_as_cubes = True   # 기본 큐브 StaticMesh로 시도
        self.building_marker_color = [0.2, 0.6, 1.0, 1.0]
        self.building_marker_size = 4
        self.building_cube_size = 8.0
        self.line_thickness = 2.0

        self.plot_depots = True
        self.depot_marker_color = [0.0, 1.0, 0.2, 1.0]
        self.depot_marker_size = 20
        self.depot_as_cube = True
        self.depot_cube_size = 12.0

        self.max_buildings_to_plot: Optional[int] = None
        self._max_points_per_call = 5000
        self._max_line_points_per_call = 20000
        self._max_draw_radius_m: Optional[float] = None

        # 스폰 추적 (정리용)
        self._spawned_objects = set()

        # (방법 1) 기본 큐브 에셋만 사용
        self._asset_cube = "/Engine/BasicShapes/Cube"

        print("🚁 AirSim 3D 시각화 시스템 초기화 완료")

    # -------- RouteManager 연동 --------
    def attach_route_stack(self, route_manager, converter: CoordinateConverter, order_generator=None):
        self.use_route_manager = True
        self.converter = converter
        self.order_generator = order_generator
        self.logical_to_vehicle = {}
        for idx, d in enumerate(self.drones_meta, start=1):
            self.logical_to_vehicle[d["id"]] = f"Drone_{idx}"
        self.route_executor = RouteExecutor(
            route_manager=route_manager,
            converter=converter,
            enqueue_fn=self.enqueue_waypoints,
            is_vehicle_idle_fn=self.is_vehicle_idle,
            logical_to_vehicle=self.logical_to_vehicle
        )
        try:
            self._draw_world_overlays()
        except Exception:
            pass

    # ---------- 연결 ----------
    def connect_to_airsim(self):
        try:
            print("🔌 AirSim 서버에 연결 중...")
            total = len(self.depots) * 3
            drone_names = [f"Drone_{i+1}" for i in range(total)]
            for name in drone_names:
                c = airsim.MultirotorClient()
                c.confirmConnection()
                self.clients[name] = c
                self.queues[name] = queue.Queue()
                self.drone_states[name] = {
                    'status': 'idle',
                    'position': None,
                    'depot_id': None,
                    'updated': datetime.now()
                }
            self.connected = True
            print("✅ 드론별 클라이언트 생성·연결 완료")
            return True
        except Exception as e:
            print(f"❌ AirSim 서버 연결 실패: {e}")
            self.connected = False
            return False

    def setup_simulation_environment(self):
        try:
            print("🌍 AirSim 시뮬레이션 환경 설정 중...")
            self._draw_world_overlays()
            self._setup_drone_initial_positions()
            self._start_vehicle_workers()
            print("✅ AirSim 시뮬레이션 환경 설정 완료")
            return True
        except Exception as e:
            print(f"❌ 환경 설정 중 오류 발생: {e}")
            return False

    # ---------- 좌표 유틸 ----------
    def _gps_to_ned(self, lat, lon, height):
        lat_diff = lat - self.base_lat
        lon_diff = lon - self.base_lon
        ned_x = lat_diff * 111_320.0
        ned_y = lon_diff * 111_320.0 * math.cos(math.radians(self.base_lat))
        ned_z = -height
        return airsim.Vector3r(ned_x, ned_y, ned_z)

    # ---------- 초기 포즈 ----------
    def _setup_drone_initial_positions(self):
        print("🚁 드론 초기 위치 설정 중...")
        idx = 1
        for depot in self.depots:
            base = self._gps_to_ned(depot['latitude'], depot['longitude'], depot.get('height', 10.0))
            for i in range(3):
                name = f"Drone_{idx}"
                offset = airsim.Vector3r(base.x_val, base.y_val + (i-1)*2.0, base.z_val - 10.0)
                pose = airsim.Pose(offset, airsim.to_quaternion(0, 0, 0))
                c = self.clients[name]
                try:
                    c.simPause(False)
                except:
                    pass
                c.simSetVehiclePose(pose, True, vehicle_name=name)
                c.enableApiControl(True, vehicle_name=name)
                c.armDisarm(False, vehicle_name=name)
                self.drone_states[name].update({
                    'position': offset, 'status': 'idle', 'depot_id': depot['id'], 'updated': datetime.now()
                })
                idx += 1
                time.sleep(0.03)
        print(f"✅ 총 {len(self.clients)}개 드론 초기 위치 설정 완료")

    def _start_vehicle_workers(self):
        for name in self.clients.keys():
            t = threading.Thread(target=self._vehicle_worker, args=(name,), daemon=True)
            t.start()
            self.workers[name] = t
        print("✅ 드론별 워커 스레드 시작")

    # ---------- 좌표/키 유틸 ----------
    def _normalize_key(self, key):
        if not key:
            return ""
        return str(key).lower().strip().replace(" ", "_").replace("-", "_")

    def _to_float(self, value):
        if value is None:
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None

    def _sanitize_llh(self, lat: float, lon: float, h: float) -> Optional[Tuple[float, float, float]]:
        if -90 <= lat <= 90 and -180 <= lon <= 180:
            return (lat, lon, h)
        if 120 <= lat <= 140 and 20 <= lon <= 55:
            lat, lon = lon, lat
            return (lat, lon, h)
        return None

    def _heuristic_llh_pick(self, rec):
        for key, value in rec.items():
            if isinstance(value, str) and ("POINT" in value.upper() or "LINESTRING" in value.upper()):
                import re
                coords = re.findall(r'[-+]?\d*\.?\d+', value)
                if len(coords) >= 2:
                    try:
                        lon, lat = float(coords[0]), float(coords[1])
                        h = float(coords[2]) if len(coords) > 2 else 0.0
                        if -90 <= lat <= 90 and -180 <= lon <= 180:
                            return (lat, lon, h)
                    except ValueError:
                        continue
        return None

    # ---------- 레코드 → NED ----------
    def _rec_to_ned(self, rec) -> Optional[airsim.Vector3r]:
        lat_keys = {"latitude", "lat", "lat_deg", "위도"}
        lon_keys = {"longitude", "lon", "lng", "long", "lon_deg", "경도"}
        h_keys = {"height", "alt", "altitude", "h", "z", "높이", "층수"}

        norm = {self._normalize_key(k): v for k, v in rec.items()}

        def pick(keys, default=None):
            for k in list(keys):
                kk = self._normalize_key(k)
                if kk in norm and norm[kk] is not None:
                    return norm[kk]
            return default

        lat = self._to_float(pick(lat_keys))
        lon = self._to_float(pick(lon_keys))
        h = self._to_float(pick(h_keys, 0.0)) or 0.0

        if lat is not None and lon is not None:
            clean = self._sanitize_llh(lat, lon, h)
            if clean:
                lat, lon, h = clean
                v = self.converter.llh_to_ned(lat, lon, h) if self.converter else self._gps_to_ned(lat, lon, h)
                return airsim.Vector3r(v.x_val, v.y_val, v.z_val)

        if lat is not None and lon is not None and (abs(lat) > 1000 and abs(lon) > 1000):
            converted = self._convert_utm_to_llh(lon, lat, h)
            if converted:
                lat2, lon2, h2 = converted
                v = self.converter.llh_to_ned(lat2, lon2, h2) if self.converter else self._gps_to_ned(lat2, lon2, h2)
                return airsim.Vector3r(v.x_val, v.y_val, v.z_val)

        guessed = self._heuristic_llh_pick(rec)
        if guessed:
            lat2, lon2, h2 = guessed
            v = self.converter.llh_to_ned(lat2, lon2, h2) if self.converter else self._gps_to_ned(lat2, lon2, h2)
            return airsim.Vector3r(v.x_val, v.y_val, v.z_val)

        x_keys = {"x", "east", "e", "동"}
        y_keys = {"y", "north", "n", "북"}
        z_keys = {"z", "height", "alt", "h", "높이"}
        x = self._to_float(pick(x_keys))
        y = self._to_float(pick(y_keys))
        z = self._to_float(pick(z_keys, 0.0)) or 0.0
        if x is not None and y is not None:
            return airsim.Vector3r(x, y, -z)

        return None

    def _convert_utm_to_llh(self, utm_x, utm_y, height):
        """UTM(Zone 52N 가정) → 위경도. pyproj 있으면 정확 변환, 없으면 근사."""
        try:
            try:
                from pyproj import Transformer
                transformer = Transformer.from_crs("EPSG:32652", "EPSG:4326", always_xy=True)
                lon, lat = transformer.transform(utm_x, utm_y)
                return (lat, lon, height)
            except Exception:
                base_lat = 36.0
                base_lon = 129.0
                lat_offset = utm_y / 111320.0
                lon_offset = utm_x / (111320.0 * math.cos(math.radians(base_lat)))
                lat2 = base_lat + lat_offset
                lon2 = base_lon + lon_offset
                if -90 <= lat2 <= 90 and -180 <= lon2 <= 180:
                    return (lat2, lon2, height)
        except Exception as e:
            print(f"⚠️ UTM 변환 실패: {e}")
        return None

    # ---------- 오버레이 ----------
    def _get_plot_client(self):
        return next(iter(self.clients.values()), None)

    def _records_from_buildings(self):
        if pd is not None and isinstance(self.building_data, pd.DataFrame):
            return self.building_data.to_dict("records")
        if isinstance(self.building_data, list):
            return self.building_data
        return []

    def _can_spawn_objects(self) -> bool:
        c = self._get_plot_client()
        return bool(c and hasattr(c, "simSpawnObject") and hasattr(c, "simDestroyObject"))

    def _spawn_static_cube(self, name: str, pos: airsim.Vector3r, scale: airsim.Vector3r) -> bool:
        """
        기본 제공 큐브(/Engine/BasicShapes/Cube.Cube)만 사용하여 스폰.
        실패 시 False 반환 (상위에서 마커 fallback)
        """
        c = self._get_plot_client()
        if not c:
            return False

        try:
            c.simSpawnObject(
                name,
                self._asset_cube,  # ✅ 기본 큐브 에셋
                airsim.Pose(pos, airsim.to_quaternion(0, 0, 0)),
                scale,
                False
            )
            self._spawned_objects.add(name)
            return True
        except Exception as e:
            print(f"⚠️ {name} spawn 실패: {e}")
            return False

    def _draw_world_overlays(self):
        c = self._get_plot_client()
        if not c:
            return
        try:
            c.simFlushPersistentMarkers()
        except Exception:
            pass

        if self.plot_buildings:
            self._create_building_static_meshes()

        if self.plot_depots and self.depots:
            self._create_depot_static_meshes()

    def _create_building_static_meshes(self):
        """건물을 StaticMeshActor로 생성. 실패 시 simPlotPoints로 대체."""
        print("🏗️ 건물 StaticMeshActor 생성 중...")
        c = self._get_plot_client()
        if not c:
            return

        recs = self._records_from_buildings()
        created_buildings = 0
        points_fallback: List[airsim.Vector3r] = []

        limit = min(len(recs), 100 if self.max_buildings_to_plot is None else self.max_buildings_to_plot)
        can_spawn = self._can_spawn_objects() and self.plot_buildings_as_cubes

        for i in range(limit):
            r = recs[i]
            v = self._rec_to_ned(r)
            if v is None:
                continue

            width = float(r.get('width', 5.0))
            depth = float(r.get('depth', 5.0))
            height = float(r.get('height', 3.0))

            if can_spawn:
                building_name = f"Building_{r.get('id', i)}"
                blocks_per_side = max(1, min(3, int(width / 2.0)))
                block_size = 2.0

                spawned_any = False
                for bx in range(blocks_per_side):
                    for bz in range(blocks_per_side):
                        block_name = f"{building_name}_B{bx}_{bz}"
                        offset_x = (bx - blocks_per_side / 2 + 0.5) * block_size
                        offset_y = (bz - blocks_per_side / 2 + 0.5) * block_size

                        block_pos = airsim.Vector3r(
                            v.x_val + offset_x,
                            v.y_val + offset_y,
                            v.z_val - height / 2.0
                        )
                        block_scale = airsim.Vector3r(block_size, block_size, height)

                        if self._spawn_static_cube(block_name, block_pos, block_scale):
                            spawned_any = True

                if spawned_any:
                    created_buildings += 1
                else:
                    points_fallback.append(v)
            else:
                points_fallback.append(v)

        if points_fallback:
            try:
                c.simPlotPoints(points_fallback, self.building_marker_color, self.building_marker_size,
                                duration=0.0, is_persistent=True)
            except Exception as e:
                print(f"⚠️ 건물 포인트 플롯 실패: {e}")

        print(f"🏢 건물 StaticMeshActor 생성 완료: {created_buildings}개 (fallback 포인트: {len(points_fallback)}개)")

    def _create_depot_static_meshes(self):
        """Depot들을 StaticMeshActor로 생성. 실패 시 포인트 마커로 대체."""
        print("🏭 Depot StaticMeshActor 생성 중...")
        c = self._get_plot_client()
        if not c:
            return

        can_spawn = self._can_spawn_objects() and self.depot_as_cube
        points_fallback: List[airsim.Vector3r] = []

        for d in self.depots:
            name = f"Depot_{d['id']}"
            v = self._gps_to_ned(d["latitude"], d["longitude"], d.get("height", 10.0))
            depot_pos = airsim.Vector3r(v.x_val, v.y_val, v.z_val - 5.0)

            if can_spawn:
                ok = self._spawn_static_cube(
                    name,
                    depot_pos,
                    airsim.Vector3r(self.depot_cube_size, self.depot_cube_size, 10.0)
                )
                if not ok:
                    points_fallback.append(depot_pos)
            else:
                points_fallback.append(depot_pos)

        if points_fallback:
            try:
                c.simPlotPoints(points_fallback, self.depot_marker_color, self.depot_marker_size,
                                duration=0.0, is_persistent=True)
            except Exception as e:
                print(f"⚠️ Depot 포인트 플롯 실패: {e}")

        print(f"🏭 Depot StaticMeshActor 생성 완료: {len(self.depots) - len(points_fallback)}개")

    # ---------- 시뮬레이션 루프 ----------
    def start_simulation(self, duration_hours=1):
        if not self.connected:
            print("❌ AirSim에 연결되지 않았습니다.")
            return False
        print("🎬 AirSim 3D 시뮬레이션 시작")
        self.simulation_running = True
        self.simulation_thread = threading.Thread(target=self._simulation_loop, daemon=True)
        self.simulation_thread.start()
        print("✅ 시뮬레이션 스레드 시작됨")
        return True

    def stop_simulation(self):
        print("⏹️ 시뮬레이션 중지 중...")
        self.simulation_running = False
        for q in self.queues.values():
            q.put(None)
        if self.simulation_thread and self.simulation_thread.is_alive():
            self.simulation_thread.join(timeout=2.0)
        for t in self.workers.values():
            t.join(timeout=2.0)
        print("✅ 시뮬레이션 중지 완료")

    def _simulation_loop(self):
        print("🔄 시뮬레이션 루프 시작")
        while self.simulation_running:
            try:
                if self.use_route_manager and self.route_executor:
                    if self.order_generator:
                        new_reqs = self.order_generator.maybe_generate()
                        for r in new_reqs:
                            try:
                                self.route_executor.rm.add_new_delivery_request(r)
                            except Exception as e:
                                print(f"⚠️ 주문 추가 실패: {e}")
                    self.route_executor.sync_active_routes()
                else:
                    self._process_orders_random()
                time.sleep(1.0)
            except Exception as e:
                print(f"⚠️ 시뮬레이션 루프 중 오류: {e}")
        print("✅ 시뮬레이션 루프 종료")

    def _process_orders_random(self):
        for name, state in list(self.drone_states.items()):
            if state['status'] == 'idle' and np.random.random() < self.order_prob:
                cur = state['position']
                if cur is None:
                    continue
                dest = airsim.Vector3r(cur.x_val + np.random.uniform(-500, 500),
                                       cur.y_val + np.random.uniform(-500, 500),
                                       -np.random.uniform(30, 80))
                print(f"📦 {name}: 새 배달 주문 생성 → ({dest.x_val:.1f},{dest.y_val:.1f},{dest.z_val:.1f})")
                self.drone_states[name]['status'] = 'queued'
                self.queues[name].put(dest)

    # -------- 큐 인터페이스 --------
    def enqueue_waypoints(self, vehicle_name: str, waypoints: List[airsim.Vector3r]):
        if not waypoints:
            return
        if self.drone_states.get(vehicle_name, {}).get('status') == 'idle':
            self.drone_states[vehicle_name]['status'] = 'queued'
        self.queues[vehicle_name].put(("waypoints", waypoints))

    def is_vehicle_idle(self, vehicle_name: str) -> bool:
        st = self.drone_states.get(vehicle_name)
        if not st or st['status'] != 'idle':
            return False
        q = self.queues.get(vehicle_name)
        return (q is not None) and q.empty()

    # -------- 워커 --------
    def _vehicle_worker(self, drone_name: str):
        c = self.clients[drone_name]
        while True:
            item = self.queues[drone_name].get()
            if item is None:
                break
            if isinstance(item, tuple) and len(item) == 2 and item[0] == "waypoints":
                for dest in item[1]:
                    self._fly_to(c, drone_name, dest)
            else:
                self._fly_to(c, drone_name, item)
            self.drone_states[drone_name]['status'] = 'idle'
            self.drone_states[drone_name]['updated'] = datetime.now()

    def _fly_to(self, c, drone_name, dest):
        try:
            self.drone_states[drone_name]['status'] = 'delivering'
            st = c.getMultirotorState(vehicle_name=drone_name)
            if st.landed_state == airsim.LandedState.Landed:
                c.armDisarm(True, vehicle_name=drone_name)
                time.sleep(0.4)
                c.takeoffAsync(vehicle_name=drone_name).join()
                print(f"🚀 {drone_name}: 이륙 완료")
            c.moveToPositionAsync(dest.x_val, dest.y_val, dest.z_val,
                                  velocity=self.cruise_speed, vehicle_name=drone_name)
        except Exception as e:
            print(f"❌ {drone_name} 비행 오류: {e}")

    # ---------- 상태 ----------
    def get_simulation_status(self):
        if not self.connected:
            return "disconnected"
        if not self.simulation_running:
            return "stopped"
        return "running"

    def get_visualization_status(self):
        if not self.connected:
            return "disconnected"
        return "active"

    # ---------- 정리 ----------
    def cleanup(self):
        print("🧹 리소스 정리 시작...")
        self.stop_simulation()

        c = self._get_plot_client()
        if c:
            try:
                c.simFlushPersistentMarkers()
            except Exception:
                pass

            if self._can_spawn_objects():
                for name in list(self._spawned_objects):
                    try:
                        c.simDestroyObject(name)
                    except Exception:
                        pass
                self._spawned_objects.clear()
                print("✅ StaticMeshActor 정리 완료")

        for name, cli in self.clients.items():
            try:
                cli.armDisarm(False, vehicle_name=name)
                cli.enableApiControl(False, vehicle_name=name)
            except Exception:
                pass
        print("✅ AirSim 리소스 정리 완료")
