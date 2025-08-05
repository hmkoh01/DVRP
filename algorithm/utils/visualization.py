"""
시각화 도구
드론 배달 시스템의 시각화를 담당하는 클래스
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class VisualizationTool:
    """
    시각화 도구 클래스
    """
    
    def __init__(self, building_data_loader=None):
        """
        초기화
        """
        self.data_loader = building_data_loader
        self.fig_size = (15, 12)
        self.colors = {
            'buildings': 'lightgray',
            'restaurants': 'red',
            'residential': 'blue',
            'depots': 'green',
            'drones': 'orange',
            'routes': 'purple',
            'pending_requests': 'yellow',
            'completed_requests': 'green',
            'failed_requests': 'red'
        }
    
    def plot_buildings_3d(self, ax=None, show_restaurants=True, show_residential=True):
        """
        3D 건물 시각화
        """
        if self.data_loader is None or self.data_loader.buildings is None:
            print("건물 데이터가 없습니다.")
            return None
        
        if ax is None:
            fig = plt.figure(figsize=self.fig_size)
            ax = fig.add_subplot(111, projection='3d')
        
        # 모든 건물 표시
        buildings = self.data_loader.buildings
        ax.scatter(buildings['longitude'], buildings['latitude'], buildings['height_m'],
                  c=self.colors['buildings'], s=20, alpha=0.6, label='Buildings')
        
        # 식당 표시
        if show_restaurants and self.data_loader.restaurants is not None:
            restaurants = self.data_loader.restaurants
            ax.scatter(restaurants['longitude'], restaurants['latitude'], restaurants['height_m'],
                      c=self.colors['restaurants'], s=100, marker='^', label='Restaurants')
        
        # 주거용 건물 표시
        if show_residential and self.data_loader.residential_buildings is not None:
            residential = self.data_loader.residential_buildings
            ax.scatter(residential['longitude'], residential['latitude'], residential['height_m'],
                      c=self.colors['residential'], s=80, marker='s', label='Residential')
        
        ax.set_xlabel('Longitude', fontsize=12)
        ax.set_ylabel('Latitude', fontsize=12)
        ax.set_zlabel('Height (m)', fontsize=12)
        ax.set_title('POSTECH Buildings 3D View', fontsize=16, fontweight='bold')
        ax.legend()
        
        return ax
    
    def plot_depots(self, depots, ax=None):
        """
        Depot 위치 시각화
        """
        if not depots:
            print("Depot 데이터가 없습니다.")
            return None
        
        if ax is None:
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111)
        
        # 건물들 배경으로 표시
        if self.data_loader and self.data_loader.buildings is not None:
            ax.scatter(self.data_loader.buildings['longitude'], 
                      self.data_loader.buildings['latitude'],
                      c=self.colors['buildings'], s=10, alpha=0.4)
        
        # Depot 표시
        depot_lons = [depot['longitude'] for depot in depots]
        depot_lats = [depot['latitude'] for depot in depots]
        
        ax.scatter(depot_lons, depot_lats, 
                  c=self.colors['depots'], s=300, marker='s', 
                  edgecolors='black', linewidth=2, label='Depots')
        
        # Depot ID 표시
        for depot in depots:
            ax.annotate(f"D{depot['depot_id']}", 
                       (depot['longitude'], depot['latitude']),
                       xytext=(10, 10), textcoords='offset points',
                       fontsize=14, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        ax.set_xlabel('Longitude', fontsize=12)
        ax.set_ylabel('Latitude', fontsize=12)
        ax.set_title('Depot Locations', fontsize=16, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def plot_delivery_requests(self, requests, ax=None, show_status=True):
        """
        배달 요청 시각화
        """
        if not requests:
            print("배달 요청이 없습니다.")
            return None
        
        if ax is None:
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111)
        
        # 건물들 배경으로 표시
        if self.data_loader and self.data_loader.buildings is not None:
            ax.scatter(self.data_loader.buildings['longitude'], 
                      self.data_loader.buildings['latitude'],
                      c=self.colors['buildings'], s=10, alpha=0.4)
        
        # 요청별로 화살표 그리기
        for request in requests:
            # 출발점 (식당)
            start_lon = request['restaurant_location']['longitude']
            start_lat = request['restaurant_location']['latitude']
            
            # 도착점 (고객)
            end_lon = request['customer_location']['longitude']
            end_lat = request['customer_location']['latitude']
            
            # 상태별 색상 결정
            if show_status:
                if request['status'] == 'pending':
                    color = self.colors['pending_requests']
                elif request['status'] == 'completed':
                    color = self.colors['completed_requests']
                elif request['status'] == 'failed':
                    color = self.colors['failed_requests']
                else:
                    color = 'gray'
            else:
                color = self.colors['routes']
            
            # 화살표 그리기
            ax.annotate('', xy=(end_lon, end_lat), xytext=(start_lon, start_lat),
                       arrowprops=dict(arrowstyle='->', color=color, lw=1.5, alpha=0.7))
            
            # 요청 ID 표시
            ax.annotate(f"R{request['request_id']}", 
                       ((start_lon + end_lon)/2, (start_lat + end_lat)/2),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8, alpha=0.8)
        
        ax.set_xlabel('Longitude', fontsize=12)
        ax.set_ylabel('Latitude', fontsize=12)
        ax.set_title('Delivery Requests', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def plot_drone_routes(self, routes, depots, ax=None):
        """
        드론 경로 시각화
        """
        if not routes:
            print("경로 데이터가 없습니다.")
            return None
        
        if ax is None:
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111)
        
        # 건물들 배경으로 표시
        if self.data_loader and self.data_loader.buildings is not None:
            ax.scatter(self.data_loader.buildings['longitude'], 
                      self.data_loader.buildings['latitude'],
                      c=self.colors['buildings'], s=10, alpha=0.4)
        
        # Depot 표시
        if depots:
            depot_lons = [depot['longitude'] for depot in depots]
            depot_lats = [depot['latitude'] for depot in depots]
            ax.scatter(depot_lons, depot_lats, 
                      c=self.colors['depots'], s=200, marker='s', 
                      edgecolors='black', linewidth=2, label='Depots')
        
        # 드론별 경로 그리기
        colors = plt.cm.Set3(np.linspace(0, 1, len(routes)))
        
        for i, route in enumerate(routes):
            drone_id = route.get('drone_id', i)
            path = route.get('path', [])
            
            if len(path) < 2:
                continue
            
            # 경로 좌표 추출
            lons = [point['longitude'] for point in path]
            lats = [point['latitude'] for point in path]
            
            # 경로 선 그리기
            ax.plot(lons, lats, color=colors[i], linewidth=2, 
                   label=f'Drone {drone_id}', alpha=0.8)
            
            # 드론 위치 표시
            if path:
                current_pos = path[-1]  # 현재 위치
                ax.scatter(current_pos['longitude'], current_pos['latitude'],
                          c=colors[i], s=100, marker='o', edgecolors='black')
        
        ax.set_xlabel('Longitude', fontsize=12)
        ax.set_ylabel('Latitude', fontsize=12)
        ax.set_title('Drone Routes', fontsize=16, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def plot_simulation_results(self, simulation_data, figsize=(15, 10)):
        """
        시뮬레이션 결과 시각화
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # 1. 시간별 완료된 배달 수
        if 'delivery_completion_times' in simulation_data:
            completion_times = simulation_data['delivery_completion_times']
            axes[0, 0].hist(completion_times, bins=20, alpha=0.7, color='green')
            axes[0, 0].set_xlabel('Completion Time (minutes)')
            axes[0, 0].set_ylabel('Number of Deliveries')
            axes[0, 0].set_title('Delivery Completion Times')
            axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 드론별 배달 수
        if 'drone_delivery_counts' in simulation_data:
            drone_counts = simulation_data['drone_delivery_counts']
            drone_ids = list(drone_counts.keys())
            counts = list(drone_counts.values())
            axes[0, 1].bar(drone_ids, counts, alpha=0.7, color='orange')
            axes[0, 1].set_xlabel('Drone ID')
            axes[0, 1].set_ylabel('Number of Deliveries')
            axes[0, 1].set_title('Deliveries per Drone')
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 시간별 활성 드론 수
        if 'active_drones_over_time' in simulation_data:
            time_data = simulation_data['active_drones_over_time']
            times = list(time_data.keys())
            active_drones = list(time_data.values())
            axes[1, 0].plot(times, active_drones, marker='o', alpha=0.7, color='blue')
            axes[1, 0].set_xlabel('Time (minutes)')
            axes[1, 0].set_ylabel('Active Drones')
            axes[1, 0].set_title('Active Drones Over Time')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 총 비용 분포
        if 'total_costs' in simulation_data:
            costs = simulation_data['total_costs']
            axes[1, 1].hist(costs, bins=15, alpha=0.7, color='red')
            axes[1, 1].set_xlabel('Total Cost')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].set_title('Total Cost Distribution')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def create_interactive_3d_plot(self, buildings=None, depots=None, routes=None, requests=None):
        """
        인터랙티브 3D 플롯 생성 (Plotly)
        """
        fig = go.Figure()
        
        # 건물들 표시
        if buildings is not None:
            fig.add_trace(go.Scatter3d(
                x=buildings['longitude'],
                y=buildings['latitude'],
                z=buildings['height_m'],
                mode='markers',
                marker=dict(
                    size=3,
                    color=self.colors['buildings'],
                    opacity=0.6
                ),
                name='Buildings'
            ))
        
        # Depot 표시
        if depots:
            depot_lons = [depot['longitude'] for depot in depots]
            depot_lats = [depot['latitude'] for depot in depots]
            depot_heights = [depot.get('height', 50) for depot in depots]
            
            fig.add_trace(go.Scatter3d(
                x=depot_lons,
                y=depot_lats,
                z=depot_heights,
                mode='markers',
                marker=dict(
                    size=10,
                    color=self.colors['depots'],
                    symbol='diamond'
                ),
                name='Depots'
            ))
        
        # 배달 요청 표시
        if requests:
            for request in requests:
                # 식당 위치
                fig.add_trace(go.Scatter3d(
                    x=[request['restaurant_location']['longitude']],
                    y=[request['restaurant_location']['latitude']],
                    z=[request['restaurant_location']['height']],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=self.colors['restaurants'],
                        symbol='triangle-up'
                    ),
                    name=f"Restaurant {request['request_id']}",
                    showlegend=False
                ))
                
                # 고객 위치
                fig.add_trace(go.Scatter3d(
                    x=[request['customer_location']['longitude']],
                    y=[request['customer_location']['latitude']],
                    z=[request['customer_location']['height']],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=self.colors['residential'],
                        symbol='square'
                    ),
                    name=f"Customer {request['request_id']}",
                    showlegend=False
                ))
        
        # 드론 경로 표시
        if routes:
            for i, route in enumerate(routes):
                path = route.get('path', [])
                if len(path) >= 2:
                    lons = [point['longitude'] for point in path]
                    lats = [point['latitude'] for point in path]
                    heights = [point.get('height', 50) for point in path]
                    
                    fig.add_trace(go.Scatter3d(
                        x=lons,
                        y=lats,
                        z=heights,
                        mode='lines+markers',
                        line=dict(width=3),
                        marker=dict(size=5),
                        name=f"Drone {route.get('drone_id', i)}"
                    ))
        
        fig.update_layout(
            title='Drone Delivery System 3D View',
            scene=dict(
                xaxis_title='Longitude',
                yaxis_title='Latitude',
                zaxis_title='Height (m)'
            ),
            width=1000,
            height=800
        )
        
        return fig
    
    def plot_performance_metrics(self, metrics_data, figsize=(15, 10)):
        """
        성능 지표 시각화
        """
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        
        # 1. 총 비용
        if 'total_cost' in metrics_data:
            axes[0, 0].bar(['Total Cost'], [metrics_data['total_cost']], color='red', alpha=0.7)
            axes[0, 0].set_title('Total Cost')
            axes[0, 0].set_ylabel('Cost (KRW)')
        
        # 2. 평균 배달 시간
        if 'average_delivery_time' in metrics_data:
            axes[0, 1].bar(['Avg Delivery Time'], [metrics_data['average_delivery_time']], color='blue', alpha=0.7)
            axes[0, 1].set_title('Average Delivery Time')
            axes[0, 1].set_ylabel('Time (minutes)')
        
        # 3. 드론 활용률
        if 'drone_utilization_rate' in metrics_data:
            axes[0, 2].bar(['Utilization Rate'], [metrics_data['drone_utilization_rate']], color='green', alpha=0.7)
            axes[0, 2].set_title('Drone Utilization Rate')
            axes[0, 2].set_ylabel('Rate (%)')
            axes[0, 2].set_ylim(0, 100)
        
        # 4. 에너지 효율성
        if 'energy_efficiency' in metrics_data:
            axes[1, 0].bar(['Energy Efficiency'], [metrics_data['energy_efficiency']], color='orange', alpha=0.7)
            axes[1, 0].set_title('Energy Efficiency')
            axes[1, 0].set_ylabel('Efficiency (%)')
            axes[1, 0].set_ylim(0, 100)
        
        # 5. 고객 만족도
        if 'customer_satisfaction' in metrics_data:
            axes[1, 1].bar(['Customer Satisfaction'], [metrics_data['customer_satisfaction']], color='purple', alpha=0.7)
            axes[1, 1].set_title('Customer Satisfaction')
            axes[1, 1].set_ylabel('Satisfaction (%)')
            axes[1, 1].set_ylim(0, 100)
        
        # 6. 안전 점수
        if 'safety_score' in metrics_data:
            axes[1, 2].bar(['Safety Score'], [metrics_data['safety_score']], color='brown', alpha=0.7)
            axes[1, 2].set_title('Safety Score')
            axes[1, 2].set_ylabel('Score (%)')
            axes[1, 2].set_ylim(0, 100)
        
        plt.tight_layout()
        return fig
    
    def save_visualization(self, fig, filename, dpi=300):
        """
        시각화 결과 저장
        """
        try:
            fig.savefig(filename, dpi=dpi, bbox_inches='tight')
            print(f"시각화 저장: {filename}")
        except Exception as e:
            print(f"시각화 저장 실패: {e}")
    
    def create_animation(self, simulation_frames, output_path):
        """
        시뮬레이션 애니메이션 생성
        """
        try:
            import matplotlib.animation as animation
            
            fig, ax = plt.subplots(figsize=(12, 10))
            
            def animate(frame):
                ax.clear()
                
                # 현재 프레임의 데이터로 시각화
                frame_data = simulation_frames[frame]
                
                # 건물들 표시
                if self.data_loader and self.data_loader.buildings is not None:
                    ax.scatter(self.data_loader.buildings['longitude'], 
                              self.data_loader.buildings['latitude'],
                              c=self.colors['buildings'], s=10, alpha=0.4)
                
                # 드론 위치 표시
                if 'drone_positions' in frame_data:
                    for drone_id, position in frame_data['drone_positions'].items():
                        ax.scatter(position['longitude'], position['latitude'],
                                 c=self.colors['drones'], s=100, marker='o')
                        ax.annotate(f'D{drone_id}', (position['longitude'], position['latitude']),
                                  xytext=(5, 5), textcoords='offset points')
                
                ax.set_title(f'Simulation Frame {frame}')
                ax.set_xlabel('Longitude')
                ax.set_ylabel('Latitude')
                ax.grid(True, alpha=0.3)
            
            anim = animation.FuncAnimation(fig, animate, frames=len(simulation_frames), 
                                         interval=200, repeat=True)
            
            # 애니메이션 저장
            anim.save(output_path, writer='pillow')
            print(f"애니메이션 저장: {output_path}")
            
        except Exception as e:
            print(f"애니메이션 생성 실패: {e}") 