"""
성능 분석기
"""

import numpy as np
from typing import Dict, Any, List
import warnings
warnings.filterwarnings('ignore')

class PerformanceAnalyzer:
    """
    성능 분석기 클래스
    """
    
    def __init__(self):
        """
        초기화
        """
        pass
    
    def analyze_performance(self, simulation_results):
        """
        시뮬레이션 결과 분석
        """
        print("성능 분석 시작...")
        
        # 기본 지표 계산
        basic_metrics = self._calculate_basic_metrics(simulation_results)
        
        # 효율성 지표 계산
        efficiency_metrics = self._calculate_efficiency_metrics(simulation_results)
        
        # 안전성 지표 계산
        safety_metrics = self._calculate_safety_metrics(simulation_results)
        
        # 고객 만족도 지표 계산
        satisfaction_metrics = self._calculate_satisfaction_metrics(simulation_results)
        
        # 종합 성능 점수 계산
        overall_score = self._calculate_overall_score(
            basic_metrics, efficiency_metrics, safety_metrics, satisfaction_metrics
        )
        
        # 결과 통합
        performance_analysis = {
            'basic_metrics': basic_metrics,
            'efficiency_metrics': efficiency_metrics,
            'safety_metrics': safety_metrics,
            'satisfaction_metrics': satisfaction_metrics,
            'overall_score': overall_score
        }
        
        print("성능 분석 완료!")
        
        return performance_analysis
    
    def _calculate_basic_metrics(self, results):
        """
        기본 성능 지표 계산
        """
        total_requests = results['total_requests']
        completed_deliveries = results['completed_deliveries']
        failed_deliveries = results['failed_deliveries']
        total_cost = results['total_cost']
        total_distance = results['total_distance']
        avg_delivery_time = results['average_delivery_time']
        
        # 완료율
        completion_rate = completed_deliveries / total_requests if total_requests > 0 else 0
        
        # 평균 비용
        avg_cost_per_delivery = total_cost / completed_deliveries if completed_deliveries > 0 else 0
        
        # 평균 거리
        avg_distance_per_delivery = total_distance / completed_deliveries if completed_deliveries > 0 else 0
        
        basic_metrics = {
            'total_requests': total_requests,
            'completed_deliveries': completed_deliveries,
            'failed_deliveries': failed_deliveries,
            'completion_rate': completion_rate,
            'total_cost': total_cost,
            'total_distance': total_distance,
            'average_delivery_time': avg_delivery_time,
            'average_cost_per_delivery': avg_cost_per_delivery,
            'average_distance_per_delivery': avg_distance_per_delivery
        }
        
        return basic_metrics
    
    def _calculate_efficiency_metrics(self, results):
        """
        효율성 지표 계산
        """
        drone_statistics = results['drone_statistics']
        total_cost = results['total_cost']
        total_distance = results['total_distance']
        completed_deliveries = results['completed_deliveries']
        
        # 드론 활용률
        utilization_rates = [stats['utilization_rate'] for stats in drone_statistics.values()]
        avg_utilization_rate = np.mean(utilization_rates) if utilization_rates else 0
        
        # 에너지 효율성 (km당 비용)
        energy_efficiency = total_cost / total_distance if total_distance > 0 else 0
        
        # 시간 효율성 (배달당 평균 시간)
        time_efficiency = results['average_delivery_time']
        
        # 드론별 효율성
        drone_efficiencies = {}
        for drone_id, stats in drone_statistics.items():
            if stats['completed_deliveries'] > 0:
                efficiency = stats['completed_deliveries'] / (stats['total_cost'] + 1)  # 비용 대비 완료율
                drone_efficiencies[drone_id] = efficiency
        
        avg_drone_efficiency = np.mean(list(drone_efficiencies.values())) if drone_efficiencies else 0
        
        efficiency_metrics = {
            'average_utilization_rate': avg_utilization_rate,
            'energy_efficiency': energy_efficiency,
            'time_efficiency': time_efficiency,
            'average_drone_efficiency': avg_drone_efficiency,
            'drone_efficiencies': drone_efficiencies
        }
        
        return efficiency_metrics
    
    def _calculate_safety_metrics(self, results):
        """
        안전성 지표 계산
        """
        # 충돌 횟수 (시뮬레이션에서 기록된 경우)
        collision_count = 0  # 실제로는 시뮬레이션에서 충돌 정보를 추출해야 함
        
        # 안전 거리 준수율 (간단한 구현)
        safety_distance_compliance = 0.95  # 95% 가정
        
        # 배터리 안전성
        drone_statistics = results['drone_statistics']
        battery_safety_score = 0.9  # 90% 가정 (실제로는 배터리 레벨 데이터 필요)
        
        # 전반적 안전 점수
        safety_score = (safety_distance_compliance + battery_safety_score) / 2
        
        safety_metrics = {
            'collision_count': collision_count,
            'safety_distance_compliance': safety_distance_compliance,
            'battery_safety_score': battery_safety_score,
            'safety_score': safety_score
        }
        
        return safety_metrics
    
    def _calculate_satisfaction_metrics(self, results):
        """
        고객 만족도 지표 계산
        """
        delivery_results = results['delivery_results']
        total_requests = results['total_requests']
        
        # 시간 준수율
        on_time_deliveries = 0
        for delivery in delivery_results:
            # 30분 내 배달을 정시로 간주
            if delivery.get('delivery_time', 0) <= 30:
                on_time_deliveries += 1
        
        on_time_rate = on_time_deliveries / len(delivery_results) if delivery_results else 0
        
        # 배달 품질 점수 (거리 기반)
        quality_scores = []
        for delivery in delivery_results:
            # 거리가 짧을수록 높은 점수
            distance = delivery.get('distance', 0)
            if distance > 0:
                quality_score = max(0, 100 - distance / 10)  # 1km당 10점 감소
                quality_scores.append(quality_score)
        
        avg_quality_score = np.mean(quality_scores) if quality_scores else 0
        
        # 비용 만족도
        cost_satisfaction = 0
        if delivery_results:
            avg_cost = np.mean([d.get('delivery_cost', 0) for d in delivery_results])
            # 비용이 낮을수록 높은 만족도
            cost_satisfaction = max(0, 100 - avg_cost / 10)  # 10원당 1점 감소
        
        # 종합 고객 만족도
        customer_satisfaction = (on_time_rate * 40 + avg_quality_score * 0.4 + cost_satisfaction * 0.2)
        
        satisfaction_metrics = {
            'on_time_rate': on_time_rate,
            'average_quality_score': avg_quality_score,
            'cost_satisfaction': cost_satisfaction,
            'customer_satisfaction': customer_satisfaction
        }
        
        return satisfaction_metrics
    
    def _calculate_overall_score(self, basic_metrics, efficiency_metrics, safety_metrics, satisfaction_metrics):
        """
        종합 성능 점수 계산
        """
        # 각 지표별 가중치
        weights = {
            'completion_rate': 0.25,
            'efficiency': 0.25,
            'safety': 0.20,
            'satisfaction': 0.30
        }
        
        # 완료율 점수 (0-100)
        completion_score = basic_metrics['completion_rate'] * 100
        
        # 효율성 점수 (0-100)
        efficiency_score = (
            efficiency_metrics['average_utilization_rate'] * 40 +
            (1 / (1 + efficiency_metrics['energy_efficiency'])) * 30 +
            (1 / (1 + efficiency_metrics['time_efficiency'])) * 30
        )
        
        # 안전성 점수 (0-100)
        safety_score = safety_metrics['safety_score'] * 100
        
        # 만족도 점수 (0-100)
        satisfaction_score = satisfaction_metrics['customer_satisfaction']
        
        # 종합 점수 계산
        overall_score = (
            completion_score * weights['completion_rate'] +
            efficiency_score * weights['efficiency'] +
            safety_score * weights['safety'] +
            satisfaction_score * weights['satisfaction']
        )
        
        return {
            'overall_score': overall_score,
            'completion_score': completion_score,
            'efficiency_score': efficiency_score,
            'safety_score': safety_score,
            'satisfaction_score': satisfaction_score,
            'weights': weights
        }
    
    def generate_performance_report(self, performance_analysis):
        """
        성능 보고서 생성
        """
        report = []
        report.append("=" * 50)
        report.append("드론 배달 시스템 성능 분석 보고서")
        report.append("=" * 50)
        
        # 기본 지표
        basic = performance_analysis['basic_metrics']
        report.append("\n1. 기본 성능 지표")
        report.append("-" * 30)
        report.append(f"총 요청 수: {basic['total_requests']}")
        report.append(f"완료된 배달: {basic['completed_deliveries']}")
        report.append(f"실패한 배달: {basic['failed_deliveries']}")
        report.append(f"완료율: {basic['completion_rate']:.2%}")
        report.append(f"총 비용: {basic['total_cost']:,.0f}원")
        report.append(f"총 거리: {basic['total_distance']:,.0f}m")
        report.append(f"평균 배달 시간: {basic['average_delivery_time']:.1f}분")
        report.append(f"배달당 평균 비용: {basic['average_cost_per_delivery']:,.0f}원")
        
        # 효율성 지표
        efficiency = performance_analysis['efficiency_metrics']
        report.append("\n2. 효율성 지표")
        report.append("-" * 30)
        report.append(f"평균 드론 활용률: {efficiency['average_utilization_rate']:.2%}")
        report.append(f"에너지 효율성: {efficiency['energy_efficiency']:.2f}원/km")
        report.append(f"시간 효율성: {efficiency['time_efficiency']:.1f}분/배달")
        report.append(f"평균 드론 효율성: {efficiency['average_drone_efficiency']:.3f}")
        
        # 안전성 지표
        safety = performance_analysis['safety_metrics']
        report.append("\n3. 안전성 지표")
        report.append("-" * 30)
        report.append(f"충돌 횟수: {safety['collision_count']}")
        report.append(f"안전 거리 준수율: {safety['safety_distance_compliance']:.2%}")
        report.append(f"배터리 안전 점수: {safety['battery_safety_score']:.2%}")
        report.append(f"전체 안전 점수: {safety['safety_score']:.2%}")
        
        # 고객 만족도
        satisfaction = performance_analysis['satisfaction_metrics']
        report.append("\n4. 고객 만족도 지표")
        report.append("-" * 30)
        report.append(f"정시 배달율: {satisfaction['on_time_rate']:.2%}")
        report.append(f"평균 품질 점수: {satisfaction['average_quality_score']:.1f}/100")
        report.append(f"비용 만족도: {satisfaction['cost_satisfaction']:.1f}/100")
        report.append(f"고객 만족도: {satisfaction['customer_satisfaction']:.1f}/100")
        
        # 종합 점수
        overall = performance_analysis['overall_score']
        report.append("\n5. 종합 성능 점수")
        report.append("-" * 30)
        report.append(f"완료율 점수: {overall['completion_score']:.1f}/100")
        report.append(f"효율성 점수: {overall['efficiency_score']:.1f}/100")
        report.append(f"안전성 점수: {overall['safety_score']:.1f}/100")
        report.append(f"만족도 점수: {overall['satisfaction_score']:.1f}/100")
        report.append(f"종합 점수: {overall['overall_score']:.1f}/100")
        
        # 등급 평가
        grade = self._evaluate_grade(overall['overall_score'])
        report.append(f"성능 등급: {grade}")
        
        report.append("\n" + "=" * 50)
        
        return "\n".join(report)
    
    def _evaluate_grade(self, score):
        """
        점수에 따른 등급 평가
        """
        if score >= 90:
            return "A+ (우수)"
        elif score >= 80:
            return "A (양호)"
        elif score >= 70:
            return "B+ (보통)"
        elif score >= 60:
            return "B (미흡)"
        else:
            return "C (불량)"
    
    def compare_algorithms(self, algorithm_results):
        """
        여러 알고리즘의 성능 비교
        """
        comparison = {}
        
        for algorithm_name, results in algorithm_results.items():
            performance = self.analyze_performance(results)
            comparison[algorithm_name] = {
                'overall_score': performance['overall_score']['overall_score'],
                'completion_rate': performance['basic_metrics']['completion_rate'],
                'total_cost': performance['basic_metrics']['total_cost'],
                'average_delivery_time': performance['basic_metrics']['average_delivery_time'],
                'customer_satisfaction': performance['satisfaction_metrics']['customer_satisfaction']
            }
        
        return comparison
    
    def generate_comparison_report(self, comparison_results):
        """
        알고리즘 비교 보고서 생성
        """
        report = []
        report.append("=" * 60)
        report.append("알고리즘 성능 비교 보고서")
        report.append("=" * 60)
        
        # 헤더
        report.append(f"{'알고리즘':<15} {'종합점수':<10} {'완료율':<10} {'총비용':<12} {'평균시간':<10} {'고객만족도':<12}")
        report.append("-" * 60)
        
        # 각 알고리즘별 결과
        for algorithm, metrics in comparison_results.items():
            report.append(
                f"{algorithm:<15} "
                f"{metrics['overall_score']:<10.1f} "
                f"{metrics['completion_rate']:<10.2%} "
                f"{metrics['total_cost']:<12,.0f} "
                f"{metrics['average_delivery_time']:<10.1f} "
                f"{metrics['customer_satisfaction']:<12.1f}"
            )
        
        # 최고 성능 알고리즘
        best_algorithm = max(comparison_results.keys(), 
                           key=lambda x: comparison_results[x]['overall_score'])
        
        report.append("\n" + "=" * 60)
        report.append(f"최고 성능 알고리즘: {best_algorithm}")
        report.append(f"종합 점수: {comparison_results[best_algorithm]['overall_score']:.1f}/100")
        
        return "\n".join(report) 