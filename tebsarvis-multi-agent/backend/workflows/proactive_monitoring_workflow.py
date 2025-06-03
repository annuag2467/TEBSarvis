"""
Proactive Monitoring Workflow - Auto alert/pattern detection and notification system
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Set
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field

from ..agents.core.base_agent import BaseAgent
from ..agents.core.agent_registry import get_global_registry
from ..agents.core.agent_communication import MessageBus, AgentCommunicator
from ..agents.core.message_types import TaskType, Priority, create_task_request
from ..agents.orchestrator.agent_coordinator import AgentCoordinator
from ..config.agent_config import get_agent_config, AgentType

logger = logging.getLogger(__name__)

class MonitoringStatus(Enum):
    INACTIVE = "inactive"
    ACTIVE = "active"
    PAUSED = "paused"
    ERROR = "error"

class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

@dataclass
class MonitoringRule:
    """Monitoring rule configuration"""
    rule_id: str
    name: str
    description: str
    rule_type: str  # volume_spike, pattern_anomaly, threshold
    conditions: Dict[str, Any]
    severity: AlertSeverity
    enabled: bool = True
    cooldown_minutes: int = 30
    notification_channels: List[str] = field(default_factory=lambda: ['email'])

@dataclass
class MonitoringAlert:
    """Generated monitoring alert"""
    alert_id: str
    rule_id: str
    severity: AlertSeverity
    title: str
    description: str
    data: Dict[str, Any]
    created_at: datetime
    status: str = "active"

class ProactiveMonitoringWorkflow:
    """
    Orchestrates proactive monitoring and alerting:
    1. Real-time incident monitoring
    2. Pattern detection and anomaly analysis
    3. Alert generation and evaluation
    4. Notification dispatch
    5. Escalation management
    """
    
    def __init__(self):
        self.registry = get_global_registry()
        self.message_bus = MessageBus()
        self.coordinator = AgentCoordinator(self.registry, self.message_bus)
        self.communicator = AgentCommunicator("proactive_monitoring_workflow", self.message_bus)
        self.config = get_agent_config()
        
        # Monitoring state
        self.status = MonitoringStatus.INACTIVE
        self.monitoring_rules: Dict[str, MonitoringRule] = {}
        self.active_alerts: Dict[str, MonitoringAlert] = {}
        self.alert_history: List[MonitoringAlert] = []
        
        # Configuration
        self.monitoring_interval = 60  # seconds
        self.pattern_analysis_interval = 300  # 5 minutes
        self.anomaly_detection_interval = 180  # 3 minutes
        self.max_alerts_per_rule = 5
        self.alert_retention_hours = 24
        
        # Background tasks
        self.background_tasks: List[asyncio.Task] = []
        
        self.logger = logging.getLogger("workflows.proactive_monitoring")
        
        # Load default monitoring rules
        self._load_default_monitoring_rules()
    
    async def start(self):
        """Start the proactive monitoring workflow"""
        try:
            await self.message_bus.start()
            await self.coordinator.start()
            
            # Start background monitoring tasks
            self.background_tasks = [
                asyncio.create_task(self._real_time_monitoring_loop()),
                asyncio.create_task(self._pattern_analysis_loop()),
                asyncio.create_task(self._anomaly_detection_loop()),
                asyncio.create_task(self._alert_management_loop()),
                asyncio.create_task(self._cleanup_loop())
            ]
            
            self.status = MonitoringStatus.ACTIVE
            self.logger.info("Proactive Monitoring Workflow started")
            
        except Exception as e:
            self.logger.error(f"Failed to start proactive monitoring: {str(e)}")
            raise
    
    async def stop(self):
        """Stop the proactive monitoring workflow"""
        try:
            self.status = MonitoringStatus.INACTIVE
            
            # Cancel background tasks
            for task in self.background_tasks:
                if not task.done():
                    task.cancel()
            
            # Wait for tasks to complete
            if self.background_tasks:
                await asyncio.gather(*self.background_tasks, return_exceptions=True)
            
            await self.coordinator.stop()
            await self.message_bus.stop()
            
            self.logger.info("Proactive Monitoring Workflow stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping proactive monitoring: {str(e)}")
    
    async def pause_monitoring(self):
        """Pause monitoring activities"""
        self.status = MonitoringStatus.PAUSED
        self.logger.info("Monitoring paused")
    
    async def resume_monitoring(self):
        """Resume monitoring activities"""
        if self.status == MonitoringStatus.PAUSED:
            self.status = MonitoringStatus.ACTIVE
            self.logger.info("Monitoring resumed")
    
    async def execute_monitoring_cycle(self) -> Dict[str, Any]:
        """Execute a single monitoring cycle manually"""
        try:
            cycle_start = datetime.now()
            cycle_results = {
                'cycle_id': f"manual_{cycle_start.strftime('%Y%m%d_%H%M%S')}",
                'started_at': cycle_start.isoformat(),
                'results': {}
            }
            
            # Step 1: Real-time monitoring
            monitoring_result = await self._execute_real_time_monitoring()
            cycle_results['results']['real_time_monitoring'] = monitoring_result
            
            # Step 2: Pattern detection
            pattern_result = await self._execute_pattern_detection()
            cycle_results['results']['pattern_detection'] = pattern_result
            
            # Step 3: Anomaly detection
            anomaly_result = await self._execute_anomaly_detection()
            cycle_results['results']['anomaly_detection'] = anomaly_result
            
            # Step 4: Alert evaluation
            alert_result = await self._evaluate_and_generate_alerts(
                monitoring_result, pattern_result, anomaly_result
            )
            cycle_results['results']['alert_evaluation'] = alert_result
            
            # Step 5: Notification dispatch
            if alert_result.get('alerts_generated'):
                notification_result = await self._dispatch_notifications(alert_result['alerts_generated'])
                cycle_results['results']['notification_dispatch'] = notification_result
            
            cycle_results['completed_at'] = datetime.now().isoformat()
            cycle_results['duration'] = (datetime.now() - cycle_start).total_seconds()
            cycle_results['status'] = 'completed'
            
            self.logger.info(f"Monitoring cycle completed in {cycle_results['duration']:.2f} seconds")
            return cycle_results
            
        except Exception as e:
            self.logger.error(f"Error in monitoring cycle: {str(e)}")
            cycle_results['status'] = 'failed'
            cycle_results['error'] = str(e)
            return cycle_results
    
    async def _real_time_monitoring_loop(self):
        """Background task for real-time monitoring"""
        while self.status != MonitoringStatus.INACTIVE:
            try:
                if self.status == MonitoringStatus.ACTIVE:
                    await self._execute_real_time_monitoring()
                
                await asyncio.sleep(self.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"Error in real-time monitoring loop: {str(e)}")
                await asyncio.sleep(30)  # Short retry interval
    
    async def _pattern_analysis_loop(self):
        """Background task for pattern analysis"""
        while self.status != MonitoringStatus.INACTIVE:
            try:
                if self.status == MonitoringStatus.ACTIVE:
                    await self._execute_pattern_detection()
                
                await asyncio.sleep(self.pattern_analysis_interval)
                
            except Exception as e:
                self.logger.error(f"Error in pattern analysis loop: {str(e)}")
                await asyncio.sleep(60)
    
    async def _anomaly_detection_loop(self):
        """Background task for anomaly detection"""
        while self.status != MonitoringStatus.INACTIVE:
            try:
                if self.status == MonitoringStatus.ACTIVE:
                    await self._execute_anomaly_detection()
                
                await asyncio.sleep(self.anomaly_detection_interval)
                
            except Exception as e:
                self.logger.error(f"Error in anomaly detection loop: {str(e)}")
                await asyncio.sleep(60)
    
    async def _alert_management_loop(self):
        """Background task for alert management and escalation"""
        while self.status != MonitoringStatus.INACTIVE:
            try:
                if self.status == MonitoringStatus.ACTIVE:
                    await self._manage_alert_lifecycle()
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Error in alert management loop: {str(e)}")
                await asyncio.sleep(60)
    
    async def _cleanup_loop(self):
        """Background task for cleanup operations"""
        while self.status != MonitoringStatus.INACTIVE:
            try:
                if self.status == MonitoringStatus.ACTIVE:
                    await self._cleanup_old_alerts()
                
                await asyncio.sleep(3600)  # Run hourly
                
            except Exception as e:
                self.logger.error(f"Error in cleanup loop: {str(e)}")
                await asyncio.sleep(300)
    
    async def _execute_real_time_monitoring(self) -> Dict[str, Any]:
        """Execute real-time monitoring using Alerting Agent"""
        try:
            # Find Alerting Agent
            alerting_agent = self.registry.get_best_agent_for_capability('real_time_monitoring')
            if not alerting_agent:
                return {'error': 'No alerting agent available'}
            
            # Get enabled monitoring rules
            enabled_rules = [rule_id for rule_id, rule in self.monitoring_rules.items() if rule.enabled]
            
            task_data = {
                'monitoring_window': {'minutes': self.monitoring_interval // 60},
                'rule_ids': enabled_rules,
                'include_metrics': True
            }
            
            response = await self.communicator.send_task_request(
                recipient_id=alerting_agent.agent_id,
                task_type='real_time_monitoring',
                task_data=task_data,
                timeout_seconds=90
            )
            
            if response.success:
                result = response.result_data
                
                # Process any generated alerts
                if result.get('generated_alerts'):
                    await self._process_generated_alerts(result['generated_alerts'])
                
                return result
            else:
                return {'error': f'Real-time monitoring failed: {response.error_message}'}
                
        except Exception as e:
            self.logger.error(f"Error in real-time monitoring: {str(e)}")
            return {'error': str(e)}
    
    async def _execute_pattern_detection(self) -> Dict[str, Any]:
        """Execute pattern detection using Pattern Detection Agent"""
        try:
            # Find Pattern Detection Agent
            pattern_agent = self.registry.get_best_agent_for_capability('trend_analysis')
            if not pattern_agent:
                return {'error': 'No pattern detection agent available'}
            
            task_data = {
                'analysis_period': {'days': 7},  # Analyze last week
                'trend_types': ['volume', 'category', 'severity'],
                'granularity': 'hourly',
                'anomaly_detection': True
            }
            
            response = await self.communicator.send_task_request(
                recipient_id=pattern_agent.agent_id,
                task_type='trend_analysis',
                task_data=task_data,
                timeout_seconds=300
            )
            
            if response.success:
                result = response.result_data
                
                # Analyze trends for alert conditions
                await self._analyze_trends_for_alerts(result)
                
                return result
            else:
                return {'error': f'Pattern detection failed: {response.error_message}'}
                
        except Exception as e:
            self.logger.error(f"Error in pattern detection: {str(e)}")
            return {'error': str(e)}
    
    async def _execute_anomaly_detection(self) -> Dict[str, Any]:
        """Execute anomaly detection using Pattern Detection Agent"""
        try:
            # Find Pattern Detection Agent
            pattern_agent = self.registry.get_best_agent_for_capability('anomaly_detection')
            if not pattern_agent:
                return {'error': 'No anomaly detection agent available'}
            
            task_data = {
                'detection_window': {'days': 1},
                'baseline_period': {'days': 30},
                'sensitivity': 'medium',
                'anomaly_types': ['volume', 'pattern', 'timing', 'category']
            }
            
            response = await self.communicator.send_task_request(
                recipient_id=pattern_agent.agent_id,
                task_type='anomaly_detection',
                task_data=task_data,
                timeout_seconds=240
            )
            
            if response.success:
                result = response.result_data
                
                # Process anomalies for alert generation
                await self._process_anomalies_for_alerts(result)
                
                return result
            else:
                return {'error': f'Anomaly detection failed: {response.error_message}'}
                
        except Exception as e:
            self.logger.error(f"Error in anomaly detection: {str(e)}")
            return {'error': str(e)}
    
    async def _evaluate_and_generate_alerts(self, monitoring_result: Dict[str, Any],
                                          pattern_result: Dict[str, Any],
                                          anomaly_result: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate monitoring data and generate alerts"""
        try:
            # Find Alerting Agent for alert evaluation
            alerting_agent = self.registry.get_best_agent_for_capability('threshold_monitoring')
            if not alerting_agent:
                return {'error': 'No alerting agent available for evaluation'}
            
            # Combine all monitoring data
            combined_data = {
                'real_time_data': monitoring_result,
                'pattern_data': pattern_result,
                'anomaly_data': anomaly_result,
                'timestamp': datetime.now().isoformat()
            }
            
            # Define alert conditions based on monitoring rules
            alert_conditions = self._build_alert_conditions()
            
            task_data = {
                'conditions': alert_conditions,
                'data': combined_data,
                'evaluation_context': {
                    'monitoring_window': 'current_cycle',
                    'severity_thresholds': self._get_severity_thresholds()
                }
            }
            
            response = await self.communicator.send_task_request(
                recipient_id=alerting_agent.agent_id,
                task_type='evaluate_conditions',
                task_data=task_data,
                timeout_seconds=120
            )
            
            if response.success:
                result = response.result_data
                
                # Generate alerts for triggered conditions
                alerts_generated = []
                if result.get('overall_triggered'):
                    alerts_generated = await self._create_alerts_from_conditions(
                        result['condition_results'], combined_data
                    )
                
                return {
                    'evaluation_result': result,
                    'alerts_generated': alerts_generated,
                    'alert_count': len(alerts_generated)
                }
            else:
                return {'error': f'Alert evaluation failed: {response.error_message}'}
                
        except Exception as e:
            self.logger.error(f"Error in alert evaluation: {str(e)}")
            return {'error': str(e)}
    
    async def _dispatch_notifications(self, alerts: List[MonitoringAlert]) -> Dict[str, Any]:
        """Dispatch notifications for generated alerts"""
        try:
            # Find Alerting Agent for notification dispatch
            alerting_agent = self.registry.get_best_agent_for_capability('notification_dispatch')
            if not alerting_agent:
                return {'error': 'No alerting agent available for notifications'}
            
            # Prepare alerts for notification
            alert_data = []
            notification_channels = set()
            
            for alert in alerts:
                alert_dict = {
                    'alert_id': alert.alert_id,
                    'severity': alert.severity.value,
                    'title': alert.title,
                    'description': alert.description,
                    'created_at': alert.created_at.isoformat(),
                    'data': alert.data
                }
                alert_data.append(alert_dict)
                
                # Get notification channels for this alert's rule
                rule = self.monitoring_rules.get(alert.rule_id)
                if rule:
                    notification_channels.update(rule.notification_channels)
            
            # Default channels if none specified
            if not notification_channels:
                notification_channels = {'email', 'webhook'}
            
            task_data = {
                'alerts': alert_data,
                'notification_channels': list(notification_channels),
                'override_config': {
                    'urgent_delivery': any(alert.severity in [AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY] 
                                         for alert in alerts)
                }
            }
            
            response = await self.communicator.send_task_request(
                recipient_id=alerting_agent.agent_id,
                task_type='notification_dispatch',
                task_data=task_data,
                timeout_seconds=60
            )
            
            if response.success:
                # Update alert status after successful notification
                for alert in alerts:
                    alert.status = 'notified'
                
                return response.result_data
            else:
                return {'error': f'Notification dispatch failed: {response.error_message}'}
                
        except Exception as e:
            self.logger.error(f"Error in notification dispatch: {str(e)}")
            return {'error': str(e)}
    
    async def _process_generated_alerts(self, generated_alerts: List[Dict[str, Any]]):
        """Process alerts generated by monitoring agents"""
        for alert_data in generated_alerts:
            try:
                # Create monitoring alert object
                alert = MonitoringAlert(
                    alert_id=alert_data.get('alert_id', f"alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
                    rule_id=alert_data.get('rule_id', 'unknown'),
                    severity=AlertSeverity(alert_data.get('severity', 'warning')),
                    title=alert_data.get('title', 'Monitoring Alert'),
                    description=alert_data.get('description', ''),
                    data=alert_data.get('data', {}),
                    created_at=datetime.now()
                )
                
                # Add to active alerts
                self.active_alerts[alert.alert_id] = alert
                
                self.logger.info(f"Processed alert: {alert.alert_id} - {alert.title}")
                
            except Exception as e:
                self.logger.error(f"Error processing alert: {str(e)}")
    
    async def _analyze_trends_for_alerts(self, trend_result: Dict[str, Any]):
        """Analyze trend data for potential alert conditions"""
        try:
            trends = trend_result.get('trends', {})
            
            # Check volume trends
            volume_trends = trends.get('volume', {})
            if volume_trends.get('trend_direction') == 'increasing':
                avg_volume = volume_trends.get('average_volume', 0)
                if avg_volume > 50:  # Threshold for high volume
                    await self._create_trend_alert(
                        'volume_trend_alert',
                        AlertSeverity.WARNING,
                        'Increasing Incident Volume Trend',
                        f'Incident volume is trending upward (avg: {avg_volume:.1f})',
                        volume_trends
                    )
            
            # Check category trends
            category_trends = trends.get('category', {})
            growing_categories = category_trends.get('growing_categories', [])
            if len(growing_categories) > 3:
                await self._create_trend_alert(
                    'category_growth_alert',
                    AlertSeverity.INFO,
                    'Multiple Growing Categories',
                    f'Categories showing growth: {", ".join(growing_categories[:3])}',
                    category_trends
                )
                
        except Exception as e:
            self.logger.error(f"Error analyzing trends for alerts: {str(e)}")
    
    async def _process_anomalies_for_alerts(self, anomaly_result: Dict[str, Any]):
        """Process anomaly detection results for alert generation"""
        try:
            anomalies = anomaly_result.get('anomalies', {})
            overall_level = anomaly_result.get('overall_anomaly_level', 'normal')
            
            # Generate alerts based on anomaly level
            if overall_level in ['critical', 'warning']:
                severity = AlertSeverity.CRITICAL if overall_level == 'critical' else AlertSeverity.WARNING
                
                await self._create_anomaly_alert(
                    'anomaly_detection_alert',
                    severity,
                    f'Anomaly Detected - {overall_level.title()} Level',
                    f'System anomalies detected at {overall_level} level',
                    anomalies
                )
            
            # Check specific anomaly types
            volume_anomalies = anomalies.get('volume', {})
            if volume_anomalies.get('anomaly_detected'):
                await self._create_anomaly_alert(
                    'volume_anomaly_alert',
                    AlertSeverity.WARNING,
                    'Volume Anomaly Detected',
                    f'Unusual incident volume pattern detected',
                    volume_anomalies
                )
                
        except Exception as e:
            self.logger.error(f"Error processing anomalies for alerts: {str(e)}")
    
    async def _create_trend_alert(self, rule_id: str, severity: AlertSeverity,
                                title: str, description: str, data: Dict[str, Any]):
        """Create alert from trend analysis"""
        alert = MonitoringAlert(
            alert_id=f"trend_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
            rule_id=rule_id,
            severity=severity,
            title=title,
            description=description,
            data=data,
            created_at=datetime.now()
        )
        
        self.active_alerts[alert.alert_id] = alert
        self.logger.info(f"Created trend alert: {alert.alert_id}")
    
    async def _create_anomaly_alert(self, rule_id: str, severity: AlertSeverity,
                                  title: str, description: str, data: Dict[str, Any]):
        """Create alert from anomaly detection"""
        alert = MonitoringAlert(
            alert_id=f"anomaly_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
            rule_id=rule_id,
            severity=severity,
            title=title,
            description=description,
            data=data,
            created_at=datetime.now()
        )
        
        self.active_alerts[alert.alert_id] = alert
        self.logger.info(f"Created anomaly alert: {alert.alert_id}")
    
    def _build_alert_conditions(self) -> Dict[str, Any]:
        """Build alert conditions from monitoring rules"""
        conditions = {}
        
        for rule_id, rule in self.monitoring_rules.items():
            if rule.enabled:
                conditions[rule_id] = {
                    'type': rule.rule_type,
                    'conditions': rule.conditions,
                    'severity': rule.severity.value
                }
        
        return conditions
    
    def _get_severity_thresholds(self) -> Dict[str, float]:
        """Get severity thresholds for alert evaluation"""
        return {
            'info': 0.3,
            'warning': 0.5,
            'critical': 0.7,
            'emergency': 0.9
        }
    
    async def _create_alerts_from_conditions(self, condition_results: Dict[str, Any],
                                           monitoring_data: Dict[str, Any]) -> List[MonitoringAlert]:
        """Create alerts from triggered conditions"""
        alerts = []
        
        for condition_name, result in condition_results.items():
            if result.get('triggered'):
                rule = self.monitoring_rules.get(condition_name)
                if rule:
                    alert = MonitoringAlert(
                        alert_id=f"condition_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
                        rule_id=rule.rule_id,
                        severity=rule.severity,
                        title=f"Alert: {rule.name}",
                        description=rule.description,
                        data={
                            'condition_result': result,
                            'monitoring_data': monitoring_data,
                            'rule_conditions': rule.conditions
                        },
                        created_at=datetime.now()
                    )
                    
                    self.active_alerts[alert.alert_id] = alert
                    alerts.append(alert)
        
        return alerts
    
    async def _manage_alert_lifecycle(self):
        """Manage alert lifecycle and escalation"""
        try:
            current_time = datetime.now()
            
            for alert_id, alert in list(self.active_alerts.items()):
                # Check for escalation
                alert_age = (current_time - alert.created_at).total_seconds() / 60  # minutes
                
                # Escalate critical alerts after 15 minutes
                if (alert.severity in [AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY] and 
                    alert_age > 15 and alert.status != 'escalated'):
                    await self._escalate_alert(alert)
                
                # Auto-resolve info alerts after 2 hours
                if (alert.severity == AlertSeverity.INFO and 
                    alert_age > 120 and alert.status == 'active'):
                    alert.status = 'auto_resolved'
                    self.logger.info(f"Auto-resolved info alert: {alert_id}")
                    
        except Exception as e:
            self.logger.error(f"Error in alert lifecycle management: {str(e)}")
    
    async def _escalate_alert(self, alert: MonitoringAlert):
        """Escalate a critical alert"""
        try:
            # Update alert status
            alert.status = 'escalated'
            
            # Send escalation notification
            escalation_data = {
                'alert': {
                    'alert_id': alert.alert_id,
                    'severity': alert.severity.value,
                    'title': f"ESCALATED: {alert.title}",
                    'description': f"Alert escalated due to no response. Original: {alert.description}",
                    'escalated_at': datetime.now().isoformat()
                }
            }
            
            # Use notification dispatch to send escalation
            alerting_agent = self.registry.get_best_agent_for_capability('notification_dispatch')
            if alerting_agent:
                await self.communicator.send_task_request(
                    recipient_id=alerting_agent.agent_id,
                    task_type='notification_dispatch',
                    task_data={
                        'alerts': [escalation_data['alert']],
                        'notification_channels': ['email', 'teams'],
                        'override_config': {'escalation': True}
                    },
                    timeout_seconds=30
                )
            
            self.logger.warning(f"Escalated alert: {alert.alert_id}")
            
        except Exception as e:
            self.logger.error(f"Error escalating alert {alert.alert_id}: {str(e)}")
    
    async def _cleanup_old_alerts(self):
        """Clean up old alerts from active list"""
        try:
            current_time = datetime.now()
            cleanup_threshold = current_time - timedelta(hours=self.alert_retention_hours)
            
            alerts_to_remove = []
            for alert_id, alert in self.active_alerts.items():
                if alert.created_at < cleanup_threshold:
                    # Move to history
                    self.alert_history.append(alert)
                    alerts_to_remove.append(alert_id)
            
            # Remove from active alerts
            for alert_id in alerts_to_remove:
                del self.active_alerts[alert_id]
                self.logger.debug(f"Cleaned up old alert: {alert_id}")
            
            # Limit history size
            if len(self.alert_history) > 1000:
                self.alert_history = self.alert_history[-500:]  # Keep last 500
                
        except Exception as e:
            self.logger.error(f"Error in alert cleanup: {str(e)}")
    
    def _load_default_monitoring_rules(self):
        """Load default monitoring rules"""
        self.monitoring_rules = {
            'volume_spike': MonitoringRule(
                rule_id='volume_spike',
                name='Volume Spike Detection',
                description='Detect unusual spikes in incident volume',
                rule_type='volume_spike',
                conditions={
                    'threshold_multiplier': 2.0,
                    'baseline_window_hours': 24,
                    'minimum_incidents': 5
                },
                severity=AlertSeverity.WARNING,
                cooldown_minutes=30,
                notification_channels=['email', 'teams']
            ),
            'pattern_anomaly': MonitoringRule(
                rule_id='pattern_anomaly',
                name='Pattern Anomaly Detection',
                description='Detect unusual patterns in incident categories',
                rule_type='pattern_anomaly',
                conditions={
                    'concentration_threshold': 0.8,
                    'minimum_incidents': 10
                },
                severity=AlertSeverity.INFO,
                cooldown_minutes=60,
                notification_channels=['email']
            ),
            'system_health': MonitoringRule(
                rule_id='system_health',
                name='System Health Monitoring',
                description='Monitor overall system health indicators',
                rule_type='threshold',
                conditions={
                    'error_rate_threshold': 0.1,  # 10%
                    'response_time_threshold': 5000  # 5 seconds
                },
                severity=AlertSeverity.CRITICAL,
                cooldown_minutes=15,
                notification_channels=['email', 'teams', 'webhook']
            )
        }
    
    # Management methods
    
    def add_monitoring_rule(self, rule: MonitoringRule):
        """Add a new monitoring rule"""
        self.monitoring_rules[rule.rule_id] = rule
        self.logger.info(f"Added monitoring rule: {rule.rule_id}")
    
    def remove_monitoring_rule(self, rule_id: str):
        """Remove a monitoring rule"""
        if rule_id in self.monitoring_rules:
            del self.monitoring_rules[rule_id]
            self.logger.info(f"Removed monitoring rule: {rule_id}")
    
    def enable_rule(self, rule_id: str):
        """Enable a monitoring rule"""
        if rule_id in self.monitoring_rules:
            self.monitoring_rules[rule_id].enabled = True
            self.logger.info(f"Enabled monitoring rule: {rule_id}")
    
    def disable_rule(self, rule_id: str):
        """Disable a monitoring rule"""
        if rule_id in self.monitoring_rules:
            self.monitoring_rules[rule_id].enabled = False
            self.logger.info(f"Disabled monitoring rule: {rule_id}")
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get list of active alerts"""
        return [
            {
                'alert_id': alert.alert_id,
                'rule_id': alert.rule_id,
                'severity': alert.severity.value,
                'title': alert.title,
                'description': alert.description,
                'created_at': alert.created_at.isoformat(),
                'status': alert.status
            }
            for alert in self.active_alerts.values()
        ]
    
    def get_monitoring_statistics(self) -> Dict[str, Any]:
        """Get monitoring system statistics"""
        return {
            'status': self.status.value,
            'active_alerts': len(self.active_alerts),
            'total_rules': len(self.monitoring_rules),
            'enabled_rules': len([r for r in self.monitoring_rules.values() if r.enabled]),
            'alert_history_count': len(self.alert_history),
            'monitoring_intervals': {
                'real_time': self.monitoring_interval,
                'pattern_analysis': self.pattern_analysis_interval,
                'anomaly_detection': self.anomaly_detection_interval
            },
            'alert_distribution': self._get_alert_distribution(),
            'last_updated': datetime.now().isoformat()
        }
    
    def _get_alert_distribution(self) -> Dict[str, int]:
        """Get distribution of alerts by severity"""
        distribution = {severity.value: 0 for severity in AlertSeverity}
        
        for alert in self.active_alerts.values():
            distribution[alert.severity.value] += 1
        
        return distribution