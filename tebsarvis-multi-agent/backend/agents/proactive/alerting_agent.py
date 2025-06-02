"""
Alerting Agent for TEBSarvis Multi-Agent System
Monitors patterns and generates early warnings with ML + Rules Engine.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Set, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import json
from collections import defaultdict, Counter

from ..core.base_agent import BaseAgent, AgentCapability
from ..shared.azure_clients import AzureClientManager

class AlertSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AlertStatus(Enum):
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"

class AlertType(Enum):
    VOLUME_SPIKE = "volume_spike"
    PATTERN_ANOMALY = "pattern_anomaly"
    RESOLUTION_DEGRADATION = "resolution_degradation"
    SYSTEM_HEALTH = "system_health"
    PREDICTIVE = "predictive"
    THRESHOLD = "threshold"

@dataclass
class Alert:
    """Alert data structure"""
    alert_id: str
    alert_type: AlertType
    severity: AlertSeverity
    status: AlertStatus
    title: str
    description: str
    created_at: datetime
    updated_at: datetime
    data: Dict[str, Any]
    conditions: Dict[str, Any]
    notifications_sent: List[str]
    escalation_level: int = 0
    suppressed_until: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    resolved_by: Optional[str] = None

@dataclass
class AlertRule:
    """Alert rule configuration"""
    rule_id: str
    name: str
    description: str
    alert_type: AlertType
    conditions: Dict[str, Any]
    severity: AlertSeverity
    enabled: bool
    cooldown_minutes: int
    escalation_rules: List[Dict[str, Any]]
    notification_channels: List[str]
    
class AlertingAgent(BaseAgent):
    """
    Alerting Agent that monitors patterns and generates early warnings.
    Uses ML + Rules Engine for proactive alerting and escalation management.
    """
    
    def __init__(self, agent_id: str = "alerting_agent"):
        capabilities = [
            AgentCapability(
                name="real_time_monitoring",
                description="Monitor incidents in real-time for alert conditions",
                input_types=["monitoring_data", "alert_rules"],
                output_types=["alerts", "notifications"],
                dependencies=["cosmos_db", "pattern_detection_agent"]
            ),
            AgentCapability(
                name="threshold_monitoring",
                description="Monitor metrics against defined thresholds",
                input_types=["metrics_data", "threshold_config"],
                output_types=["threshold_alerts"],
                dependencies=["cosmos_db"]
            ),
            AgentCapability(
                name="predictive_alerting",
                description="Generate predictive alerts based on trends",
                input_types=["trend_data", "predictive_models"],
                output_types=["predictive_alerts"],
                dependencies=["pattern_detection_agent", "azure_ml"]
            ),
            AgentCapability(
                name="alert_management",
                description="Manage alert lifecycle and escalations",
                input_types=["alert_data", "escalation_rules"],
                output_types=["managed_alerts", "escalations"],
                dependencies=[]
            ),
            AgentCapability(
                name="notification_dispatch",
                description="Send notifications through various channels",
                input_types=["alerts", "notification_config"],
                output_types=["notifications_sent"],
                dependencies=["azure_functions"]
            )
        ]
        
        super().__init__(agent_id, "alerting", capabilities)
        
        self.azure_manager = AzureClientManager()
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_rules: Dict[str, AlertRule] = {}
        self.notification_history: List[Dict[str, Any]] = []
        self.suppression_rules: Dict[str, Any] = {}
        
        # Configuration
        self.monitoring_interval = 60  # seconds
        self.escalation_interval = 300  # 5 minutes
        self.max_alerts_per_rule = 10
        self.alert_retention_days = 30
        
        # Load default rules
        self._load_default_alert_rules()
        
        # Initialize agent
        asyncio.create_task(self._initialize())
    
    async def _initialize(self):
        """Initialize the alerting agent"""
        try:
            await self.azure_manager.initialize()
            
            # Start monitoring tasks
            asyncio.create_task(self._real_time_monitor())
            asyncio.create_task(self._escalation_manager())
            asyncio.create_task(self._alert_cleanup())
            
            self.logger.info("Alerting Agent initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize Alerting Agent: {str(e)}")
            raise
    
    def get_capabilities(self) -> List[AgentCapability]:
        """Return agent capabilities"""
        return self.capabilities
    
    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process alerting tasks.
        
        Args:
            task: Task data containing alerting parameters
            
        Returns:
            Alerting results with generated alerts
        """
        task_type = task.get('type', 'real_time_monitoring')
        
        if task_type == 'real_time_monitoring':
            return await self._perform_real_time_monitoring(task)
        elif task_type == 'threshold_monitoring':
            return await self._monitor_thresholds(task)
        elif task_type == 'predictive_alerting':
            return await self._generate_predictive_alerts(task)
        elif task_type == 'alert_management':
            return await self._manage_alerts(task)
        elif task_type == 'notification_dispatch':
            return await self._dispatch_notifications(task)
        elif task_type == 'evaluate_conditions':
            return await self._evaluate_alert_conditions(task)
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    async def _perform_real_time_monitoring(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform real-time monitoring for alert conditions.
        
        Args:
            task: Task containing monitoring parameters
            
        Returns:
            Dictionary with monitoring results and generated alerts
        """
        try:
            monitoring_window = task.get('monitoring_window', {'minutes': 15})
            rule_ids = task.get('rule_ids', list(self.alert_rules.keys()))
            
            generated_alerts = []
            evaluated_rules = 0
            
            # Get recent incident data
            recent_data = await self._get_recent_monitoring_data(monitoring_window)
            
            # Evaluate each rule
            for rule_id in rule_ids:
                if rule_id not in self.alert_rules:
                    continue
                
                rule = self.alert_rules[rule_id]
                if not rule.enabled:
                    continue
                
                # Check if rule is in cooldown
                if self._is_rule_in_cooldown(rule_id):
                    continue
                
                # Evaluate rule conditions
                alert = await self._evaluate_rule(rule, recent_data)
                if alert:
                    generated_alerts.append(alert)
                    self.active_alerts[alert.alert_id] = alert
                    
                    # Send notifications
                    await self._send_alert_notifications(alert)
                
                evaluated_rules += 1
            
            return {
                'generated_alerts': [asdict(alert) for alert in generated_alerts],
                'evaluated_rules': evaluated_rules,
                'active_alerts_count': len(self.active_alerts),
                'monitoring_window': monitoring_window,
                'monitoring_metadata': {
                    'agent_id': self.agent_id,
                    'timestamp': datetime.now().isoformat(),
                    'data_points_analyzed': len(recent_data)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in real-time monitoring: {str(e)}")
            raise
    
    async def _monitor_thresholds(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Monitor metrics against defined thresholds.
        
        Args:
            task: Task containing threshold monitoring parameters
            
        Returns:
            Dictionary with threshold monitoring results
        """
        try:
            metrics_data = task.get('metrics_data', {})
            threshold_config = task.get('threshold_config', {})
            
            threshold_alerts = []
            
            # Check volume thresholds
            if 'volume_thresholds' in threshold_config:
                volume_alerts = await self._check_volume_thresholds(
                    metrics_data, threshold_config['volume_thresholds']
                )
                threshold_alerts.extend(volume_alerts)
            
            # Check resolution time thresholds
            if 'resolution_time_thresholds' in threshold_config:
                resolution_alerts = await self._check_resolution_time_thresholds(
                    metrics_data, threshold_config['resolution_time_thresholds']
                )
                threshold_alerts.extend(resolution_alerts)
            
            # Check error rate thresholds
            if 'error_rate_thresholds' in threshold_config:
                error_alerts = await self._check_error_rate_thresholds(
                    metrics_data, threshold_config['error_rate_thresholds']
                )
                threshold_alerts.extend(error_alerts)
            
            return {
                'threshold_alerts': [asdict(alert) for alert in threshold_alerts],
                'thresholds_checked': len(threshold_config),
                'alerts_generated': len(threshold_alerts),
                'monitoring_metadata': {
                    'agent_id': self.agent_id,
                    'timestamp': datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in threshold monitoring: {str(e)}")
            raise
    
    async def _generate_predictive_alerts(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate predictive alerts based on trends and patterns.
        
        Args:
            task: Task containing predictive alerting parameters
            
        Returns:
            Dictionary with predictive alerts
        """
        try:
            trend_data = task.get('trend_data', {})
            prediction_horizon = task.get('prediction_horizon', {'hours': 24})
            confidence_threshold = task.get('confidence_threshold', 0.7)
            
            predictive_alerts = []
            
            # Analyze volume trends for predictions
            if 'volume_trends' in trend_data:
                volume_predictions = await self._predict_volume_issues(
                    trend_data['volume_trends'], prediction_horizon, confidence_threshold
                )
                predictive_alerts.extend(volume_predictions)
            
            # Analyze pattern trends for predictions
            if 'pattern_trends' in trend_data:
                pattern_predictions = await self._predict_pattern_issues(
                    trend_data['pattern_trends'], prediction_horizon, confidence_threshold
                )
                predictive_alerts.extend(pattern_predictions)
            
            # Analyze resolution time trends
            if 'resolution_trends' in trend_data:
                resolution_predictions = await self._predict_resolution_issues(
                    trend_data['resolution_trends'], prediction_horizon, confidence_threshold
                )
                predictive_alerts.extend(resolution_predictions)
            
            return {
                'predictive_alerts': [asdict(alert) for alert in predictive_alerts],
                'prediction_horizon': prediction_horizon,
                'confidence_threshold': confidence_threshold,
                'trends_analyzed': len(trend_data),
                'prediction_metadata': {
                    'agent_id': self.agent_id,
                    'timestamp': datetime.now().isoformat(),
                    'prediction_method': 'trend_analysis'
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in predictive alerting: {str(e)}")
            raise
    
    async def _manage_alerts(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Manage alert lifecycle and escalations.
        
        Args:
            task: Task containing alert management parameters
            
        Returns:
            Dictionary with alert management results
        """
        try:
            action = task.get('action', 'list')  # list, acknowledge, resolve, suppress
            alert_ids = task.get('alert_ids', [])
            user_id = task.get('user_id', 'system')
            
            results = {
                'action': action,
                'processed_alerts': [],
                'errors': []
            }
            
            if action == 'list':
                results['active_alerts'] = [asdict(alert) for alert in self.active_alerts.values()]
                results['total_active'] = len(self.active_alerts)
            
            elif action == 'acknowledge':
                for alert_id in alert_ids:
                    if alert_id in self.active_alerts:
                        alert = self.active_alerts[alert_id]
                        alert.status = AlertStatus.ACKNOWLEDGED
                        alert.acknowledged_by = user_id
                        alert.updated_at = datetime.now()
                        results['processed_alerts'].append(alert_id)
                        self.logger.info(f"Alert {alert_id} acknowledged by {user_id}")
                    else:
                        results['errors'].append(f"Alert {alert_id} not found")
            
            elif action == 'resolve':
                for alert_id in alert_ids:
                    if alert_id in self.active_alerts:
                        alert = self.active_alerts[alert_id]
                        alert.status = AlertStatus.RESOLVED
                        alert.resolved_by = user_id
                        alert.updated_at = datetime.now()
                        results['processed_alerts'].append(alert_id)
                        self.logger.info(f"Alert {alert_id} resolved by {user_id}")
                    else:
                        results['errors'].append(f"Alert {alert_id} not found")
            
            elif action == 'suppress':
                suppress_duration = task.get('suppress_duration', {'hours': 1})
                suppress_until = datetime.now() + timedelta(**suppress_duration)
                
                for alert_id in alert_ids:
                    if alert_id in self.active_alerts:
                        alert = self.active_alerts[alert_id]
                        alert.status = AlertStatus.SUPPRESSED
                        alert.suppressed_until = suppress_until
                        alert.updated_at = datetime.now()
                        results['processed_alerts'].append(alert_id)
                        self.logger.info(f"Alert {alert_id} suppressed until {suppress_until}")
                    else:
                        results['errors'].append(f"Alert {alert_id} not found")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in alert management: {str(e)}")
            raise
    
    async def _dispatch_notifications(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send notifications through various channels.
        
        Args:
            task: Task containing notification parameters
            
        Returns:
            Dictionary with notification dispatch results
        """
        try:
            alerts = task.get('alerts', [])
            notification_channels = task.get('notification_channels', ['email', 'webhook'])
            override_config = task.get('override_config', {})
            
            dispatch_results = {
                'notifications_sent': [],
                'failed_notifications': [],
                'channels_used': notification_channels
            }
            
            for alert_data in alerts:
                alert_id = alert_data.get('alert_id')
                
                for channel in notification_channels:
                    try:
                        if channel == 'email':
                            result = await self._send_email_notification(alert_data, override_config)
                        elif channel == 'webhook':
                            result = await self._send_webhook_notification(alert_data, override_config)
                        elif channel == 'teams':
                            result = await self._send_teams_notification(alert_data, override_config)
                        elif channel == 'slack':
                            result = await self._send_slack_notification(alert_data, override_config)
                        else:
                            result = {'success': False, 'error': f'Unknown channel: {channel}'}
                        
                        if result.get('success'):
                            dispatch_results['notifications_sent'].append({
                                'alert_id': alert_id,
                                'channel': channel,
                                'timestamp': datetime.now().isoformat()
                            })
                        else:
                            dispatch_results['failed_notifications'].append({
                                'alert_id': alert_id,
                                'channel': channel,
                                'error': result.get('error', 'Unknown error')
                            })
                    
                    except Exception as e:
                        dispatch_results['failed_notifications'].append({
                            'alert_id': alert_id,
                            'channel': channel,
                            'error': str(e)
                        })
            
            return dispatch_results
            
        except Exception as e:
            self.logger.error(f"Error in notification dispatch: {str(e)}")
            raise
    
    async def _evaluate_alert_conditions(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate specific alert conditions.
        
        Args:
            task: Task containing conditions to evaluate
            
        Returns:
            Dictionary with evaluation results
        """
        try:
            conditions = task.get('conditions', {})
            data = task.get('data', {})
            
            evaluation_results = {}
            
            for condition_name, condition_config in conditions.items():
                result = await self._evaluate_single_condition(condition_config, data)
                evaluation_results[condition_name] = result
            
            # Determine overall result
            overall_triggered = any(result.get('triggered', False) for result in evaluation_results.values())
            
            return {
                'overall_triggered': overall_triggered,
                'condition_results': evaluation_results,
                'evaluation_metadata': {
                    'agent_id': self.agent_id,
                    'timestamp': datetime.now().isoformat(),
                    'conditions_evaluated': len(conditions)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error evaluating alert conditions: {str(e)}")
            raise
    
    async def _get_recent_monitoring_data(self, time_window: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get recent incident data for monitoring"""
        try:
            # Calculate time range
            end_time = datetime.now()
            if 'minutes' in time_window:
                start_time = end_time - timedelta(minutes=time_window['minutes'])
            elif 'hours' in time_window:
                start_time = end_time - timedelta(hours=time_window['hours'])
            else:
                start_time = end_time - timedelta(minutes=15)  # Default
            
            # Query recent incidents
            query = """
            SELECT * FROM c 
            WHERE c.date_submitted >= @start_time 
            ORDER BY c.date_submitted DESC
            """
            
            parameters = [
                {"name": "@start_time", "value": start_time.strftime('%d-%m-%Y %H:%M')}
            ]
            
            incidents = await self.azure_manager.query_incidents(query, parameters)
            
            return incidents
            
        except Exception as e:
            self.logger.error(f"Error getting recent monitoring data: {str(e)}")
            return []
    
    async def _evaluate_rule(self, rule: AlertRule, data: List[Dict[str, Any]]) -> Optional[Alert]:
        """Evaluate a single alert rule against data"""
        try:
            conditions = rule.conditions
            
            # Volume spike detection
            if rule.alert_type == AlertType.VOLUME_SPIKE:
                return await self._evaluate_volume_spike_rule(rule, data)
            
            # Pattern anomaly detection
            elif rule.alert_type == AlertType.PATTERN_ANOMALY:
                return await self._evaluate_pattern_anomaly_rule(rule, data)
            
            # Resolution degradation detection
            elif rule.alert_type == AlertType.RESOLUTION_DEGRADATION:
                return await self._evaluate_resolution_degradation_rule(rule, data)
            
            # System health monitoring
            elif rule.alert_type == AlertType.SYSTEM_HEALTH:
                return await self._evaluate_system_health_rule(rule, data)
            
            # Generic threshold rule
            elif rule.alert_type == AlertType.THRESHOLD:
                return await self._evaluate_threshold_rule(rule, data)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error evaluating rule {rule.rule_id}: {str(e)}")
            return None
    
    async def _evaluate_volume_spike_rule(self, rule: AlertRule, data: List[Dict[str, Any]]) -> Optional[Alert]:
        """Evaluate volume spike rule"""
        try:
            conditions = rule.conditions
            threshold = conditions.get('threshold_multiplier', 2.0)
            baseline_window = conditions.get('baseline_window_hours', 24)
            
            current_volume = len(data)
            
            # Get baseline data
            baseline_end = datetime.now() - timedelta(hours=1)
            baseline_start = baseline_end - timedelta(hours=baseline_window)
            
            baseline_query = """
            SELECT COUNT(1) as count FROM c 
            WHERE c.date_submitted >= @start_time 
            AND c.date_submitted <= @end_time
            """
            
            parameters = [
                {"name": "@start_time", "value": baseline_start.strftime('%d-%m-%Y %H:%M')},
                {"name": "@end_time", "value": baseline_end.strftime('%d-%m-%Y %H:%M')}
            ]
            
            baseline_result = await self.azure_manager.query_incidents(baseline_query, parameters)
            baseline_volume = baseline_result[0].get('count', 0) if baseline_result else 0
            baseline_avg = baseline_volume / baseline_window
            
            # Check if current volume exceeds threshold
            if current_volume > baseline_avg * threshold:
                alert_id = f"volume_spike_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
                return Alert(
                    alert_id=alert_id,
                    alert_type=AlertType.VOLUME_SPIKE,
                    severity=rule.severity,
                    status=AlertStatus.ACTIVE,
                    title=f"Volume Spike Detected",
                    description=f"Incident volume ({current_volume}) is {current_volume/baseline_avg:.1f}x higher than baseline ({baseline_avg:.1f})",
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                    data={
                        'current_volume': current_volume,
                        'baseline_average': baseline_avg,
                        'threshold_multiplier': threshold,
                        'spike_factor': current_volume / baseline_avg if baseline_avg > 0 else 0
                    },
                    conditions=conditions,
                    notifications_sent=[]
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error evaluating volume spike rule: {str(e)}")
            return None
    
    async def _evaluate_pattern_anomaly_rule(self, rule: AlertRule, data: List[Dict[str, Any]]) -> Optional[Alert]:
        """Evaluate pattern anomaly rule"""
        try:
            # This would integrate with the Pattern Detection Agent
            # For now, simple category distribution check
            
            if not data:
                return None
            
            conditions = rule.conditions
            anomaly_threshold = conditions.get('anomaly_threshold', 0.3)
            
            # Check category distribution
            categories = Counter([inc.get('category', 'Unknown') for inc in data])
            total_incidents = len(data)
            
            # Look for unusual concentration in one category
            for category, count in categories.items():
                percentage = count / total_incidents
                if percentage > 0.8:  # More than 80% in one category
                    alert_id = f"pattern_anomaly_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    
                    return Alert(
                        alert_id=alert_id,
                        alert_type=AlertType.PATTERN_ANOMALY,
                        severity=rule.severity,
                        status=AlertStatus.ACTIVE,
                        title=f"Pattern Anomaly: {category} Concentration",
                        description=f"{percentage:.1%} of recent incidents are in '{category}' category",
                        created_at=datetime.now(),
                        updated_at=datetime.now(),
                        data={
                            'dominant_category': category,
                            'concentration_percentage': percentage,
                            'total_incidents': total_incidents,
                            'category_distribution': dict(categories)
                        },
                        conditions=conditions,
                        notifications_sent=[]
                    )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error evaluating pattern anomaly rule: {str(e)}")
            return None
    
    async def _evaluate_resolution_degradation_rule(self, rule: AlertRule, data: List[Dict[str, Any]]) -> Optional[Alert]:
        """Evaluate resolution time degradation rule"""
        try:
            conditions = rule.conditions
            degradation_threshold = conditions.get('degradation_threshold', 1.5)
            
            # Calculate current average resolution time
            resolved_incidents = [inc for inc in data if inc.get('resolution_date')]
            
            if not resolved_incidents:
                return None
            
            current_resolution_times = []
            for incident in resolved_incidents:
                try:
                    submitted = datetime.strptime(incident.get('date_submitted', ''), '%d-%m-%Y %H:%M')
                    resolved = datetime.strptime(incident.get('resolution_date', ''), '%d-%m-%Y %H:%M')
                    resolution_time = (resolved - submitted).total_seconds() / 3600  # hours
                    current_resolution_times.append(resolution_time)
                except ValueError:
                    continue
            
            if not current_resolution_times:
                return None
            
            current_avg = sum(current_resolution_times) / len(current_resolution_times)
            
            # Compare with historical average (simplified)
            historical_avg = 4.0  # This would be calculated from historical data
            
            if current_avg > historical_avg * degradation_threshold:
                alert_id = f"resolution_degradation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
                return Alert(
                    alert_id=alert_id,
                    alert_type=AlertType.RESOLUTION_DEGRADATION,
                    severity=rule.severity,
                    status=AlertStatus.ACTIVE,
                    title="Resolution Time Degradation",
                    description=f"Average resolution time ({current_avg:.1f}h) is {current_avg/historical_avg:.1f}x higher than baseline ({historical_avg:.1f}h)",
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                    data={
                        'current_average_hours': current_avg,
                        'historical_average_hours': historical_avg,
                        'degradation_factor': current_avg / historical_avg,
                        'incidents_analyzed': len(current_resolution_times)
                    },
                    conditions=conditions,
                    notifications_sent=[]
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error evaluating resolution degradation rule: {str(e)}")
            return None
    
    def _is_rule_in_cooldown(self, rule_id: str) -> bool:
        """Check if a rule is in cooldown period"""
        # Implementation would track last alert time for each rule
        # For now, simple check
        return False
    
    async def _send_alert_notifications(self, alert: Alert):
        """Send notifications for an alert"""
        try:
            rule = self.alert_rules.get(alert.alert_type.value)
            if not rule:
                return
            
            for channel in rule.notification_channels:
                try:
                    if channel == 'email':
                        await self._send_email_notification(asdict(alert), {})
                    elif channel == 'webhook':
                        await self._send_webhook_notification(asdict(alert), {})
                    
                    alert.notifications_sent.append(f"{channel}_{datetime.now().isoformat()}")
                    
                except Exception as e:
                    self.logger.error(f"Failed to send {channel} notification for alert {alert.alert_id}: {str(e)}")
        
        except Exception as e:
            self.logger.error(f"Error sending alert notifications: {str(e)}")
    
    async def _send_email_notification(self, alert_data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Send email notification (placeholder implementation)"""
        # This would integrate with actual email service
        self.logger.info(f"Email notification sent for alert: {alert_data.get('title')}")
        return {'success': True, 'channel': 'email'}
    
    async def _send_webhook_notification(self, alert_data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Send webhook notification (placeholder implementation)"""
        # This would make HTTP POST to webhook URL
        self.logger.info(f"Webhook notification sent for alert: {alert_data.get('title')}")
        return {'success': True, 'channel': 'webhook'}
    
    async def _send_teams_notification(self, alert_data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Send Microsoft Teams notification (placeholder implementation)"""
        self.logger.info(f"Teams notification sent for alert: {alert_data.get('title')}")
        return {'success': True, 'channel': 'teams'}
    
    async def _send_slack_notification(self, alert_data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Send Slack notification (placeholder implementation)"""
        self.logger.info(f"Slack notification sent for alert: {alert_data.get('title')}")
        return {'success': True, 'channel': 'slack'}
    
    def _load_default_alert_rules(self):
        """Load default alert rules"""
        self.alert_rules = {
            'volume_spike': AlertRule(
                rule_id='volume_spike',
                name='Volume Spike Detection',
                description='Detect unusual spikes in incident volume',
                alert_type=AlertType.VOLUME_SPIKE,
                conditions={'threshold_multiplier': 2.0, 'baseline_window_hours': 24},
                severity=AlertSeverity.HIGH,
                enabled=True,
                cooldown_minutes=30,
                escalation_rules=[
                    {'level': 1, 'delay_minutes': 15, 'channels': ['email']},
                    {'level': 2, 'delay_minutes': 30, 'channels': ['email', 'teams']}
                ],
                notification_channels=['email', 'webhook']
            ),
            'pattern_anomaly': AlertRule(
                rule_id='pattern_anomaly',
                name='Pattern Anomaly Detection',
                description='Detect unusual patterns in incident categories',
                alert_type=AlertType.PATTERN_ANOMALY,
                conditions={'anomaly_threshold': 0.3, 'concentration_threshold': 0.8},
                severity=AlertSeverity.MEDIUM,
                enabled=True,
                cooldown_minutes=60,
                escalation_rules=[
                    {'level': 1, 'delay_minutes': 30, 'channels': ['email']}
                ],
                notification_channels=['email']
            ),
            'resolution_degradation': AlertRule(
                rule_id='resolution_degradation',
                name='Resolution Time Degradation',
                description='Detect degradation in resolution times',
                alert_type=AlertType.RESOLUTION_DEGRADATION,
                conditions={'degradation_threshold': 1.5, 'minimum_incidents': 5},
                severity=AlertSeverity.HIGH,
                enabled=True,
                cooldown_minutes=45,
                escalation_rules=[
                    {'level': 1, 'delay_minutes': 20, 'channels': ['email', 'teams']}
                ],
                notification_channels=['email', 'teams']
            )
        }
    
    async def _real_time_monitor(self):
        """Background task for real-time monitoring"""
        while True:
            try:
                # Perform monitoring check
                monitoring_task = {
                    'type': 'real_time_monitoring',
                    'monitoring_window': {'minutes': self.monitoring_interval // 60}
                }
                
                await self._perform_real_time_monitoring(monitoring_task)
                
                # Wait for next monitoring cycle
                await asyncio.sleep(self.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"Error in real-time monitor: {str(e)}")
                await asyncio.sleep(30)  # Shorter retry interval
    
    async def _escalation_manager(self):
        """Background task for managing alert escalations"""
        while True:
            try:
                current_time = datetime.now()
                
                for alert in self.active_alerts.values():
                    if alert.status != AlertStatus.ACTIVE:
                        continue
                    
                    # Check if alert needs escalation
                    rule = self.alert_rules.get(alert.alert_type.value)
                    if not rule or not rule.escalation_rules:
                        continue
                    
                    # Calculate time since alert creation
                    time_since_creation = current_time - alert.created_at
                    
                    # Check escalation levels
                    for escalation in rule.escalation_rules:
                        level = escalation['level']
                        delay_minutes = escalation['delay_minutes']
                        
                        # Skip if already escalated to this level or higher
                        if alert.escalation_level >= level:
                            continue
                        
                        # Check if enough time has passed
                        if time_since_creation.total_seconds() >= delay_minutes * 60:
                            await self._escalate_alert(alert, escalation)
                            alert.escalation_level = level
                            alert.updated_at = current_time
                
                await asyncio.sleep(self.escalation_interval)
                
            except Exception as e:
                self.logger.error(f"Error in escalation manager: {str(e)}")
                await asyncio.sleep(60)
    
    async def _escalate_alert(self, alert: Alert, escalation: Dict[str, Any]):
        """Escalate an alert to the next level"""
        try:
            level = escalation['level']
            channels = escalation['channels']
            
            self.logger.warning(f"Escalating alert {alert.alert_id} to level {level}")
            
            # Send escalation notifications
            escalation_data = asdict(alert)
            escalation_data['escalation_level'] = level
            escalation_data['escalation_reason'] = f"No acknowledgment after {escalation['delay_minutes']} minutes"
            
            dispatch_task = {
                'alerts': [escalation_data],
                'notification_channels': channels,
                'override_config': {'escalation': True, 'level': level}
            }
            
            await self._dispatch_notifications(dispatch_task)
            
        except Exception as e:
            self.logger.error(f"Error escalating alert {alert.alert_id}: {str(e)}")
    
    async def _alert_cleanup(self):
        """Background task for cleaning up old alerts"""
        while True:
            try:
                current_time = datetime.now()
                cleanup_threshold = current_time - timedelta(days=self.alert_retention_days)
                
                # Find alerts to clean up
                alerts_to_remove = []
                for alert_id, alert in self.active_alerts.items():
                    # Remove resolved alerts older than retention period
                    if (alert.status == AlertStatus.RESOLVED and 
                        alert.updated_at < cleanup_threshold):
                        alerts_to_remove.append(alert_id)
                    
                    # Remove suppressed alerts that are no longer suppressed
                    elif (alert.status == AlertStatus.SUPPRESSED and 
                          alert.suppressed_until and 
                          current_time > alert.suppressed_until):
                        alert.status = AlertStatus.ACTIVE
                        alert.suppressed_until = None
                        alert.updated_at = current_time
                
                # Clean up old alerts
                for alert_id in alerts_to_remove:
                    del self.active_alerts[alert_id]
                    self.logger.debug(f"Cleaned up old alert: {alert_id}")
                
                # Run cleanup every hour
                await asyncio.sleep(3600)
                
            except Exception as e:
                self.logger.error(f"Error in alert cleanup: {str(e)}")
                await asyncio.sleep(300)  # Retry in 5 minutes
    
    async def _check_volume_thresholds(self, metrics_data: Dict[str, Any], 
                                     thresholds: Dict[str, Any]) -> List[Alert]:
        """Check volume-based thresholds"""
        alerts = []
        
        try:
            current_volume = metrics_data.get('incident_count', 0)
            
            for threshold_name, threshold_config in thresholds.items():
                threshold_value = threshold_config.get('value', 100)
                severity = AlertSeverity(threshold_config.get('severity', 'medium'))
                
                if current_volume > threshold_value:
                    alert_id = f"volume_threshold_{threshold_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    
                    alert = Alert(
                        alert_id=alert_id,
                        alert_type=AlertType.THRESHOLD,
                        severity=severity,
                        status=AlertStatus.ACTIVE,
                        title=f"Volume Threshold Exceeded: {threshold_name}",
                        description=f"Current incident volume ({current_volume}) exceeds threshold ({threshold_value})",
                        created_at=datetime.now(),
                        updated_at=datetime.now(),
                        data={
                            'current_volume': current_volume,
                            'threshold_value': threshold_value,
                            'threshold_name': threshold_name
                        },
                        conditions=threshold_config,
                        notifications_sent=[]
                    )
                    
                    alerts.append(alert)
        
        except Exception as e:
            self.logger.error(f"Error checking volume thresholds: {str(e)}")
        
        return alerts
    
    async def _check_resolution_time_thresholds(self, metrics_data: Dict[str, Any], 
                                              thresholds: Dict[str, Any]) -> List[Alert]:
        """Check resolution time thresholds"""
        alerts = []
        
        try:
            avg_resolution_time = metrics_data.get('average_resolution_time_hours', 0)
            
            for threshold_name, threshold_config in thresholds.items():
                threshold_value = threshold_config.get('value', 24)  # hours
                severity = AlertSeverity(threshold_config.get('severity', 'medium'))
                
                if avg_resolution_time > threshold_value:
                    alert_id = f"resolution_threshold_{threshold_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    
                    alert = Alert(
                        alert_id=alert_id,
                        alert_type=AlertType.THRESHOLD,
                        severity=severity,
                        status=AlertStatus.ACTIVE,
                        title=f"Resolution Time Threshold Exceeded: {threshold_name}",
                        description=f"Average resolution time ({avg_resolution_time:.1f}h) exceeds threshold ({threshold_value}h)",
                        created_at=datetime.now(),
                        updated_at=datetime.now(),
                        data={
                            'current_avg_hours': avg_resolution_time,
                            'threshold_hours': threshold_value,
                            'threshold_name': threshold_name
                        },
                        conditions=threshold_config,
                        notifications_sent=[]
                    )
                    
                    alerts.append(alert)
        
        except Exception as e:
            self.logger.error(f"Error checking resolution time thresholds: {str(e)}")
        
        return alerts
    
    async def _check_error_rate_thresholds(self, metrics_data: Dict[str, Any], 
                                         thresholds: Dict[str, Any]) -> List[Alert]:
        """Check error rate thresholds"""
        alerts = []
        
        try:
            error_rate = metrics_data.get('error_rate_percentage', 0)
            
            for threshold_name, threshold_config in thresholds.items():
                threshold_value = threshold_config.get('value', 10)  # percentage
                severity = AlertSeverity(threshold_config.get('severity', 'medium'))
                
                if error_rate > threshold_value:
                    alert_id = f"error_rate_threshold_{threshold_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    
                    alert = Alert(
                        alert_id=alert_id,
                        alert_type=AlertType.THRESHOLD,
                        severity=severity,
                        status=AlertStatus.ACTIVE,
                        title=f"Error Rate Threshold Exceeded: {threshold_name}",
                        description=f"Current error rate ({error_rate:.1f}%) exceeds threshold ({threshold_value}%)",
                        created_at=datetime.now(),
                        updated_at=datetime.now(),
                        data={
                            'current_error_rate': error_rate,
                            'threshold_percentage': threshold_value,
                            'threshold_name': threshold_name
                        },
                        conditions=threshold_config,
                        notifications_sent=[]
                    )
                    
                    alerts.append(alert)
        
        except Exception as e:
            self.logger.error(f"Error checking error rate thresholds: {str(e)}")
        
        return alerts
    
    async def _predict_volume_issues(self, volume_trends: Dict[str, Any], 
                                   prediction_horizon: Dict[str, Any], 
                                   confidence_threshold: float) -> List[Alert]:
        """Predict volume-related issues"""
        alerts = []
        
        try:
            trend_direction = volume_trends.get('trend_direction', 'stable')
            current_avg = volume_trends.get('average_volume', 0)
            volatility = volume_trends.get('volatility', 0)
            
            if trend_direction == 'increasing':
                # Predict if volume will exceed capacity
                growth_rate = volume_trends.get('growth_rate', 0.1)  # Would be calculated from trend data
                
                # Simple linear prediction
                horizon_hours = prediction_horizon.get('hours', 24)
                predicted_volume = current_avg * (1 + growth_rate * (horizon_hours / 24))
                
                # Assume capacity limit of 200 incidents per day
                capacity_limit = 200
                
                if predicted_volume > capacity_limit and growth_rate > 0.2:  # 20% growth rate threshold
                    alert_id = f"predictive_volume_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    
                    alert = Alert(
                        alert_id=alert_id,
                        alert_type=AlertType.PREDICTIVE,
                        severity=AlertSeverity.MEDIUM,
                        status=AlertStatus.ACTIVE,
                        title="Predicted Volume Capacity Issue",
                        description=f"Volume predicted to reach {predicted_volume:.0f} in {horizon_hours}h, approaching capacity limit ({capacity_limit})",
                        created_at=datetime.now(),
                        updated_at=datetime.now(),
                        data={
                            'predicted_volume': predicted_volume,
                            'capacity_limit': capacity_limit,
                            'growth_rate': growth_rate,
                            'prediction_horizon_hours': horizon_hours,
                            'confidence': min(1.0 - (volatility / current_avg), 1.0) if current_avg > 0 else 0.5
                        },
                        conditions={'prediction_type': 'volume_capacity'},
                        notifications_sent=[]
                    )
                    
                    alerts.append(alert)
        
        except Exception as e:
            self.logger.error(f"Error predicting volume issues: {str(e)}")
        
        return alerts
    
    async def _predict_pattern_issues(self, pattern_trends: Dict[str, Any], 
                                    prediction_horizon: Dict[str, Any], 
                                    confidence_threshold: float) -> List[Alert]:
        """Predict pattern-related issues"""
        alerts = []
        
        try:
            growing_categories = pattern_trends.get('growing_categories', [])
            
            for category in growing_categories:
                category_data = pattern_trends.get('category_trends', {}).get(category, {})
                trend_direction = category_data.get('trend_direction', 'stable')
                
                if trend_direction == 'increasing':
                    current_percentage = category_data.get('percentage_of_total', 0)
                    
                    # Predict if category will dominate (>70% of incidents)
                    if current_percentage > 50:  # Already concerning
                        alert_id = f"predictive_pattern_{category}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                        
                        alert = Alert(
                            alert_id=alert_id,
                            alert_type=AlertType.PREDICTIVE,
                            severity=AlertSeverity.MEDIUM,
                            status=AlertStatus.ACTIVE,
                            title=f"Predicted Pattern Dominance: {category}",
                            description=f"Category '{category}' ({current_percentage:.1f}%) predicted to dominate incident patterns",
                            created_at=datetime.now(),
                            updated_at=datetime.now(),
                            data={
                                'category': category,
                                'current_percentage': current_percentage,
                                'trend_direction': trend_direction,
                                'prediction_type': 'pattern_dominance'
                            },
                            conditions={'prediction_type': 'pattern_dominance'},
                            notifications_sent=[]
                        )
                        
                        alerts.append(alert)
        
        except Exception as e:
            self.logger.error(f"Error predicting pattern issues: {str(e)}")
        
        return alerts
    
    async def _predict_resolution_issues(self, resolution_trends: Dict[str, Any], 
                                       prediction_horizon: Dict[str, Any], 
                                       confidence_threshold: float) -> List[Alert]:
        """Predict resolution time issues"""
        alerts = []
        
        try:
            trend_direction = resolution_trends.get('trend_direction', 'stable')
            current_avg = resolution_trends.get('overall_average_hours', 0)
            
            if trend_direction == 'increasing':
                # Predict if resolution times will exceed SLA
                sla_threshold = 8.0  # 8 hours SLA
                
                if current_avg > sla_threshold * 0.8:  # 80% of SLA threshold
                    alert_id = f"predictive_resolution_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    
                    alert = Alert(
                        alert_id=alert_id,
                        alert_type=AlertType.PREDICTIVE,
                        severity=AlertSeverity.HIGH,
                        status=AlertStatus.ACTIVE,
                        title="Predicted SLA Breach Risk",
                        description=f"Resolution times ({current_avg:.1f}h) trending upward, risk of SLA breach ({sla_threshold}h)",
                        created_at=datetime.now(),
                        updated_at=datetime.now(),
                        data={
                            'current_avg_hours': current_avg,
                            'sla_threshold_hours': sla_threshold,
                            'trend_direction': trend_direction,
                            'sla_risk_percentage': (current_avg / sla_threshold) * 100
                        },
                        conditions={'prediction_type': 'sla_breach_risk'},
                        notifications_sent=[]
                    )
                    
                    alerts.append(alert)
        
        except Exception as e:
            self.logger.error(f"Error predicting resolution issues: {str(e)}")
        
        return alerts
    
    async def _evaluate_single_condition(self, condition_config: Dict[str, Any], 
                                       data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a single condition"""
        try:
            condition_type = condition_config.get('type', 'threshold')
            
            if condition_type == 'threshold':
                metric = condition_config.get('metric')
                operator = condition_config.get('operator', '>')
                threshold = condition_config.get('threshold')
                
                value = data.get(metric, 0)
                
                if operator == '>':
                    triggered = value > threshold
                elif operator == '<':
                    triggered = value < threshold
                elif operator == '>=':
                    triggered = value >= threshold
                elif operator == '<=':
                    triggered = value <= threshold
                elif operator == '==':
                    triggered = value == threshold
                else:
                    triggered = False
                
                return {
                    'triggered': triggered,
                    'metric': metric,
                    'value': value,
                    'threshold': threshold,
                    'operator': operator
                }
            
            elif condition_type == 'percentage_change':
                metric = condition_config.get('metric')
                baseline = condition_config.get('baseline', 0)
                change_threshold = condition_config.get('change_threshold', 0.2)  # 20%
                
                current_value = data.get(metric, 0)
                
                if baseline > 0:
                    change_percentage = abs(current_value - baseline) / baseline
                    triggered = change_percentage > change_threshold
                else:
                    triggered = False
                
                return {
                    'triggered': triggered,
                    'metric': metric,
                    'current_value': current_value,
                    'baseline': baseline,
                    'change_percentage': change_percentage if baseline > 0 else 0,
                    'change_threshold': change_threshold
                }
            
            else:
                return {'triggered': False, 'error': f'Unknown condition type: {condition_type}'}
        
        except Exception as e:
            return {'triggered': False, 'error': str(e)}
    
    async def _evaluate_system_health_rule(self, rule: AlertRule, data: List[Dict[str, Any]]) -> Optional[Alert]:
        """Evaluate system health rule (placeholder implementation)"""
        # This would integrate with system monitoring
        return None
    
    async def _evaluate_threshold_rule(self, rule: AlertRule, data: List[Dict[str, Any]]) -> Optional[Alert]:
        """Evaluate generic threshold rule"""
        # This would evaluate generic threshold conditions
        return None
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alerting statistics"""
        try:
            active_alerts = [a for a in self.active_alerts.values() if a.status == AlertStatus.ACTIVE]
            acknowledged_alerts = [a for a in self.active_alerts.values() if a.status == AlertStatus.ACKNOWLEDGED]
            resolved_alerts = [a for a in self.active_alerts.values() if a.status == AlertStatus.RESOLVED]
            
            # Severity distribution
            severity_dist = Counter([alert.severity.value for alert in active_alerts])
            
            # Type distribution
            type_dist = Counter([alert.alert_type.value for alert in active_alerts])
            
            return {
                'total_alerts': len(self.active_alerts),
                'active_alerts': len(active_alerts),
                'acknowledged_alerts': len(acknowledged_alerts),
                'resolved_alerts': len(resolved_alerts),
                'severity_distribution': dict(severity_dist),
                'type_distribution': dict(type_dist),
                'enabled_rules': len([r for r in self.alert_rules.values() if r.enabled]),
                'total_rules': len(self.alert_rules),
                'notifications_sent_today': len([
                    n for n in self.notification_history 
                    if n.get('timestamp', '').startswith(datetime.now().strftime('%Y-%m-%d'))
                ])
            }
        
        except Exception as e:
            self.logger.error(f"Error getting alert statistics: {str(e)}")
            return {}
    
    def export_alerts(self, status_filter: Optional[AlertStatus] = None) -> List[Dict[str, Any]]:
        """Export alerts for reporting"""
        try:
            alerts = list(self.active_alerts.values())
            
            if status_filter:
                alerts = [a for a in alerts if a.status == status_filter]
            
            return [asdict(alert) for alert in alerts]
        
        except Exception as e:
            self.logger.error(f"Error exporting alerts: {str(e)}")
            return []