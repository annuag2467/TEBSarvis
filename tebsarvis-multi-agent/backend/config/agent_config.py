"""
Agent Configuration Management for TEBSarvis Multi-Agent System
Defines agent capabilities, behaviors, and orchestration settings.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum
import logging
from .message_types import Priority 


class AgentType(Enum):
    """Types of agents in the system"""
    RESOLUTION = "resolution"
    SEARCH = "search"
    CONVERSATION = "conversation"
    CONTEXT = "context"
    PATTERN_DETECTION = "pattern_detection"
    ALERTING = "alerting"
    ORCHESTRATOR = "orchestrator"

class TaskPriority(Enum):
    """Task priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4
    EMERGENCY = 5

@dataclass
class AgentCapabilityConfig:
    """Configuration for agent capabilities"""
    name: str
    description: str
    input_types: List[str]
    output_types: List[str]
    dependencies: List[str] = field(default_factory=list)
    timeout_seconds: int = 300
    retry_count: int = 3
    priority: Priority = Priority.NORMAL

@dataclass
class AgentPerformanceConfig:
    """Performance configuration for agents"""
    max_concurrent_tasks: int = 5
    task_timeout: int = 300
    heartbeat_interval: int = 30
    max_memory_usage_mb: int = 512
    max_cpu_usage_percent: int = 80
    cache_size: int = 100
    cache_ttl: int = 300

@dataclass
class AgentCommunicationConfig:
    """Communication configuration for agents"""
    message_timeout: int = 30
    max_retries: int = 3
    retry_delay: int = 5
    max_message_size: int = 1048576  # 1MB
    compression_enabled: bool = True
    encryption_enabled: bool = False

class AgentConfigManager:
    """
    Configuration manager for the multi-agent system.
    Defines agent behaviors, capabilities, and system settings.
    """
    
    def __init__(self):
        self.logger = logging.getLogger("agent.config")
        
        # Initialize agent configurations
        self.agent_configs = self._initialize_agent_configs()
        self.orchestration_config = self._initialize_orchestration_config()
        self.collaboration_config = self._initialize_collaboration_config()
        self.workflow_config = self._initialize_workflow_config()
    
    def _initialize_agent_configs(self) -> Dict[AgentType, Dict[str, Any]]:
        """Initialize configurations for each agent type"""
        configs = {}
        
        # Resolution Agent Configuration
        configs[AgentType.RESOLUTION] = {
            'capabilities': [
                AgentCapabilityConfig(
                    name="incident_resolution",
                    description="Generate ranked solution recommendations for incidents",
                    input_types=["incident_data", "resolution_request"],
                    output_types=["solution_recommendations", "resolution_steps"],
                    dependencies=["azure_openai", "search_agent"],
                    timeout_seconds=300,
                    retry_count=2
                ),
                AgentCapabilityConfig(
                    name="solution_ranking",
                    description="Rank and score solution alternatives",
                    input_types=["solution_candidates", "incident_context"],
                    output_types=["ranked_solutions"],
                    timeout_seconds=120
                ),
                AgentCapabilityConfig(
                    name="resolution_validation",
                    description="Validate and improve solution quality",
                    input_types=["proposed_solution", "incident_data"],
                    output_types=["validation_result", "improved_solution"],
                    timeout_seconds=180
                )
            ],
            'performance': AgentPerformanceConfig(
                max_concurrent_tasks=3,
                task_timeout=300,
                cache_size=50
            ),
            'communication': AgentCommunicationConfig(
                message_timeout=45,
                max_retries=3
            ),
            'config': {
                'max_similar_incidents': 5,
                'confidence_threshold': 0.6,
                'use_rag': True,
                'enable_solution_templates': True
            }
        }
        
        # Search Agent Configuration
        configs[AgentType.SEARCH] = {
            'capabilities': [
                AgentCapabilityConfig(
                    name="semantic_search",
                    description="Perform semantic similarity search across incidents",
                    input_types=["search_query", "text_input"],
                    output_types=["search_results", "similarity_scores"],
                    dependencies=["azure_search", "azure_openai"]
                ),
                AgentCapabilityConfig(
                    name="vector_search",
                    description="Perform vector similarity search using embeddings",
                    input_types=["query_vector", "search_parameters"],
                    output_types=["vector_results"]
                ),
                AgentCapabilityConfig(
                    name="hybrid_search",
                    description="Combine text and vector search for best results",
                    input_types=["search_query", "search_context"],
                    output_types=["ranked_results"]
                )
            ],
            'performance': AgentPerformanceConfig(
                max_concurrent_tasks=10,
                task_timeout=60,
                cache_size=200,
                cache_ttl=300
            ),
            'config': {
                'default_max_results': 10,
                'similarity_threshold': 0.3,
                'enable_caching': True,
                'cache_ttl': 300
            }
        }
        
        # Conversation Agent Configuration
        configs[AgentType.CONVERSATION] = {
            'capabilities': [
                AgentCapabilityConfig(
                    name="natural_language_qa",
                    description="Answer questions in natural language",
                    input_types=["user_question", "conversation_context"],
                    output_types=["natural_response", "follow_up_suggestions"],
                    dependencies=["azure_openai", "search_agent"]
                ),
                AgentCapabilityConfig(
                    name="intent_recognition",
                    description="Recognize user intent and extract entities",
                    input_types=["user_input"],
                    output_types=["intent", "entities"]
                ),
                AgentCapabilityConfig(
                    name="conversation_management",
                    description="Manage multi-turn conversation state",
                    input_types=["conversation_history"],
                    output_types=["conversation_state", "context_summary"]
                )
            ],
            'performance': AgentPerformanceConfig(
                max_concurrent_tasks=8,
                task_timeout=120,
                cache_size=100
            ),
            'config': {
                'session_timeout': 3600,
                'max_context_length': 4000,
                'enable_conversation_history': True,
                'max_history_length': 20
            }
        }
        
        # Context Agent Configuration
        configs[AgentType.CONTEXT] = {
            'capabilities': [
                AgentCapabilityConfig(
                    name="metadata_enrichment",
                    description="Extract and enrich metadata from incident data",
                    input_types=["incident_data", "raw_text"],
                    output_types=["enriched_metadata", "extracted_entities"]
                ),
                AgentCapabilityConfig(
                    name="environmental_analysis",
                    description="Analyze environmental factors affecting incidents",
                    input_types=["incident_context", "system_data"],
                    output_types=["environmental_factors", "context_insights"]
                ),
                AgentCapabilityConfig(
                    name="data_validation",
                    description="Validate and cleanse incident data",
                    input_types=["raw_incident_data"],
                    output_types=["validated_data", "quality_assessment"]
                )
            ],
            'performance': AgentPerformanceConfig(
                max_concurrent_tasks=5,
                task_timeout=180
            ),
            'config': {
                'cache_duration': 300,
                'enable_ai_extraction': True,
                'validation_rules': ['required_fields', 'data_types', 'format_validation']
            }
        }
        
        # Pattern Detection Agent Configuration
        configs[AgentType.PATTERN_DETECTION] = {
            'capabilities': [
                AgentCapabilityConfig(
                    name="incident_clustering",
                    description="Cluster incidents using ML algorithms",
                    input_types=["incident_data_batch", "clustering_parameters"],
                    output_types=["incident_clusters", "cluster_insights"],
                    timeout_seconds=600
                ),
                AgentCapabilityConfig(
                    name="trend_analysis",
                    description="Analyze trends in incident patterns over time",
                    input_types=["time_series_data", "trend_parameters"],
                    output_types=["trend_analysis", "forecasts"],
                    timeout_seconds=480
                ),
                AgentCapabilityConfig(
                    name="anomaly_detection",
                    description="Detect anomalous patterns in incident data",
                    input_types=["incident_metrics", "baseline_data"],
                    output_types=["anomalies", "anomaly_analysis"],
                    timeout_seconds=360
                )
            ],
            'performance': AgentPerformanceConfig(
                max_concurrent_tasks=2,
                task_timeout=600,
                max_memory_usage_mb=1024
            ),
            'config': {
                'min_cluster_size': 3,
                'trend_window_days': 30,
                'analysis_cache_duration': 1800,
                'enable_ml_clustering': True
            }
        }
        
        # Alerting Agent Configuration
        configs[AgentType.ALERTING] = {
            'capabilities': [
                AgentCapabilityConfig(
                    name="real_time_monitoring",
                    description="Monitor incidents in real-time for alert conditions",
                    input_types=["monitoring_data", "alert_rules"],
                    output_types=["alerts", "notifications"],
                    priority=TaskPriority.HIGH
                ),
                AgentCapabilityConfig(
                    name="threshold_monitoring",
                    description="Monitor metrics against defined thresholds",
                    input_types=["metrics_data", "threshold_config"],
                    output_types=["threshold_alerts"]
                ),
                AgentCapabilityConfig(
                    name="notification_dispatch",
                    description="Send notifications through various channels",
                    input_types=["alerts", "notification_config"],
                    output_types=["notifications_sent"],
                    priority=TaskPriority.CRITICAL
                )
            ],
            'performance': AgentPerformanceConfig(
                max_concurrent_tasks=15,
                task_timeout=60,
                heartbeat_interval=15
            ),
            'config': {
                'monitoring_interval': 60,
                'escalation_interval': 300,
                'max_alerts_per_rule': 10,
                'alert_retention_days': 30,
                'notification_channels': ['email', 'teams', 'webhook']
            }
        }
        
        return configs
    
    def _initialize_orchestration_config(self) -> Dict[str, Any]:
        """Initialize orchestration configuration"""
        return {
            'max_concurrent_workflows': 10,
            'default_timeout_minutes': 60,
            'coordination_strategies': {
                'sequential': {'enabled': True, 'default_timeout': 300},
                'parallel': {'enabled': True, 'max_parallel_tasks': 5},
                'conditional': {'enabled': True},
                'priority_based': {'enabled': True}
            },
            'load_balancing': {
                'strategy': 'least_loaded',
                'round_robin_enabled': True,
                'health_check_interval': 30
            },
            'error_handling': {
                'max_retries': 3,
                'backoff_strategy': 'exponential',
                'circuit_breaker_enabled': True,
                'circuit_breaker_threshold': 5
            }
        }
    
    def _initialize_collaboration_config(self) -> Dict[str, Any]:
        """Initialize collaboration configuration"""
        return {
            'max_concurrent_sessions': 20,
            'default_timeout_minutes': 30,
            'context_cleanup_hours': 24,
            'collaboration_types': {
                'consensus_building': {
                    'enabled': True,
                    'required_agreement': 0.7,
                    'timeout_minutes': 20
                },
                'knowledge_synthesis': {
                    'enabled': True,
                    'min_confidence': 0.6,
                    'timeout_minutes': 30
                },
                'expert_consultation': {
                    'enabled': True,
                    'expert_agreement': 0.8,
                    'timeout_minutes': 25
                },
                'peer_review': {
                    'enabled': True,
                    'min_reviewers': 2,
                    'timeout_minutes': 15
                }
            },
            'message_patterns': {
                'incident_resolution_consensus': {
                    'required_agent_types': ['resolution', 'search', 'context'],
                    'workflow_steps': [
                        {'step': 'context_enrichment', 'agent_type': 'context'},
                        {'step': 'similarity_search', 'agent_type': 'search'},
                        {'step': 'resolution_generation', 'agent_type': 'resolution'},
                        {'step': 'consensus_building', 'agent_type': 'all'}
                    ]
                }
            }
        }
    
    def _initialize_workflow_config(self) -> Dict[str, Any]:
        """Initialize workflow configuration"""
        return {
            'max_concurrent_executions': 50,
            'execution_cleanup_hours': 24,
            'default_step_timeout': 300,
            'workflow_templates': {
                'incident_resolution': {
                    'timeout_minutes': 30,
                    'max_retries': 3,
                    'steps': [
                        {
                            'id': 'enrich_context',
                            'agent_type': 'context',
                            'task_type': 'metadata_enrichment',
                            'timeout': 180
                        },
                        {
                            'id': 'search_similar',
                            'agent_type': 'search',
                            'task_type': 'semantic_search',
                            'dependencies': ['enrich_context'],
                            'timeout': 120
                        },
                        {
                            'id': 'generate_resolution',
                            'agent_type': 'resolution',
                            'task_type': 'incident_resolution',
                            'dependencies': ['search_similar'],
                            'timeout': 300
                        },
                        {
                            'id': 'generate_response',
                            'agent_type': 'conversation',
                            'task_type': 'response_generation',
                            'dependencies': ['generate_resolution'],
                            'timeout': 120
                        }
                    ]
                },
                'pattern_analysis': {
                    'timeout_minutes': 45,
                    'max_retries': 2,
                    'steps': [
                        {
                            'id': 'cluster_incidents',
                            'agent_type': 'pattern_detection',
                            'task_type': 'incident_clustering',
                            'timeout': 600
                        },
                        {
                            'id': 'analyze_trends',
                            'agent_type': 'pattern_detection',
                            'task_type': 'trend_analysis',
                            'timeout': 480
                        },
                        {
                            'id': 'detect_anomalies',
                            'agent_type': 'pattern_detection',
                            'task_type': 'anomaly_detection',
                            'timeout': 360
                        }
                    ]
                },
                'proactive_monitoring': {
                    'timeout_minutes': 10,
                    'max_retries': 1,
                    'steps': [
                        {
                            'id': 'real_time_monitoring',
                            'agent_type': 'alerting',
                            'task_type': 'real_time_monitoring',
                            'timeout': 60
                        },
                        {
                            'id': 'evaluate_alerts',
                            'agent_type': 'alerting',
                            'task_type': 'threshold_monitoring',
                            'dependencies': ['real_time_monitoring'],
                            'timeout': 30
                        },
                        {
                            'id': 'dispatch_notifications',
                            'agent_type': 'alerting',
                            'task_type': 'notification_dispatch',
                            'dependencies': ['evaluate_alerts'],
                            'timeout': 30
                        }
                    ]
                }
            }
        }
    
    def get_agent_config(self, agent_type: AgentType) -> Dict[str, Any]:
        """Get configuration for a specific agent type"""
        return self.agent_configs.get(agent_type, {})
    
    def get_agent_capabilities(self, agent_type: AgentType) -> List[AgentCapabilityConfig]:
        """Get capabilities for a specific agent type"""
        config = self.get_agent_config(agent_type)
        return config.get('capabilities', [])
    
    def get_agent_performance_config(self, agent_type: AgentType) -> AgentPerformanceConfig:
        """Get performance configuration for an agent"""
        config = self.get_agent_config(agent_type)
        return config.get('performance', AgentPerformanceConfig())
    
    def get_capability_by_name(self, agent_type: AgentType, capability_name: str) -> Optional[AgentCapabilityConfig]:
        """Get a specific capability configuration"""
        capabilities = self.get_agent_capabilities(agent_type)
        for capability in capabilities:
            if capability.name == capability_name:
                return capability
        return None
    
    def get_workflow_template(self, workflow_name: str) -> Optional[Dict[str, Any]]:
        """Get a workflow template by name"""
        return self.workflow_config['workflow_templates'].get(workflow_name)
    
    def get_collaboration_config(self, collaboration_type: str) -> Optional[Dict[str, Any]]:
        """Get collaboration configuration by type"""
        return self.collaboration_config['collaboration_types'].get(collaboration_type)
    
    def validate_agent_config(self, agent_type: AgentType) -> Dict[str, bool]:
        """Validate agent configuration"""
        config = self.get_agent_config(agent_type)
        
        validation_results = {
            'has_capabilities': bool(config.get('capabilities')),
            'has_performance_config': bool(config.get('performance')),
            'has_communication_config': bool(config.get('communication')),
            'capabilities_valid': True
        }
        
        # Validate each capability
        capabilities = config.get('capabilities', [])
        for capability in capabilities:
            if not capability.name or not capability.input_types or not capability.output_types:
                validation_results['capabilities_valid'] = False
                break
        
        return validation_results
    
    def get_system_limits(self) -> Dict[str, int]:
        """Get system-wide limits and thresholds"""
        return {
            'max_concurrent_workflows': self.orchestration_config['max_concurrent_workflows'],
            'max_collaboration_sessions': self.collaboration_config['max_concurrent_sessions'],
            'max_workflow_executions': self.workflow_config['max_concurrent_executions'],
            'max_agents_per_type': 10,
            'max_message_queue_size': 1000,
            'max_cache_size_mb': 256,
            'max_session_duration_hours': 24
        }
    
    def get_timeout_config(self) -> Dict[str, int]:
        """Get timeout configurations for all components"""
        return {
            'agent_task_timeout': 300,
            'workflow_step_timeout': self.workflow_config['default_step_timeout'],
            'collaboration_timeout': self.collaboration_config['default_timeout_minutes'] * 60,
            'message_timeout': 30,
            'health_check_timeout': 10,
            'orchestration_timeout': self.orchestration_config['default_timeout_minutes'] * 60
        }
    
    def export_config(self) -> Dict[str, Any]:
        """Export complete configuration as dictionary"""
        return {
            'agent_configs': {
                agent_type.value: {
                    'capabilities': [
                        {
                            'name': cap.name,
                            'description': cap.description,
                            'timeout': cap.timeout_seconds,
                            'retry_count': cap.retry_count,
                            'priority': cap.priority.name
                        }
                        for cap in config.get('capabilities', [])
                    ],
                    'performance': {
                        'max_concurrent_tasks': config.get('performance', AgentPerformanceConfig()).max_concurrent_tasks,
                        'task_timeout': config.get('performance', AgentPerformanceConfig()).task_timeout,
                        'cache_size': config.get('performance', AgentPerformanceConfig()).cache_size
                    },
                    'config': config.get('config', {})
                }
                for agent_type, config in self.agent_configs.items()
            },
            'orchestration': self.orchestration_config,
            'collaboration': self.collaboration_config,
            'workflows': self.workflow_config,
            'system_limits': self.get_system_limits(),
            'timeouts': self.get_timeout_config()
        }

# Global configuration instance
_agent_config_instance: Optional[AgentConfigManager] = None

def get_agent_config() -> AgentConfigManager:
    """Get the global agent configuration instance"""
    global _agent_config_instance
    if _agent_config_instance is None:
        _agent_config_instance = AgentConfigManager()
    return _agent_config_instance

def get_agent_capability(agent_type: AgentType, capability_name: str) -> Optional[AgentCapabilityConfig]:
    """Get a specific agent capability configuration"""
    config_manager = get_agent_config()
    return config_manager.get_capability_by_name(agent_type, capability_name)

def get_workflow_template(workflow_name: str) -> Optional[Dict[str, Any]]:
    """Get a workflow template configuration"""
    config_manager = get_agent_config()
    return config_manager.get_workflow_template(workflow_name)