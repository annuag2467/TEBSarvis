
"""
Message Types and Schemas for TEBSarvis Multi-Agent Communication
Defines all message formats used for inter-agent communication.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from enum import Enum
import uuid

class MessageType(Enum):
    """Types of messages that can be sent between agents"""
    TASK_REQUEST = "task_request"
    TASK_RESPONSE = "task_response"
    COLLABORATION_REQUEST = "collaboration_request"
    COLLABORATION_RESPONSE = "collaboration_response"
    STATUS_CHECK = "status_check"
    STATUS_RESPONSE = "status_response"
    ERROR = "error"
    HEARTBEAT = "heartbeat"
    SHUTDOWN = "shutdown"

class TaskType(Enum):
    """Types of tasks that agents can process"""
    INCIDENT_RESOLUTION = "incident_resolution"
    SIMILARITY_SEARCH = "similarity_search"
    CONVERSATION = "conversation"
    CONTEXT_ENRICHMENT = "context_enrichment"
    PATTERN_DETECTION = "pattern_detection"
    ALERT_GENERATION = "alert_generation"
    DATA_PROCESSING = "data_processing"

class Priority(Enum):
    """Message priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class BaseMessage:
    """Base message structure for all inter-agent communication"""
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    message_type: MessageType = MessageType.TASK_REQUEST
    sender_id: str = ""
    recipient_id: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    priority: Priority = Priority.NORMAL
    correlation_id: Optional[str] = None  # For tracking related messages
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary for serialization"""
        return {
            'message_id': self.message_id,
            'message_type': self.message_type.value,
            'sender_id': self.sender_id,
            'recipient_id': self.recipient_id,
            'timestamp': self.timestamp.isoformat(),
            'priority': self.priority.value,
            'correlation_id': self.correlation_id
        }

@dataclass
class TaskRequestMessage(BaseMessage):
    """Message for requesting task execution from an agent"""
    message_type: MessageType = MessageType.TASK_REQUEST
    task_type: TaskType = TaskType.INCIDENT_RESOLUTION
    task_data: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    callback_url: Optional[str] = None
    timeout_seconds: int = 300
    requires_collaboration: bool = False
    collaborating_agents: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        base_dict = super().to_dict()
        base_dict.update({
            'task_type': self.task_type.value,
            'task_data': self.task_data,
            'context': self.context,
            'callback_url': self.callback_url,
            'timeout_seconds': self.timeout_seconds,
            'requires_collaboration': self.requires_collaboration,
            'collaborating_agents': self.collaborating_agents
        })
        return base_dict

@dataclass
class TaskResponseMessage(BaseMessage):
    """Message for responding to task requests"""
    message_type: MessageType = MessageType.TASK_RESPONSE
    task_id: str = ""
    success: bool = True
    result_data: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    processing_time_ms: int = 0
    confidence_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        base_dict = super().to_dict()
        base_dict.update({
            'task_id': self.task_id,
            'success': self.success,
            'result_data': self.result_data,
            'error_message': self.error_message,
            'processing_time_ms': self.processing_time_ms,
            'confidence_score': self.confidence_score,
            'metadata': self.metadata
        })
        return base_dict

@dataclass
class CollaborationRequestMessage(BaseMessage):
    """Message for requesting collaboration between agents"""
    message_type: MessageType = MessageType.COLLABORATION_REQUEST
    collaboration_type: str = ""  # e.g., "context_sharing", "joint_processing"
    shared_data: Dict[str, Any] = field(default_factory=dict)
    expected_contribution: str = ""  # What the requesting agent expects
    workflow_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        base_dict = super().to_dict()
        base_dict.update({
            'collaboration_type': self.collaboration_type,
            'shared_data': self.shared_data,
            'expected_contribution': self.expected_contribution,
            'workflow_id': self.workflow_id
        })
        return base_dict

@dataclass
class CollaborationResponseMessage(BaseMessage):
    """Message for responding to collaboration requests"""
    message_type: MessageType = MessageType.COLLABORATION_RESPONSE
    accepted: bool = True
    contribution_data: Dict[str, Any] = field(default_factory=dict)
    availability_status: str = "available"  # "available", "busy", "unavailable"
    estimated_completion_time: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        base_dict = super().to_dict()
        base_dict.update({
            'accepted': self.accepted,
            'contribution_data': self.contribution_data,
            'availability_status': self.availability_status,
            'estimated_completion_time': self.estimated_completion_time.isoformat() if self.estimated_completion_time else None
        })
        return base_dict

@dataclass
class StatusCheckMessage(BaseMessage):
    """Message for checking agent status"""
    message_type: MessageType = MessageType.STATUS_CHECK
    requested_info: List[str] = field(default_factory=lambda: ["health", "metrics", "capabilities"])
    
    def to_dict(self) -> Dict[str, Any]:
        base_dict = super().to_dict()
        base_dict.update({
            'requested_info': self.requested_info
        })
        return base_dict

@dataclass
class StatusResponseMessage(BaseMessage):
    """Message for responding to status checks"""
    message_type: MessageType = MessageType.STATUS_RESPONSE
    agent_status: str = "active"  # "active", "idle", "busy", "error", "offline"
    health_metrics: Dict[str, Any] = field(default_factory=dict)
    capabilities: List[str] = field(default_factory=list)
    current_load: float = 0.0  # 0.0 to 1.0
    uptime_seconds: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        base_dict = super().to_dict()
        base_dict.update({
            'agent_status': self.agent_status,
            'health_metrics': self.health_metrics,
            'capabilities': self.capabilities,
            'current_load': self.current_load,
            'uptime_seconds': self.uptime_seconds
        })
        return base_dict

@dataclass
class ErrorMessage(BaseMessage):
    """Message for reporting errors"""
    message_type: MessageType = MessageType.ERROR
    error_code: str = ""
    error_description: str = ""
    error_details: Dict[str, Any] = field(default_factory=dict)
    is_recoverable: bool = True
    suggested_action: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        base_dict = super().to_dict()
        base_dict.update({
            'error_code': self.error_code,
            'error_description': self.error_description,
            'error_details': self.error_details,
            'is_recoverable': self.is_recoverable,
            'suggested_action': self.suggested_action
        })
        return base_dict

# Specific task data structures

@dataclass
class IncidentData:
    """Data structure for incident resolution tasks"""
    incident_id: str
    summary: str
    description: str
    category: str
    severity: str
    priority: str
    reporter: str
    date_submitted: datetime
    additional_context: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SearchQuery:
    """Data structure for search tasks"""
    query_text: str
    search_type: str = "semantic"  # "semantic", "keyword", "hybrid"
    filters: Dict[str, Any] = field(default_factory=dict)
    max_results: int = 10
    include_metadata: bool = True

@dataclass
class ConversationContext:
    """Data structure for conversation tasks"""
    user_id: str
    session_id: str
    message: str
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    intent: Optional[str] = None
    entities: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PatternAnalysisRequest:
    """Data structure for pattern detection tasks"""
    data_source: str
    analysis_type: str  # "temporal", "categorical", "anomaly"
    time_range: Dict[str, datetime] = field(default_factory=dict)
    filters: Dict[str, Any] = field(default_factory=dict)
    threshold_parameters: Dict[str, float] = field(default_factory=dict)

@dataclass
class AlertConfiguration:
    """Data structure for alert generation tasks"""
    alert_type: str
    conditions: Dict[str, Any]
    severity_level: str
    notification_channels: List[str] = field(default_factory=list)
    escalation_rules: Dict[str, Any] = field(default_factory=dict)

# Message factory functions

def create_task_request(sender_id: str, recipient_id: str, task_type: TaskType, 
                       task_data: Dict[str, Any], **kwargs) -> TaskRequestMessage:
    """Factory function to create task request messages"""
    return TaskRequestMessage(
        sender_id=sender_id,
        recipient_id=recipient_id,
        task_type=task_type,
        task_data=task_data,
        **kwargs
    )

def create_task_response(sender_id: str, recipient_id: str, task_id: str,
                        success: bool, result_data: Dict[str, Any], **kwargs) -> TaskResponseMessage:
    """Factory function to create task response messages"""
    return TaskResponseMessage(
        sender_id=sender_id,
        recipient_id=recipient_id,
        task_id=task_id,
        success=success,
        result_data=result_data,
        **kwargs
    )

def create_collaboration_request(sender_id: str, recipient_id: str, 
                               collaboration_type: str, shared_data: Dict[str, Any],
                               **kwargs) -> CollaborationRequestMessage:
    """Factory function to create collaboration request messages"""
    return CollaborationRequestMessage(
        sender_id=sender_id,
        recipient_id=recipient_id,
        collaboration_type=collaboration_type,
        shared_data=shared_data,
        **kwargs
    )

def create_status_check(sender_id: str, recipient_id: str, 
                       requested_info: List[str] = None) -> StatusCheckMessage:
    """Factory function to create status check messages"""
    if requested_info is None:
        requested_info = ["health", "metrics", "capabilities"]
    
    return StatusCheckMessage(
        sender_id=sender_id,
        recipient_id=recipient_id,
        requested_info=requested_info
    )

def create_error_message(sender_id: str, recipient_id: str, error_code: str,
                        error_description: str, **kwargs) -> ErrorMessage:
    """Factory function to create error messages"""
    return ErrorMessage(
        sender_id=sender_id,
        recipient_id=recipient_id,
        error_code=error_code,
        error_description=error_description,
        **kwargs
    )