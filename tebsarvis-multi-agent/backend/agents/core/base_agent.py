"""
Base Agent Class for TEBSarvis Multi-Agent System
Abstract base class that defines the common interface and functionality for all agents.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
import asyncio
import logging
import uuid
from enum import Enum
from .message_types import Priority  # Replace MessagePriority
from .agent_communication import MessageBus 

class AgentStatus(Enum):
    IDLE = "idle"
    PROCESSING = "processing"
    ERROR = "error"
    OFFLINE = "offline"


@dataclass
class AgentCapability:
    """Defines what an agent can do"""
    name: str
    description: str
    input_types: List[str]
    output_types: List[str]
    dependencies: List[str] = None

@dataclass
class AgentMetrics:
    """Agent performance metrics"""
    total_tasks_processed: int = 0
    successful_tasks: int = 0
    failed_tasks: int = 0
    average_processing_time: float = 0.0
    last_activity: datetime = None

class BaseAgent(ABC):
    """
    Abstract base class for all TEBSarvis agents.
    Provides common functionality for communication, logging, and task management.
    """
    
    def __init__(self, agent_id: str, agent_type: str, capabilities: List[AgentCapability], 
             message_bus: MessageBus = None, agent_system=None):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.capabilities = capabilities
        self.status = AgentStatus.IDLE
        self.metrics = AgentMetrics()
        self.logger = logging.getLogger(f"agent.{agent_type}.{agent_id}")
        self.message_queue = asyncio.Queue()
        self.active_tasks = {}
        self.collaborating_agents = set()
        
        # ADD: Communication integration
        self.message_bus = message_bus
        self.communicator = None  # Will be set when registering
        self.agent_system = agent_system  # Now properly passed in constructor
        
        self.logger.info(f"Agent {self.agent_id} of type {self.agent_type} initialized")
    
    @abstractmethod
    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a task assigned to this agent.
        Must be implemented by each specific agent.
        
        Args:
            task: Task data containing type, payload, and metadata
            
        Returns:
            Result dictionary with processed data
        """
        pass
    
    @abstractmethod
    def get_capabilities(self) -> List[AgentCapability]:
        """
        Return the capabilities this agent provides.
        Used by the orchestrator for task routing.
        """
        pass
    
    async def start(self):
        """Start the agent and register with system"""
        self.status = AgentStatus.IDLE
        
        # Register with agent system if available
        if self.agent_system:
            success = await self.agent_system.register_agent(self)
            if not success:
                self.logger.error(f"Failed to register agent {self.agent_id} with system")
                return
        
        self.logger.info(f"Agent {self.agent_id} started")
        
        # Start message processing loop
        asyncio.create_task(self._message_processing_loop())

    async def stop(self):
        """Stop the agent gracefully"""
        self.status = AgentStatus.OFFLINE
        
        # Deregister from agent system
        if self.agent_system:
            await self.agent_system.deregister_agent(self.agent_id)
        
        self.logger.info(f"Agent {self.agent_id} stopped")
    
    async def _message_processing_loop(self):
        """Main message processing loop"""
        while self.status != AgentStatus.OFFLINE:
            try:
                # Wait for messages with timeout
                message = await asyncio.wait_for(
                    self.message_queue.get(), 
                    timeout=1.0
                )
                await self._handle_message(message)
            except asyncio.TimeoutError:
                # No message received, continue loop
                continue
            except Exception as e:
                self.logger.error(f"Error in message processing loop: {str(e)}")
                await asyncio.sleep(1)
    
    async def _handle_message(self, message: Dict[str, Any]):
        """Handle incoming messages"""
        try:
            message_type = message.get('type', 'unknown')
            task_id = message.get('task_id', str(uuid.uuid4()))
            
            self.logger.debug(f"Processing message type: {message_type}, task_id: {task_id}")
            
            if message_type == 'task':
                await self._process_task_message(message, task_id)
            elif message_type == 'collaboration':
                await self._handle_collaboration_message(message)
            elif message_type == 'status_check':
                await self._handle_status_check(message)
            else:
                self.logger.warning(f"Unknown message type: {message_type}")
                
        except Exception as e:
            self.logger.error(f"Error handling message: {str(e)}")
            self.metrics.failed_tasks += 1
    
    async def _process_task_message(self, message: Dict[str, Any], task_id: str):
        """Process a task message"""
        start_time = datetime.now()
        self.status = AgentStatus.PROCESSING
        self.active_tasks[task_id] = start_time
        
        try:
            # Extract task data
            task_data = message.get('payload', {})
            task_data['task_id'] = task_id
            task_data['agent_id'] = self.agent_id
            
            # Process the task
            result = await self.process_task(task_data)
            
            # Update metrics
            processing_time = (datetime.now() - start_time).total_seconds()
            self._update_metrics(processing_time, success=True)
            
            # Send result back if callback specified
            if 'callback' in message:
                await self._send_response(message['callback'], result, task_id)
            
            self.logger.info(f"Task {task_id} completed successfully in {processing_time:.2f}s")
            
        except Exception as e:
            self.logger.error(f"Task {task_id} failed: {str(e)}")
            self._update_metrics(0, success=False)
            
            # Send error response if callback specified
            if 'callback' in message:
                error_response = {
                    'error': str(e),
                    'task_id': task_id,
                    'agent_id': self.agent_id
                }
                await self._send_response(message['callback'], error_response, task_id)
        
        finally:
            self.status = AgentStatus.IDLE
            if task_id in self.active_tasks:
                del self.active_tasks[task_id]
    
    async def _handle_collaboration_message(self, message: Dict[str, Any]):
        """Handle collaboration requests from other agents"""
        collaborator_id = message.get('from_agent')
        collaboration_type = message.get('collaboration_type')
        
        if collaborator_id:
            self.collaborating_agents.add(collaborator_id)
            self.logger.debug(f"Collaboration with agent {collaborator_id}: {collaboration_type}")
    
    async def _handle_status_check(self, message: Dict[str, Any]):
        """Handle status check requests"""
        status_info = {
            'agent_id': self.agent_id,
            'agent_type': self.agent_type,
            'status': self.status.value,
            'active_tasks': len(self.active_tasks),
            'metrics': {
                'total_processed': self.metrics.total_tasks_processed,
                'success_rate': self._calculate_success_rate(),
                'avg_processing_time': self.metrics.average_processing_time
            },
            'capabilities': [cap.name for cap in self.capabilities]
        }
        
        if 'callback' in message:
            await self._send_response(message['callback'], status_info, 'status_check')

    
    def _update_metrics(self, processing_time: float, success: bool):
        """Update agent performance metrics"""
        self.metrics.total_tasks_processed += 1
        self.metrics.last_activity = datetime.now()
        
        if success:
            self.metrics.successful_tasks += 1
        else:
            self.metrics.failed_tasks += 1
        
        # Update average processing time
        if processing_time > 0:
            total_time = (self.metrics.average_processing_time * 
                         (self.metrics.total_tasks_processed - 1) + processing_time)
            self.metrics.average_processing_time = total_time / self.metrics.total_tasks_processed
    
    def _calculate_success_rate(self) -> float:
        """Calculate task success rate"""
        if self.metrics.total_tasks_processed == 0:
            return 0.0
        return (self.metrics.successful_tasks / self.metrics.total_tasks_processed) * 100
    
    async def send_message_to_agent(self, target_agent_id: str, message: Dict[str, Any]):
        """Send message to another agent"""
        if not self.communicator:
            self.logger.error("No communicator available - agent not registered with system")
            return False
        
        try:
            if message.get('type') == 'collaboration':
                success = await self.communicator.send_collaboration_request(
                    target_agent_id,
                    message.get('collaboration_type', ''),
                    message.get('data', {})
                )
                return success
            else:
                self.logger.debug(f"Sending message to agent {target_agent_id}")
                return True
        except Exception as e:
            self.logger.error(f"Failed to send message to {target_agent_id}: {str(e)}")
            return False

    async def _send_response(self, callback: str, data: Dict[str, Any], task_id: str):
        """Send response back to requester"""
        if not self.communicator:
            self.logger.error("No communicator available for response")
            return
        
        try:
            recipient_id = callback.split(':')[0] if ':' in callback else callback
            
            await self.communicator.send_response(
                original_message_id=task_id,
                recipient_id=recipient_id,
                success=True,
                result_data=data
            )
        except Exception as e:
            self.logger.error(f"Failed to send response: {str(e)}")
    
    async def request_collaboration(self, target_agent_id: str, collaboration_type: str, data: Dict[str, Any]):
        """Request collaboration with another agent"""
        collaboration_message = {
            'type': 'collaboration',
            'from_agent': self.agent_id,
            'collaboration_type': collaboration_type,
            'data': data
        }
        await self.send_message_to_agent(target_agent_id, collaboration_message)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current agent status"""
        return {
            'agent_id': self.agent_id,
            'agent_type': self.agent_type,
            'status': self.status.value,
            'active_tasks': len(self.active_tasks),
            'collaborating_agents': list(self.collaborating_agents),
            'metrics': {
                'total_processed': self.metrics.total_tasks_processed,
                'successful': self.metrics.successful_tasks,
                'failed': self.metrics.failed_tasks,
                'success_rate': self._calculate_success_rate(),
                'avg_processing_time': self.metrics.average_processing_time,
                'last_activity': self.metrics.last_activity.isoformat() if self.metrics.last_activity else None
            }
        }
    
    def can_handle_task(self, task_type: str) -> bool:
        """Check if agent can handle a specific task type"""
        for capability in self.capabilities:
            if task_type in capability.input_types:
                return True
        return False
    

