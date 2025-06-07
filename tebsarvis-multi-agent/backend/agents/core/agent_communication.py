"""
Agent Communication Protocol for TEBSarvis Multi-Agent System
Handles message routing, delivery, and communication between agents.
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from collections import defaultdict, deque
import weakref
from dataclasses import asdict
from .base_agent import AgentStatus

from .message_types import (
    BaseMessage, MessageType, TaskRequestMessage, TaskResponseMessage,
    CollaborationRequestMessage, StatusCheckMessage, ErrorMessage, Priority
)

class MessageBus:
    """
    Central message bus for agent communication.
    Handles message routing, queuing, and delivery.
    """
    
    def __init__(self):
        self.agents = {}  # agent_id -> agent_reference
        self.message_queues = defaultdict(asyncio.Queue)  # agent_id -> queue
        self.message_history = defaultdict(lambda: deque(maxlen=1000))  # agent_id -> message history
        self.routing_table = {}  # message_type -> list of capable agent_ids
        self.logger = logging.getLogger("message_bus")
        self.running = False
        self.message_processors = {}  # agent_id -> processor task
        
        # Statistics
        self.stats = {
            'messages_sent': 0,
            'messages_delivered': 0,
            'messages_failed': 0,
            'start_time': datetime.now()
        }
    
    async def start(self):
        """Start the message bus"""
        self.running = True
        self.stats['start_time'] = datetime.now()
        self.logger.info("Message bus started")
    
    async def stop(self):
        """Stop the message bus gracefully"""
        self.running = False
        
        # Cancel all message processors
        for task in self.message_processors.values():
            task.cancel()
        
        # Wait for processors to finish
        if self.message_processors:
            await asyncio.gather(*self.message_processors.values(), return_exceptions=True)
        
        self.logger.info("Message bus stopped")
    
    def register_agent(self, agent_id: str, agent_ref, capabilities: List[str]):
        """
        Register an agent with the message bus.
        
        Args:
            agent_id: Unique identifier for the agent
            agent_ref: Reference to the agent object
            capabilities: List of message types the agent can handle
        """
        self.agents[agent_id] = weakref.ref(agent_ref)
        
        # Update routing table
        for capability in capabilities:
            if capability not in self.routing_table:
                self.routing_table[capability] = []
            if agent_id not in self.routing_table[capability]:
                self.routing_table[capability].append(agent_id)
        
        # Start message processor for this agent
        self.message_processors[agent_id] = asyncio.create_task(
            self._process_agent_messages(agent_id)
        )
        
        self.logger.info(f"Agent {agent_id} registered with capabilities: {capabilities}")
    
    def unregister_agent(self, agent_id: str):
        """Unregister an agent from the message bus"""
        if agent_id in self.agents:
            del self.agents[agent_id]
        
        # Remove from routing table
        for capability_agents in self.routing_table.values():
            if agent_id in capability_agents:
                capability_agents.remove(agent_id)
        
        # Cancel message processor
        if agent_id in self.message_processors:
            self.message_processors[agent_id].cancel()
            del self.message_processors[agent_id]
        
        self.logger.info(f"Agent {agent_id} unregistered")
    
    async def send_message(self, message: BaseMessage) -> bool:
        """
        Send a message to an agent.
        
        Args:
            message: Message to send
            
        Returns:
            True if message was queued successfully, False otherwise
        """
        try:
            recipient_id = message.recipient_id
            
            # Validate recipient exists
            if recipient_id not in self.agents:
                self.logger.error(f"Recipient {recipient_id} not registered")
                self.stats['messages_failed'] += 1
                return False
            
            # Add to message queue
            await self.message_queues[recipient_id].put(message)
            
            # Store in message history
            self.message_history[recipient_id].append({
                'timestamp': message.timestamp,
                'sender': message.sender_id,
                'message_type': message.message_type.value,
                'message_id': message.message_id
            })
            
            self.stats['messages_sent'] += 1
            self.logger.debug(f"Message {message.message_id} queued for {recipient_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send message: {str(e)}")
            self.stats['messages_failed'] += 1
            return False
    
    async def broadcast_message(self, message: BaseMessage, agent_ids: List[str]) -> Dict[str, bool]:
        """
        Broadcast a message to multiple agents.
        
        Args:
            message: Message to broadcast
            agent_ids: List of agent IDs to send to
            
        Returns:
            Dictionary mapping agent_id to success status
        """
        results = {}
        
        for agent_id in agent_ids:
            # Create a copy of the message for each recipient
            message_copy = type(message)(**asdict(message))
            message_copy.recipient_id = agent_id
            
            results[agent_id] = await self.send_message(message_copy)
        
        return results
    
    async def _process_agent_messages(self, agent_id: str):
        """Process messages for a specific agent"""
        while self.running:
            try:
                # Get agent reference
                agent_ref = self.agents.get(agent_id)
                if not agent_ref:
                    break
                
                agent = agent_ref()
                if not agent:
                    # Agent has been garbage collected
                    self.unregister_agent(agent_id)
                    break
                
                # Wait for messages
                try:
                    message = await asyncio.wait_for(
                        self.message_queues[agent_id].get(),
                        timeout=1.0
                    )
                    
                    # Deliver message to agent
                    await self._deliver_message(agent, message)
                    self.stats['messages_delivered'] += 1
                    
                except asyncio.TimeoutError:
                    # No message received, continue
                    continue
                    
            except Exception as e:
                self.logger.error(f"Error processing messages for agent {agent_id}: {str(e)}")
                await asyncio.sleep(1)
    
    async def _deliver_message(self, agent, message: BaseMessage):
        """Deliver a message to an agent"""
        try:
            # Convert message to dict for agent processing
            message_dict = message.to_dict()
            message_dict['payload'] = getattr(message, 'task_data', {})
            message_dict['type'] = 'task' if message.message_type == MessageType.TASK_REQUEST else message.message_type.value
            
            # Add message to agent's queue
            await agent.message_queue.put(message_dict)
            
            self.logger.debug(f"Message {message.message_id} delivered to agent {agent.agent_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to deliver message {message.message_id}: {str(e)}")
            raise
    
    def find_capable_agents(self, capability: str) -> List[str]:
        """Find agents capable of handling a specific task type"""
        return self.routing_table.get(capability, [])
    
    def get_agent_load(self, agent_id: str) -> float:
        """Get current load for an agent (0.0 to 1.0)"""
        if agent_id not in self.agents:
            return 1.0  # Agent not available
        
        queue_size = self.message_queues[agent_id].qsize()
        # Simple load calculation based on queue size
        return min(queue_size / 10.0, 1.0)
    
    def get_least_loaded_agent(self, capability: str) -> Optional[str]:
        """Find the least loaded agent capable of handling a task"""
        capable_agents = self.find_capable_agents(capability)
        if not capable_agents:
            return None
        
        # Find agent with lowest load
        min_load = float('inf')
        best_agent = None
        
        for agent_id in capable_agents:
            load = self.get_agent_load(agent_id)
            if load < min_load:
                min_load = load
                best_agent = agent_id
        
        return best_agent
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get message bus statistics"""
        uptime = datetime.now() - self.stats['start_time']
        return {
            'uptime_seconds': uptime.total_seconds(),
            'messages_sent': self.stats['messages_sent'],
            'messages_delivered': self.stats['messages_delivered'],
            'messages_failed': self.stats['messages_failed'],
            'registered_agents': len(self.agents),
            'total_queue_size': sum(q.qsize() for q in self.message_queues.values()),
            'routing_table_size': len(self.routing_table)
        }

class AgentCommunicator:
    """
    Communication interface for agents to interact with the message bus.
    """
    
    def __init__(self, agent_id: str, message_bus: MessageBus):
        self.agent_id = agent_id
        self.message_bus = message_bus
        self.logger = logging.getLogger(f"communicator.{agent_id}")
        self.pending_responses = {}  # correlation_id -> future
        self.response_timeout = 30  # seconds
    
    async def send_task_request(self, recipient_id: str, task_type: str, 
                               task_data: Dict[str, Any], **kwargs) -> TaskResponseMessage:
        """
        Send a task request and wait for response.
        
        Args:
            recipient_id: ID of the agent to send to
            task_type: Type of task to perform
            task_data: Task data payload
            **kwargs: Additional message parameters
            
        Returns:
            Task response message
        """
        from .message_types import TaskType, create_task_request
        
        # Create task request message
        message = create_task_request(
            sender_id=self.agent_id,
            recipient_id=recipient_id,
            task_type=TaskType(task_type),
            task_data=task_data,
            **kwargs
        )
        
        # Set up response waiting
        future = asyncio.Future()
        self.pending_responses[message.correlation_id or message.message_id] = future
        
        # Send message
        success = await self.message_bus.send_message(message)
        if not success:
            del self.pending_responses[message.correlation_id or message.message_id]
            raise Exception(f"Failed to send message to {recipient_id}")
        
        try:
            # Wait for response with timeout
            response = await asyncio.wait_for(future, timeout=self.response_timeout)
            return response
        except asyncio.TimeoutError:
            # Clean up pending response
            if message.correlation_id or message.message_id in self.pending_responses:
                del self.pending_responses[message.correlation_id or message.message_id]
            raise Exception(f"Timeout waiting for response from {recipient_id}")
    
    async def send_collaboration_request(self, recipient_id: str, collaboration_type: str,
                                       shared_data: Dict[str, Any]) -> bool:
        """Send a collaboration request to another agent"""
        from .message_types import create_collaboration_request
        
        message = create_collaboration_request(
            sender_id=self.agent_id,
            recipient_id=recipient_id,
            collaboration_type=collaboration_type,
            shared_data=shared_data
        )
        
        return await self.message_bus.send_message(message)
    
    async def send_response(self, original_message_id: str, recipient_id: str,
                           success: bool, result_data: Dict[str, Any], **kwargs):
        """Send a response to a previous request"""
        from .message_types import create_task_response
        
        response = create_task_response(
            sender_id=self.agent_id,
            recipient_id=recipient_id,
            task_id=original_message_id,
            success=success,
            result_data=result_data,
            correlation_id=original_message_id,
            **kwargs
        )
        
        await self.message_bus.send_message(response)
    
    async def check_agent_status(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Check the status of another agent"""
        from .message_types import create_status_check
        
        message = create_status_check(
            sender_id=self.agent_id,
            recipient_id=agent_id
        )
        
        # Set up response waiting
        future = asyncio.Future()
        self.pending_responses[message.message_id] = future
        
        success = await self.message_bus.send_message(message)
        if not success:
            del self.pending_responses[message.message_id]
            return None
        
        try:
            response = await asyncio.wait_for(future, timeout=10)
            return response.to_dict() if hasattr(response, 'to_dict') else response
        except asyncio.TimeoutError:
            if message.message_id in self.pending_responses:
                del self.pending_responses[message.message_id]
            return None
    
    async def broadcast_to_capable_agents(self, capability: str, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Broadcast a task to all agents capable of handling it"""
        from .message_types import TaskType, create_task_request
        
        capable_agents = self.message_bus.find_capable_agents(capability)
        results = {}
        
        for agent_id in capable_agents:
            try:
                response = await self.send_task_request(
                    recipient_id=agent_id,
                    task_type=capability,
                    task_data=task_data
                )
                results[agent_id] = response.to_dict()
            except Exception as e:
                results[agent_id] = {'error': str(e)}
        
        return results
    
    def handle_response(self, message: BaseMessage):
        """Handle incoming response messages"""
        correlation_id = message.correlation_id or message.message_id
        
        if correlation_id in self.pending_responses:
            future = self.pending_responses[correlation_id]
            if not future.done():
                future.set_result(message)
            del self.pending_responses[correlation_id]

class MessageRouter:
    """
    Intelligent message routing system.
    Routes messages to appropriate agents based on capabilities and load.
    """
    
    def __init__(self, message_bus: MessageBus):
        self.message_bus = message_bus
        self.logger = logging.getLogger("message_router")
        self.routing_strategies = {
            'round_robin': self._round_robin_routing,
            'least_loaded': self._least_loaded_routing,
            'capability_based': self._capability_based_routing
        }
        self.agent_counters = defaultdict(int)  # For round robin
    
    async def route_message(self, message: BaseMessage, strategy: str = 'least_loaded') -> bool:
        """
        Route a message using the specified strategy.
        
        Args:
            message: Message to route
            strategy: Routing strategy to use
            
        Returns:
            True if message was routed successfully
        """
        if strategy not in self.routing_strategies:
            self.logger.error(f"Unknown routing strategy: {strategy}")
            return False
        
        return await self.routing_strategies[strategy](message)
    
    async def _round_robin_routing(self, message: BaseMessage) -> bool:
        """Route message using round-robin strategy"""
        if not hasattr(message, 'task_type'):
            return await self.message_bus.send_message(message)
        
        capable_agents = self.message_bus.find_capable_agents(message.task_type.value)
        if not capable_agents:
            self.logger.error(f"No agents capable of handling {message.task_type.value}")
            return False
        
        # Select next agent in round-robin fashion
        agent_key = message.task_type.value
        agent_index = self.agent_counters[agent_key] % len(capable_agents)
        selected_agent = capable_agents[agent_index]
        self.agent_counters[agent_key] += 1
        
        message.recipient_id = selected_agent
        return await self.message_bus.send_message(message)
    
    async def _least_loaded_routing(self, message: BaseMessage) -> bool:
        """Route message to least loaded capable agent"""
        if not hasattr(message, 'task_type'):
            return await self.message_bus.send_message(message)
        
        selected_agent = self.message_bus.get_least_loaded_agent(message.task_type.value)
        if not selected_agent:
            self.logger.error(f"No agents capable of handling {message.task_type.value}")
            return False
        
        message.recipient_id = selected_agent
        return await self.message_bus.send_message(message)
    
    async def _capability_based_routing(self, message: BaseMessage) -> bool:
        """Route message based on agent capabilities and message priority"""
        if not hasattr(message, 'task_type'):
            return await self.message_bus.send_message(message)
        
        capable_agents = self.message_bus.find_capable_agents(message.task_type.value)
        if not capable_agents:
            self.logger.error(f"No agents capable of handling {message.task_type.value}")
            return False
        
        # For high priority messages, use least loaded
        # For normal priority, use round robin
        if message.priority in [Priority.HIGH, Priority.CRITICAL]:
            selected_agent = self.message_bus.get_least_loaded_agent(message.task_type.value)
        else:
            agent_key = message.task_type.value
            agent_index = self.agent_counters[agent_key] % len(capable_agents)
            selected_agent = capable_agents[agent_index]
            self.agent_counters[agent_key] += 1
        
        if not selected_agent:
            return False
        
        message.recipient_id = selected_agent
        return await self.message_bus.send_message(message)