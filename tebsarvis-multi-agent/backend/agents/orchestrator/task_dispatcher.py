"""
Task Dispatcher for TEBSarvis Multi-Agent System
Handles task routing, load balancing, and agent assignment optimization.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import heapq
import random

from ..core.base_agent import BaseAgent, AgentCapability
from ..core.agent_registry import AgentRegistry, AgentRegistration
from ..core.agent_communication import MessageBus, AgentCommunicator
from ..core.message_types import TaskType, Priority, TaskRequestMessage

class LoadBalancingStrategy(Enum):
    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    CAPABILITY_BASED = "capability_based"
    GEOGRAPHIC = "geographic"
    PERFORMANCE_BASED = "performance_based"

class TaskPriority(Enum):
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4
    EMERGENCY = 5

@dataclass
class TaskRequest:
    """Task request with routing information"""
    task_id: str
    task_type: TaskType
    task_data: Dict[str, Any]
    priority: TaskPriority
    requester_id: str
    created_at: datetime
    timeout_seconds: int
    retry_count: int
    routing_hints: Dict[str, Any]
    assigned_agent: Optional[str] = None
    routing_attempts: int = 0
    last_attempt_at: Optional[datetime] = None

@dataclass
class AgentLoadMetrics:
    """Agent load and performance metrics"""
    agent_id: str
    current_tasks: int
    max_capacity: int
    cpu_utilization: float
    memory_utilization: float
    average_response_time: float
    success_rate: float
    last_updated: datetime
    geographic_location: str = "default"
    specialized_capabilities: List[str] = None

class TaskDispatcher:
    """
    Intelligent task dispatcher that routes tasks to optimal agents.
    Implements multiple load balancing strategies and performance optimization.
    """
    
    def __init__(self, registry: AgentRegistry, message_bus: MessageBus):
        self.registry = registry
        self.message_bus = message_bus
        self.communicator = AgentCommunicator("task_dispatcher", message_bus)
        
        # Task queues by priority
        self.task_queues = {
            priority: deque() for priority in TaskPriority
        }
        
        # Agent metrics and state
        self.agent_metrics: Dict[str, AgentLoadMetrics] = {}
        self.agent_performance_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        # Load balancing state
        self.round_robin_counters: Dict[str, int] = defaultdict(int)
        self.weighted_counters: Dict[str, int] = defaultdict(int)
        
        # Configuration
        self.load_balancing_strategy = LoadBalancingStrategy.LEAST_LOADED
        self.max_queue_size = 1000
        self.metrics_update_interval = 30  # seconds
        self.performance_history_limit = 100
        
        # Statistics
        self.dispatch_stats = {
            'tasks_dispatched': 0,
            'tasks_failed': 0,
            'average_dispatch_time': 0.0,
            'agent_utilization': {},
            'strategy_effectiveness': {}
        }
        
        self.logger = logging.getLogger("task_dispatcher")
        
        # Background tasks
        self.running = False
        self.background_tasks = []
    
    async def start(self):
        """Start the task dispatcher"""
        self.running = True
        
        # Start background monitoring tasks
        self.background_tasks = [
            asyncio.create_task(self._update_agent_metrics()),
            asyncio.create_task(self._process_task_queues()),
            asyncio.create_task(self._cleanup_old_metrics()),
            asyncio.create_task(self._optimize_routing_strategy())
        ]
        
        self.logger.info("Task Dispatcher started")
    
    async def stop(self):
        """Stop the task dispatcher gracefully"""
        self.running = False
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self.background_tasks:
            await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        self.logger.info("Task Dispatcher stopped")
    
    async def dispatch_task(self, task_request: TaskRequest) -> Optional[str]:
        """
        Dispatch a task to the best available agent.
        
        Args:
            task_request: Task to be dispatched
            
        Returns:
            Agent ID if successfully dispatched, None otherwise
        """
        try:
            start_time = datetime.now()
            
            # Validate task request
            if not self._validate_task_request(task_request):
                self.logger.error(f"Invalid task request: {task_request.task_id}")
                return None
            
            # Check queue capacity
            if self._get_total_queue_size() >= self.max_queue_size:
                self.logger.warning("Task queue at capacity, rejecting task")
                return None
            
            # Add to appropriate priority queue
            self.task_queues[task_request.priority].append(task_request)
            
            # Try immediate dispatch for high priority tasks
            if task_request.priority in [TaskPriority.CRITICAL, TaskPriority.EMERGENCY]:
                agent_id = await self._dispatch_immediately(task_request)
                if agent_id:
                    self._update_dispatch_stats(start_time, True)
                    return agent_id
            
            self.logger.info(f"Task {task_request.task_id} queued for dispatch")
            return "queued"
            
        except Exception as e:
            self.logger.error(f"Error dispatching task: {str(e)}")
            self._update_dispatch_stats(start_time, False)
            return None
    
    async def get_optimal_agent(self, task_type: TaskType, 
                              priority: TaskPriority = TaskPriority.NORMAL,
                              routing_hints: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        Get the optimal agent for a task type without dispatching.
        
        Args:
            task_type: Type of task
            priority: Task priority
            routing_hints: Additional routing information
            
        Returns:
            Optimal agent ID or None
        """
        try:
            # Get capable agents
            capable_agents = self.registry.find_agents_by_capability(task_type.value)
            
            if not capable_agents:
                return None
            
            # Apply load balancing strategy
            return await self._select_agent_by_strategy(
                capable_agents, task_type, priority, routing_hints
            )
            
        except Exception as e:
            self.logger.error(f"Error finding optimal agent: {str(e)}")
            return None
    
    async def update_agent_load(self, agent_id: str, load_data: Dict[str, Any]):
        """
        Update agent load metrics.
        
        Args:
            agent_id: Agent identifier
            load_data: Current load and performance data
        """
        try:
            metrics = AgentLoadMetrics(
                agent_id=agent_id,
                current_tasks=load_data.get('current_tasks', 0),
                max_capacity=load_data.get('max_capacity', 10),
                cpu_utilization=load_data.get('cpu_utilization', 0.0),
                memory_utilization=load_data.get('memory_utilization', 0.0),
                average_response_time=load_data.get('average_response_time', 0.0),
                success_rate=load_data.get('success_rate', 100.0),
                last_updated=datetime.now(),
                geographic_location=load_data.get('location', 'default'),
                specialized_capabilities=load_data.get('specialized_capabilities', [])
            )
            
            self.agent_metrics[agent_id] = metrics
            
            # Update performance history
            performance_record = {
                'timestamp': datetime.now(),
                'response_time': metrics.average_response_time,
                'success_rate': metrics.success_rate,
                'cpu_utilization': metrics.cpu_utilization
            }
            
            history = self.agent_performance_history[agent_id]
            history.append(performance_record)
            
            # Limit history size
            if len(history) > self.performance_history_limit:
                history.pop(0)
            
        except Exception as e:
            self.logger.error(f"Error updating agent load: {str(e)}")
    
    async def set_load_balancing_strategy(self, strategy: LoadBalancingStrategy):
        """Change the load balancing strategy"""
        self.load_balancing_strategy = strategy
        self.logger.info(f"Load balancing strategy changed to: {strategy.value}")
    
    async def get_agent_recommendations(self, task_type: TaskType, 
                                      count: int = 3) -> List[Tuple[str, float]]:
        """
        Get ranked list of agent recommendations for a task type.
        
        Args:
            task_type: Type of task
            count: Number of recommendations to return
            
        Returns:
            List of (agent_id, score) tuples
        """
        try:
            capable_agents = self.registry.find_agents_by_capability(task_type.value)
            
            if not capable_agents:
                return []
            
            # Score each agent
            agent_scores = []
            for agent_reg in capable_agents:
                score = await self._calculate_agent_score(agent_reg, task_type)
                agent_scores.append((agent_reg.agent_id, score))
            
            # Sort by score (descending) and return top N
            agent_scores.sort(key=lambda x: x[1], reverse=True)
            return agent_scores[:count]
            
        except Exception as e:
            self.logger.error(f"Error getting agent recommendations: {str(e)}")
            return []
    
    async def _dispatch_immediately(self, task_request: TaskRequest) -> Optional[str]:
        """Attempt immediate dispatch for high priority tasks"""
        try:
            capable_agents = self.registry.find_agents_by_capability(task_request.task_type.value)
            
            if not capable_agents:
                return None
            
            # Find agent with immediate capacity
            for agent_reg in capable_agents:
                metrics = self.agent_metrics.get(agent_reg.agent_id)
                if metrics and metrics.current_tasks < metrics.max_capacity:
                    # Try to dispatch
                    success = await self._send_task_to_agent(task_request, agent_reg.agent_id)
                    if success:
                        return agent_reg.agent_id
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error in immediate dispatch: {str(e)}")
            return None
    
    async def _process_task_queues(self):
        """Background task to process queued tasks"""
        while self.running:
            try:
                # Process queues in priority order
                for priority in sorted(TaskPriority, key=lambda p: p.value, reverse=True):
                    queue = self.task_queues[priority]
                    
                    if queue:
                        task_request = queue.popleft()
                        agent_id = await self._find_and_assign_agent(task_request)
                        
                        if agent_id:
                            await self._send_task_to_agent(task_request, agent_id)
                        else:
                            # Re-queue if no agent available
                            queue.appendleft(task_request)
                            break  # Avoid infinite loop
                
                await asyncio.sleep(1)  # Prevent busy waiting
                
            except Exception as e:
                self.logger.error(f"Error processing task queues: {str(e)}")
                await asyncio.sleep(5)
    
    async def _find_and_assign_agent(self, task_request: TaskRequest) -> Optional[str]:
        """Find and assign the best agent for a task"""
        try:
            capable_agents = self.registry.find_agents_by_capability(task_request.task_type.value)
            
            if not capable_agents:
                return None
            
            # Apply load balancing strategy
            selected_agent = await self._select_agent_by_strategy(
                capable_agents, 
                task_request.task_type, 
                task_request.priority, 
                task_request.routing_hints
            )
            
            if selected_agent:
                task_request.assigned_agent = selected_agent
                task_request.routing_attempts += 1
                task_request.last_attempt_at = datetime.now()
            
            return selected_agent
            
        except Exception as e:
            self.logger.error(f"Error finding agent: {str(e)}")
            return None
    
    async def _select_agent_by_strategy(self, capable_agents: List[AgentRegistration],
                                      task_type: TaskType, priority: TaskPriority,
                                      routing_hints: Optional[Dict[str, Any]]) -> Optional[str]:
        """Select agent based on current load balancing strategy"""
        
        if self.load_balancing_strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return self._round_robin_selection(capable_agents, task_type.value)
        
        elif self.load_balancing_strategy == LoadBalancingStrategy.LEAST_LOADED:
            return self._least_loaded_selection(capable_agents)
        
        elif self.load_balancing_strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
            return self._weighted_round_robin_selection(capable_agents, task_type.value)
        
        elif self.load_balancing_strategy == LoadBalancingStrategy.CAPABILITY_BASED:
            return await self._capability_based_selection(capable_agents, task_type, routing_hints)
        
        elif self.load_balancing_strategy == LoadBalancingStrategy.PERFORMANCE_BASED:
            return await self._performance_based_selection(capable_agents, task_type)
        
        elif self.load_balancing_strategy == LoadBalancingStrategy.GEOGRAPHIC:
            return self._geographic_selection(capable_agents, routing_hints)
        
        else:
            # Default to least loaded
            return self._least_loaded_selection(capable_agents)
    
    def _round_robin_selection(self, capable_agents: List[AgentRegistration], 
                             task_type: str) -> Optional[str]:
        """Round robin agent selection"""
        if not capable_agents:
            return None
        
        counter = self.round_robin_counters[task_type]
        selected_agent = capable_agents[counter % len(capable_agents)]
        self.round_robin_counters[task_type] = counter + 1
        
        return selected_agent.agent_id
    
    def _least_loaded_selection(self, capable_agents: List[AgentRegistration]) -> Optional[str]:
        """Select agent with least current load"""
        if not capable_agents:
            return None
        
        best_agent = None
        min_load = float('inf')
        
        for agent_reg in capable_agents:
            metrics = self.agent_metrics.get(agent_reg.agent_id)
            if metrics:
                # Calculate load percentage
                load_percentage = metrics.current_tasks / max(metrics.max_capacity, 1)
                
                # Consider CPU utilization as well
                combined_load = (load_percentage + metrics.cpu_utilization / 100) / 2
                
                if combined_load < min_load:
                    min_load = combined_load
                    best_agent = agent_reg.agent_id
            else:
                # No metrics available, consider it available
                if min_load > 0:
                    best_agent = agent_reg.agent_id
                    min_load = 0
        
        return best_agent
    
    def _weighted_round_robin_selection(self, capable_agents: List[AgentRegistration],
                                      task_type: str) -> Optional[str]:
        """Weighted round robin based on agent capacity"""
        if not capable_agents:
            return None
        
        # Calculate weights based on agent capacity and performance
        weights = []
        for agent_reg in capable_agents:
            metrics = self.agent_metrics.get(agent_reg.agent_id)
            if metrics:
                # Weight based on capacity and success rate
                weight = metrics.max_capacity * (metrics.success_rate / 100)
            else:
                weight = 1.0  # Default weight
            weights.append(weight)
        
        # Weighted selection
        total_weight = sum(weights)
        if total_weight == 0:
            return self._round_robin_selection(capable_agents, task_type)
        
        # Use counter to determine selection
        counter = self.weighted_counters[task_type]
        
        # Find agent based on weighted distribution
        cumulative_weight = 0
        normalized_counter = counter % int(total_weight)
        
        for i, weight in enumerate(weights):
            cumulative_weight += weight
            if normalized_counter < cumulative_weight:
                self.weighted_counters[task_type] = counter + 1
                return capable_agents[i].agent_id
        
        # Fallback
        return capable_agents[0].agent_id
    
    async def _capability_based_selection(self, capable_agents: List[AgentRegistration],
                                        task_type: TaskType, 
                                        routing_hints: Optional[Dict[str, Any]]) -> Optional[str]:
        """Select agent based on specialized capabilities"""
        if not capable_agents:
            return None
        
        # Look for specialized capabilities in routing hints
        required_specializations = routing_hints.get('specializations', []) if routing_hints else []
        
        if required_specializations:
            # Find agents with required specializations
            specialized_agents = []
            for agent_reg in capable_agents:
                metrics = self.agent_metrics.get(agent_reg.agent_id)
                if metrics and metrics.specialized_capabilities:
                    if any(spec in metrics.specialized_capabilities for spec in required_specializations):
                        specialized_agents.append(agent_reg)
            
            if specialized_agents:
                return self._least_loaded_selection(specialized_agents)
        
        # Fallback to least loaded
        return self._least_loaded_selection(capable_agents)
    
    async def _performance_based_selection(self, capable_agents: List[AgentRegistration],
                                         task_type: TaskType) -> Optional[str]:
        """Select agent based on historical performance"""
        if not capable_agents:
            return None
        
        best_agent = None
        best_score = -1
        
        for agent_reg in capable_agents:
            score = await self._calculate_agent_score(agent_reg, task_type)
            if score > best_score:
                best_score = score
                best_agent = agent_reg.agent_id
        
        return best_agent
    
    def _geographic_selection(self, capable_agents: List[AgentRegistration],
                            routing_hints: Optional[Dict[str, Any]]) -> Optional[str]:
        """Select agent based on geographic proximity"""
        if not capable_agents:
            return None
        
        preferred_location = routing_hints.get('location') if routing_hints else None
        
        if preferred_location:
            # Find agents in preferred location
            local_agents = []
            for agent_reg in capable_agents:
                metrics = self.agent_metrics.get(agent_reg.agent_id)
                if metrics and metrics.geographic_location == preferred_location:
                    local_agents.append(agent_reg)
            
            if local_agents:
                return self._least_loaded_selection(local_agents)
        
        # Fallback to least loaded
        return self._least_loaded_selection(capable_agents)
    
    async def _calculate_agent_score(self, agent_reg: AgentRegistration, 
                                   task_type: TaskType) -> float:
        """Calculate composite score for agent selection"""
        base_score = 50.0  # Base score
        
        metrics = self.agent_metrics.get(agent_reg.agent_id)
        if not metrics:
            return base_score
        
        # Performance factors
        success_rate_score = metrics.success_rate
        
        # Load factor (inverse - lower load is better)
        load_percentage = metrics.current_tasks / max(metrics.max_capacity, 1)
        load_score = max(0, 100 - (load_percentage * 100))
        
        # Response time factor (inverse - lower time is better)
        if metrics.average_response_time > 0:
            response_time_score = max(0, 100 - (metrics.average_response_time / 10))
        else:
            response_time_score = 100
        
        # Resource utilization factor
        cpu_score = max(0, 100 - metrics.cpu_utilization)
        memory_score = max(0, 100 - metrics.memory_utilization)
        
        # Weighted composite score
        composite_score = (
            success_rate_score * 0.3 +
            load_score * 0.25 +
            response_time_score * 0.2 +
            cpu_score * 0.15 +
            memory_score * 0.1
        )
        
        return composite_score
    
    async def _send_task_to_agent(self, task_request: TaskRequest, agent_id: str) -> bool:
        """Send task to specific agent"""
        try:
            response = await self.communicator.send_task_request(
                recipient_id=agent_id,
                task_type=task_request.task_type.value,
                task_data=task_request.task_data,
                timeout_seconds=task_request.timeout_seconds
            )
            
            if response.success:
                self.logger.info(f"Task {task_request.task_id} sent to agent {agent_id}")
                return True
            else:
                self.logger.error(f"Failed to send task {task_request.task_id} to agent {agent_id}: {response.error_message}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error sending task to agent: {str(e)}")
            return False
    
    async def _update_agent_metrics(self):
        """Background task to update agent metrics"""
        while self.running:
            try:
                # Query registry for active agents
                active_agents = self.registry.get_all_active_agents()
                
                for agent_reg in active_agents:
                    # Request status update from agent
                    try:
                        status = await self.communicator.check_agent_status(agent_reg.agent_id)
                        if status:
                            await self.update_agent_load(agent_reg.agent_id, status)
                    except Exception as e:
                        self.logger.warning(f"Failed to get status from agent {agent_reg.agent_id}: {str(e)}")
                
                await asyncio.sleep(self.metrics_update_interval)
                
            except Exception as e:
                self.logger.error(f"Error updating agent metrics: {str(e)}")
                await asyncio.sleep(10)
    
    async def _cleanup_old_metrics(self):
        """Clean up old performance metrics"""
        while self.running:
            try:
                cutoff_time = datetime.now() - timedelta(hours=24)
                
                # Clean up old performance history
                for agent_id, history in self.agent_performance_history.items():
                    self.agent_performance_history[agent_id] = [
                        record for record in history
                        if record['timestamp'] > cutoff_time
                    ]
                
                # Clean up stale agent metrics
                stale_agents = []
                for agent_id, metrics in self.agent_metrics.items():
                    if metrics.last_updated < cutoff_time:
                        stale_agents.append(agent_id)
                
                for agent_id in stale_agents:
                    del self.agent_metrics[agent_id]
                
                await asyncio.sleep(3600)  # Run every hour
                
            except Exception as e:
                self.logger.error(f"Error in cleanup: {str(e)}")
                await asyncio.sleep(300)
    
    async def _optimize_routing_strategy(self):
        """Optimize routing strategy based on performance"""
        while self.running:
            try:
                # Analyze performance of current strategy
                current_effectiveness = self._calculate_strategy_effectiveness()
                
                # Store effectiveness data
                strategy_name = self.load_balancing_strategy.value
                if strategy_name not in self.dispatch_stats['strategy_effectiveness']:
                    self.dispatch_stats['strategy_effectiveness'][strategy_name] = []
                
                self.dispatch_stats['strategy_effectiveness'][strategy_name].append({
                    'timestamp': datetime.now(),
                    'effectiveness': current_effectiveness
                })
                
                # Consider strategy changes (simplified logic)
                if current_effectiveness < 0.7:  # 70% threshold
                    await self._suggest_strategy_change()
                
                await asyncio.sleep(300)  # Run every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Error optimizing routing strategy: {str(e)}")
                await asyncio.sleep(60)
    
    def _calculate_strategy_effectiveness(self) -> float:
        """Calculate effectiveness of current routing strategy"""
        try:
            total_tasks = self.dispatch_stats['tasks_dispatched']
            failed_tasks = self.dispatch_stats['tasks_failed']
            
            if total_tasks == 0:
                return 1.0
            
            success_rate = (total_tasks - failed_tasks) / total_tasks
            
            # Factor in average dispatch time
            avg_dispatch_time = self.dispatch_stats['average_dispatch_time']
            time_factor = max(0, 1 - (avg_dispatch_time / 60))  # Normalize to 1 minute
            
            # Composite effectiveness score
            effectiveness = (success_rate * 0.7) + (time_factor * 0.3)
            
            return min(effectiveness, 1.0)
            
        except Exception:
            return 0.5  # Default value
    
    async def _suggest_strategy_change(self):
        """Suggest a different routing strategy based on current conditions"""
        try:
            # Simple strategy suggestion logic
            current_strategy = self.load_balancing_strategy
            
            # Analyze current conditions
            total_agents = len(self.agent_metrics)
            high_load_agents = len([
                metrics for metrics in self.agent_metrics.values()
                if metrics.current_tasks / max(metrics.max_capacity, 1) > 0.8
            ])
            
            load_imbalance = high_load_agents / max(total_agents, 1)
            
            # Suggest strategy based on conditions
            if load_imbalance > 0.5:  # High load imbalance
                if current_strategy != LoadBalancingStrategy.LEAST_LOADED:
                    self.logger.info("Suggesting strategy change to LEAST_LOADED due to load imbalance")
            
            elif self._has_performance_variations():
                if current_strategy != LoadBalancingStrategy.PERFORMANCE_BASED:
                    self.logger.info("Suggesting strategy change to PERFORMANCE_BASED due to performance variations")
            
        except Exception as e:
            self.logger.error(f"Error suggesting strategy change: {str(e)}")
    
    def _has_performance_variations(self) -> bool:
        """Check if there are significant performance variations between agents"""
        try:
            if len(self.agent_metrics) < 2:
                return False
            
            success_rates = [metrics.success_rate for metrics in self.agent_metrics.values()]
            response_times = [metrics.average_response_time for metrics in self.agent_metrics.values()]
            
            # Check for significant variations
            success_rate_range = max(success_rates) - min(success_rates)
            avg_response_time = sum(response_times) / len(response_times)
            response_time_variation = max(abs(rt - avg_response_time) for rt in response_times)
            
            return success_rate_range > 20 or response_time_variation > avg_response_time * 0.5
            
        except Exception:
            return False
    
    def _validate_task_request(self, task_request: TaskRequest) -> bool:
        """Validate task request before processing"""
        return (
            task_request.task_id and
            task_request.task_type and
            task_request.requester_id and
            task_request.timeout_seconds > 0
        )
    
    def _get_total_queue_size(self) -> int:
        """Get total size of all task queues"""
        return sum(len(queue) for queue in self.task_queues.values())
    
    def _update_dispatch_stats(self, start_time: datetime, success: bool):
        """Update dispatch statistics"""
        dispatch_time = (datetime.now() - start_time).total_seconds()
        
        self.dispatch_stats['tasks_dispatched'] += 1
        if not success:
            self.dispatch_stats['tasks_failed'] += 1
        
        # Update average dispatch time
        total_tasks = self.dispatch_stats['tasks_dispatched']
        current_avg = self.dispatch_stats['average_dispatch_time']
        new_avg = ((current_avg * (total_tasks - 1)) + dispatch_time) / total_tasks
        self.dispatch_stats['average_dispatch_time'] = new_avg
    
    def get_dispatcher_status(self) -> Dict[str, Any]:
        """Get current dispatcher status and metrics"""
        queue_sizes = {
            priority.name: len(queue) 
            for priority, queue in self.task_queues.items()
        }
        
        agent_status = {}
        for agent_id, metrics in self.agent_metrics.items():
            agent_status[agent_id] = {
                'current_load': metrics.current_tasks / max(metrics.max_capacity, 1),
                'success_rate': metrics.success_rate,
                'response_time': metrics.average_response_time,
                'last_updated': metrics.last_updated.isoformat()
            }
        
        return {
            'status': 'running' if self.running else 'stopped',
            'load_balancing_strategy': self.load_balancing_strategy.value,
            'queue_sizes': queue_sizes,
            'total_queue_size': self._get_total_queue_size(),
            'agent_count': len(self.agent_metrics),
            'dispatch_stats': self.dispatch_stats,
            'agent_status': agent_status,
            'timestamp': datetime.now().isoformat()
        }