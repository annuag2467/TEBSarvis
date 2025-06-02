"""
Agent Coordinator for TEBSarvis Multi-Agent System
Central coordination hub that orchestrates multi-agent workflows.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, asdict
import uuid

from ..core.base_agent import BaseAgent, AgentCapability
from ..core.agent_registry import AgentRegistry, get_global_registry
from ..core.agent_communication import MessageBus, AgentCommunicator, MessageRouter
from ..core.message_types import (
    TaskType, Priority, create_task_request, create_collaboration_request
)

class WorkflowStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class CoordinationStrategy(Enum):
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"
    PRIORITY_BASED = "priority_based"

@dataclass
class WorkflowTask:
    """Individual task within a workflow"""
    task_id: str
    agent_type: str
    task_type: TaskType
    task_data: Dict[str, Any]
    dependencies: List[str]
    priority: Priority
    timeout_seconds: int
    retry_count: int
    status: WorkflowStatus
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    assigned_agent: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

@dataclass
class Workflow:
    """Complete workflow definition"""
    workflow_id: str
    name: str
    description: str
    tasks: List[WorkflowTask]
    strategy: CoordinationStrategy
    status: WorkflowStatus
    metadata: Dict[str, Any]
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    total_duration: Optional[float] = None

class AgentCoordinator:
    """
    Central coordinator that orchestrates multi-agent workflows.
    Manages task distribution, dependency resolution, and result aggregation.
    """
    
    def __init__(self, registry: Optional[AgentRegistry] = None, 
                 message_bus: Optional[MessageBus] = None):
        self.registry = registry or get_global_registry()
        self.message_bus = message_bus or MessageBus()
        self.message_router = MessageRouter(self.message_bus)
        self.communicator = AgentCommunicator("coordinator", self.message_bus)
        
        self.active_workflows: Dict[str, Workflow] = {}
        self.workflow_history: List[Workflow] = []
        self.coordination_strategies = {
            CoordinationStrategy.SEQUENTIAL: self._execute_sequential,
            CoordinationStrategy.PARALLEL: self._execute_parallel,
            CoordinationStrategy.CONDITIONAL: self._execute_conditional,
            CoordinationStrategy.PRIORITY_BASED: self._execute_priority_based
        }
        
        self.logger = logging.getLogger("agent_coordinator")
        self.max_concurrent_workflows = 10
        self.default_timeout = 300  # 5 minutes
        
        # Performance metrics
        self.metrics = {
            'workflows_executed': 0,
            'workflows_successful': 0,
            'workflows_failed': 0,
            'average_execution_time': 0.0,
            'agent_utilization': {}
        }
    
    async def start(self):
        """Start the coordinator"""
        await self.message_bus.start()
        self.logger.info("Agent Coordinator started")
    
    async def stop(self):
        """Stop the coordinator gracefully"""
        # Cancel all active workflows
        for workflow in self.active_workflows.values():
            await self._cancel_workflow(workflow.workflow_id)
        
        await self.message_bus.stop()
        self.logger.info("Agent Coordinator stopped")
    
    async def execute_workflow(self, workflow_definition: Dict[str, Any]) -> str:
        """
        Execute a multi-agent workflow.
        
        Args:
            workflow_definition: Workflow configuration
            
        Returns:
            Workflow ID for tracking
        """
        try:
            # Create workflow from definition
            workflow = self._create_workflow_from_definition(workflow_definition)
            
            # Validate workflow
            validation_result = await self._validate_workflow(workflow)
            if not validation_result['valid']:
                raise ValueError(f"Invalid workflow: {validation_result['errors']}")
            
            # Check capacity
            if len(self.active_workflows) >= self.max_concurrent_workflows:
                raise RuntimeError("Maximum concurrent workflows reached")
            
            # Add to active workflows
            self.active_workflows[workflow.workflow_id] = workflow
            
            # Start workflow execution
            asyncio.create_task(self._execute_workflow_async(workflow))
            
            self.logger.info(f"Workflow {workflow.workflow_id} started")
            return workflow.workflow_id
            
        except Exception as e:
            self.logger.error(f"Error starting workflow: {str(e)}")
            raise
    
    async def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of a workflow"""
        try:
            # Check active workflows
            if workflow_id in self.active_workflows:
                workflow = self.active_workflows[workflow_id]
                return self._serialize_workflow_status(workflow)
            
            # Check history
            for workflow in self.workflow_history:
                if workflow.workflow_id == workflow_id:
                    return self._serialize_workflow_status(workflow)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting workflow status: {str(e)}")
            return None
    
    async def cancel_workflow(self, workflow_id: str) -> bool:
        """Cancel an active workflow"""
        try:
            if workflow_id in self.active_workflows:
                await self._cancel_workflow(workflow_id)
                return True
            return False
            
        except Exception as e:
            self.logger.error(f"Error cancelling workflow: {str(e)}")
            return False
    
    async def coordinate_incident_resolution(self, incident_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Coordinate a complete incident resolution workflow.
        
        Args:
            incident_data: Incident information
            
        Returns:
            Coordination results with agent outputs
        """
        try:
            # Create incident resolution workflow
            workflow_def = {
                'name': 'incident_resolution',
                'description': f"Resolve incident: {incident_data.get('summary', '')}",
                'strategy': 'sequential',
                'tasks': [
                    {
                        'agent_type': 'context',
                        'task_type': 'metadata_enrichment',
                        'task_data': {'incident_data': incident_data},
                        'dependencies': [],
                        'priority': 'normal'
                    },
                    {
                        'agent_type': 'search',
                        'task_type': 'semantic_search',
                        'task_data': {
                            'query': f"{incident_data.get('summary', '')} {incident_data.get('description', '')}",
                            'max_results': 5
                        },
                        'dependencies': ['context_enrichment'],
                        'priority': 'normal'
                    },
                    {
                        'agent_type': 'resolution',
                        'task_type': 'incident_resolution',
                        'task_data': {'incident_data': incident_data},
                        'dependencies': ['search_similar'],
                        'priority': 'high'
                    },
                    {
                        'agent_type': 'conversation',
                        'task_type': 'response_generation',
                        'task_data': {
                            'user_query': f"How to resolve: {incident_data.get('summary', '')}",
                            'knowledge_sources': ['incidents', 'resolutions']
                        },
                        'dependencies': ['generate_resolution'],
                        'priority': 'normal'
                    }
                ]
            }
            
            workflow_id = await self.execute_workflow(workflow_def)
            
            # Wait for completion with timeout
            result = await self._wait_for_workflow_completion(workflow_id, timeout=600)
            
            return {
                'workflow_id': workflow_id,
                'status': result['status'],
                'incident_id': incident_data.get('id'),
                'resolution_results': result.get('results', {}),
                'coordination_metadata': {
                    'coordinator_id': 'agent_coordinator',
                    'timestamp': datetime.now().isoformat(),
                    'execution_time': result.get('duration', 0)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error coordinating incident resolution: {str(e)}")
            raise
    
    async def coordinate_pattern_analysis(self, analysis_request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Coordinate pattern analysis across multiple agents.
        
        Args:
            analysis_request: Pattern analysis parameters
            
        Returns:
            Comprehensive analysis results
        """
        try:
            workflow_def = {
                'name': 'pattern_analysis',
                'description': 'Comprehensive pattern analysis workflow',
                'strategy': 'parallel',
                'tasks': [
                    {
                        'agent_type': 'pattern_detection',
                        'task_type': 'incident_clustering',
                        'task_data': analysis_request,
                        'dependencies': [],
                        'priority': 'normal'
                    },
                    {
                        'agent_type': 'pattern_detection',
                        'task_type': 'trend_analysis',
                        'task_data': analysis_request,
                        'dependencies': [],
                        'priority': 'normal'
                    },
                    {
                        'agent_type': 'pattern_detection',
                        'task_type': 'anomaly_detection',
                        'task_data': analysis_request,
                        'dependencies': [],
                        'priority': 'normal'
                    },
                    {
                        'agent_type': 'alerting',
                        'task_type': 'evaluate_conditions',
                        'task_data': {
                            'conditions': analysis_request.get('alert_conditions', {}),
                            'data': analysis_request
                        },
                        'dependencies': ['clustering', 'trends', 'anomalies'],
                        'priority': 'high'
                    }
                ]
            }
            
            workflow_id = await self.execute_workflow(workflow_def)
            result = await self._wait_for_workflow_completion(workflow_id)
            
            return {
                'workflow_id': workflow_id,
                'analysis_results': result.get('results', {}),
                'insights_generated': self._combine_pattern_insights(result.get('results', {})),
                'coordination_metadata': {
                    'coordinator_id': 'agent_coordinator',
                    'timestamp': datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error coordinating pattern analysis: {str(e)}")
            raise
    
    async def coordinate_proactive_monitoring(self, monitoring_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Coordinate proactive monitoring workflow.
        
        Args:
            monitoring_config: Monitoring parameters
            
        Returns:
            Monitoring results and alerts
        """
        try:
            workflow_def = {
                'name': 'proactive_monitoring',
                'description': 'Proactive monitoring and alerting workflow',
                'strategy': 'conditional',
                'tasks': [
                    {
                        'agent_type': 'pattern_detection',
                        'task_type': 'anomaly_detection',
                        'task_data': monitoring_config,
                        'dependencies': [],
                        'priority': 'high'
                    },
                    {
                        'agent_type': 'alerting',
                        'task_type': 'real_time_monitoring',
                        'task_data': monitoring_config,
                        'dependencies': [],
                        'priority': 'high'
                    },
                    {
                        'agent_type': 'alerting',
                        'task_type': 'notification_dispatch',
                        'task_data': {
                            'alerts': [],  # Will be populated from previous tasks
                            'notification_channels': monitoring_config.get('channels', ['email'])
                        },
                        'dependencies': ['anomaly_detection', 'monitoring'],
                        'priority': 'critical',
                        'condition': 'alerts_detected'
                    }
                ]
            }
            
            workflow_id = await self.execute_workflow(workflow_def)
            result = await self._wait_for_workflow_completion(workflow_id)
            
            return {
                'workflow_id': workflow_id,
                'monitoring_results': result.get('results', {}),
                'alerts_generated': self._extract_alerts_from_results(result.get('results', {})),
                'coordination_metadata': {
                    'coordinator_id': 'agent_coordinator',
                    'timestamp': datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error coordinating proactive monitoring: {str(e)}")
            raise
    
    def _create_workflow_from_definition(self, definition: Dict[str, Any]) -> Workflow:
        """Create a workflow object from definition"""
        workflow_id = str(uuid.uuid4())
        
        tasks = []
        for i, task_def in enumerate(definition.get('tasks', [])):
            task = WorkflowTask(
                task_id=f"{workflow_id}_task_{i}",
                agent_type=task_def['agent_type'],
                task_type=TaskType(task_def['task_type']),
                task_data=task_def['task_data'],
                dependencies=task_def.get('dependencies', []),
                priority=Priority[task_def.get('priority', 'normal').upper()],
                timeout_seconds=task_def.get('timeout', self.default_timeout),
                retry_count=task_def.get('retry_count', 1),
                status=WorkflowStatus.PENDING
            )
            tasks.append(task)
        
        return Workflow(
            workflow_id=workflow_id,
            name=definition['name'],
            description=definition['description'],
            tasks=tasks,
            strategy=CoordinationStrategy(definition.get('strategy', 'sequential')),
            status=WorkflowStatus.PENDING,
            metadata=definition.get('metadata', {}),
            created_at=datetime.now()
        )
    
    async def _validate_workflow(self, workflow: Workflow) -> Dict[str, Any]:
        """Validate workflow before execution"""
        errors = []
        warnings = []
        
        # Check if required agents are available
        for task in workflow.tasks:
            available_agents = self.registry.find_agents_by_type(task.agent_type)
            if not available_agents:
                errors.append(f"No agents available for type: {task.agent_type}")
        
        # Check dependency graph for cycles
        if self._has_circular_dependencies(workflow.tasks):
            errors.append("Circular dependencies detected in workflow")
        
        # Check for unreachable tasks
        unreachable = self._find_unreachable_tasks(workflow.tasks)
        if unreachable:
            warnings.append(f"Unreachable tasks detected: {unreachable}")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }
    
    async def _execute_workflow_async(self, workflow: Workflow):
        """Execute workflow asynchronously"""
        try:
            workflow.status = WorkflowStatus.RUNNING
            workflow.started_at = datetime.now()
            
            # Execute based on strategy
            strategy_executor = self.coordination_strategies[workflow.strategy]
            await strategy_executor(workflow)
            
            # Update final status
            if all(task.status == WorkflowStatus.COMPLETED for task in workflow.tasks):
                workflow.status = WorkflowStatus.COMPLETED
                self.metrics['workflows_successful'] += 1
            else:
                workflow.status = WorkflowStatus.FAILED
                self.metrics['workflows_failed'] += 1
            
            workflow.completed_at = datetime.now()
            workflow.total_duration = (workflow.completed_at - workflow.started_at).total_seconds()
            
            # Move to history
            self.workflow_history.append(workflow)
            del self.active_workflows[workflow.workflow_id]
            
            # Update metrics
            self.metrics['workflows_executed'] += 1
            self._update_performance_metrics(workflow)
            
            self.logger.info(f"Workflow {workflow.workflow_id} completed with status: {workflow.status}")
            
        except Exception as e:
            workflow.status = WorkflowStatus.FAILED
            self.logger.error(f"Workflow execution error: {str(e)}")
    
    async def _execute_sequential(self, workflow: Workflow):
        """Execute tasks sequentially based on dependencies"""
        completed_tasks = set()
        
        while len(completed_tasks) < len(workflow.tasks):
            # Find tasks ready to execute
            ready_tasks = [
                task for task in workflow.tasks
                if task.status == WorkflowStatus.PENDING and
                all(dep in completed_tasks for dep in task.dependencies)
            ]
            
            if not ready_tasks:
                # Check for failed dependencies
                failed_tasks = [task for task in workflow.tasks if task.status == WorkflowStatus.FAILED]
                if failed_tasks:
                    self.logger.error(f"Workflow failed due to failed tasks: {[t.task_id for t in failed_tasks]}")
                    break
                
                # Wait for running tasks
                await asyncio.sleep(1)
                continue
            
            # Execute next ready task
            task = ready_tasks[0]
            success = await self._execute_task(task, workflow)
            
            if success:
                completed_tasks.add(task.task_id)
            else:
                self.logger.error(f"Task {task.task_id} failed")
                break
    
    async def _execute_parallel(self, workflow: Workflow):
        """Execute tasks in parallel where possible"""
        # Group tasks by dependency level
        task_levels = self._calculate_task_levels(workflow.tasks)
        
        for level in sorted(task_levels.keys()):
            level_tasks = task_levels[level]
            
            # Execute all tasks at this level in parallel
            task_coroutines = [
                self._execute_task(task, workflow) for task in level_tasks
            ]
            
            results = await asyncio.gather(*task_coroutines, return_exceptions=True)
            
            # Check if any tasks failed
            for i, result in enumerate(results):
                if isinstance(result, Exception) or not result:
                    self.logger.error(f"Task {level_tasks[i].task_id} failed")
                    return
    
    async def _execute_conditional(self, workflow: Workflow):
        """Execute tasks with conditional logic"""
        completed_tasks = set()
        
        while len(completed_tasks) < len(workflow.tasks):
            ready_tasks = [
                task for task in workflow.tasks
                if task.status == WorkflowStatus.PENDING and
                all(dep in completed_tasks for dep in task.dependencies) and
                self._evaluate_task_condition(task, workflow, completed_tasks)
            ]
            
            if not ready_tasks:
                await asyncio.sleep(1)
                continue
            
            # Execute ready tasks in parallel
            task_coroutines = [self._execute_task(task, workflow) for task in ready_tasks]
            results = await asyncio.gather(*task_coroutines, return_exceptions=True)
            
            for i, result in enumerate(results):
                if result and not isinstance(result, Exception):
                    completed_tasks.add(ready_tasks[i].task_id)
    
    async def _execute_priority_based(self, workflow: Workflow):
        """Execute tasks based on priority"""
        completed_tasks = set()
        
        while len(completed_tasks) < len(workflow.tasks):
            # Get ready tasks sorted by priority
            ready_tasks = [
                task for task in workflow.tasks
                if task.status == WorkflowStatus.PENDING and
                all(dep in completed_tasks for dep in task.dependencies)
            ]
            
            if not ready_tasks:
                await asyncio.sleep(1)
                continue
            
            # Sort by priority (higher priority value = higher priority)
            ready_tasks.sort(key=lambda t: t.priority.value, reverse=True)
            
            # Execute highest priority task
            task = ready_tasks[0]
            success = await self._execute_task(task, workflow)
            
            if success:
                completed_tasks.add(task.task_id)
    
    async def _execute_task(self, task: WorkflowTask, workflow: Workflow) -> bool:
        """Execute a single task"""
        try:
            task.status = WorkflowStatus.RUNNING
            task.started_at = datetime.now()
            
            # Find best agent for the task
            best_agent = self.registry.get_best_agent_for_capability(task.task_type.value)
            if not best_agent:
                task.status = WorkflowStatus.FAILED
                task.error = f"No available agent for task type: {task.task_type.value}"
                return False
            
            task.assigned_agent = best_agent.agent_id
            
            # Send task to agent
            response = await self.communicator.send_task_request(
                recipient_id=best_agent.agent_id,
                task_type=task.task_type.value,
                task_data=task.task_data,
                timeout_seconds=task.timeout_seconds
            )
            
            # Process response
            if response.success:
                task.status = WorkflowStatus.COMPLETED
                task.result = response.result_data
                task.completed_at = datetime.now()
                return True
            else:
                task.status = WorkflowStatus.FAILED
                task.error = response.error_message
                return False
                
        except Exception as e:
            task.status = WorkflowStatus.FAILED
            task.error = str(e)
            self.logger.error(f"Task execution error: {str(e)}")
            return False
    
    def _has_circular_dependencies(self, tasks: List[WorkflowTask]) -> bool:
        """Check for circular dependencies in task graph"""
        # Build dependency graph
        graph = {task.task_id: task.dependencies for task in tasks}
        
        # Use DFS to detect cycles
        visited = set()
        rec_stack = set()
        
        def dfs(node):
            if node in rec_stack:
                return True
            if node in visited:
                return False
            
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in graph.get(node, []):
                if dfs(neighbor):
                    return True
            
            rec_stack.remove(node)
            return False
        
        for task_id in graph:
            if dfs(task_id):
                return True
        
        return False
    
    def _find_unreachable_tasks(self, tasks: List[WorkflowTask]) -> List[str]:
        """Find tasks that can never be executed due to dependencies"""
        task_ids = {task.task_id for task in tasks}
        unreachable = []
        
        for task in tasks:
            for dep in task.dependencies:
                if dep not in task_ids:
                    unreachable.append(task.task_id)
                    break
        
        return unreachable
    
    def _calculate_task_levels(self, tasks: List[WorkflowTask]) -> Dict[int, List[WorkflowTask]]:
        """Calculate dependency levels for parallel execution"""
        levels = {}
        task_levels = {}
        
        # Calculate level for each task
        def get_level(task_id, tasks_dict):
            if task_id in task_levels:
                return task_levels[task_id]
            
            task = tasks_dict[task_id]
            if not task.dependencies:
                level = 0
            else:
                level = max(get_level(dep, tasks_dict) for dep in task.dependencies) + 1
            
            task_levels[task_id] = level
            return level
        
        tasks_dict = {task.task_id: task for task in tasks}
        
        for task in tasks:
            level = get_level(task.task_id, tasks_dict)
            if level not in levels:
                levels[level] = []
            levels[level].append(task)
        
        return levels
    
    def _evaluate_task_condition(self, task: WorkflowTask, workflow: Workflow, completed_tasks: set) -> bool:
        """Evaluate if a conditional task should be executed"""
        # Simple condition evaluation - can be extended
        if hasattr(task, 'condition'):
            condition = task.condition
            
            if condition == 'alerts_detected':
                # Check if previous tasks generated alerts
                for completed_task_id in completed_tasks:
                    completed_task = next((t for t in workflow.tasks if t.task_id == completed_task_id), None)
                    if completed_task and completed_task.result:
                        if 'alerts' in completed_task.result and completed_task.result['alerts']:
                            return True
                return False
        
        return True  # Default: execute if dependencies are met
    
    async def _wait_for_workflow_completion(self, workflow_id: str, timeout: int = 300) -> Dict[str, Any]:
        """Wait for workflow completion with timeout"""
        start_time = datetime.now()
        
        while (datetime.now() - start_time).total_seconds() < timeout:
            status = await self.get_workflow_status(workflow_id)
            if not status:
                raise ValueError(f"Workflow {workflow_id} not found")
            
            if status['status'] in ['completed', 'failed', 'cancelled']:
                return status
            
            await asyncio.sleep(1)
        
        # Timeout reached
        await self.cancel_workflow(workflow_id)
        raise TimeoutError(f"Workflow {workflow_id} timed out after {timeout} seconds")
    
    async def _cancel_workflow(self, workflow_id: str):
        """Cancel a workflow and all its tasks"""
        if workflow_id in self.active_workflows:
            workflow = self.active_workflows[workflow_id]
            workflow.status = WorkflowStatus.CANCELLED
            
            # Cancel running tasks
            for task in workflow.tasks:
                if task.status == WorkflowStatus.RUNNING:
                    task.status = WorkflowStatus.CANCELLED
            
            self.logger.info(f"Workflow {workflow_id} cancelled")
    
    def _serialize_workflow_status(self, workflow: Workflow) -> Dict[str, Any]:
        """Serialize workflow status for API response"""
        return {
            'workflow_id': workflow.workflow_id,
            'name': workflow.name,
            'status': workflow.status.value,
            'progress': {
                'total_tasks': len(workflow.tasks),
                'completed_tasks': len([t for t in workflow.tasks if t.status == WorkflowStatus.COMPLETED]),
                'failed_tasks': len([t for t in workflow.tasks if t.status == WorkflowStatus.FAILED]),
                'percentage': (len([t for t in workflow.tasks if t.status == WorkflowStatus.COMPLETED]) / len(workflow.tasks)) * 100
            },
            'tasks': [
                {
                    'task_id': task.task_id,
                    'agent_type': task.agent_type,
                    'status': task.status.value,
                    'assigned_agent': task.assigned_agent,
                    'error': task.error,
                    'duration': (task.completed_at - task.started_at).total_seconds() if task.started_at and task.completed_at else None
                }
                for task in workflow.tasks
            ],
            'results': {
                task.task_id: task.result 
                for task in workflow.tasks 
                if task.result is not None
            },
            'duration': workflow.total_duration,
            'started_at': workflow.started_at.isoformat() if workflow.started_at else None,
            'completed_at': workflow.completed_at.isoformat() if workflow.completed_at else None
        }
    
    def _combine_pattern_insights(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Combine insights from multiple pattern analysis agents"""
        combined_insights = {
            'key_findings': [],
            'recommendations': [],
            'risk_factors': [],
            'confidence_score': 0.0
        }
        
        # Extract insights from each agent's results
        for task_id, result in results.items():
            if 'insights' in result:
                insights = result['insights']
                combined_insights['key_findings'].extend(insights.get('key_findings', []))
                combined_insights['recommendations'].extend(insights.get('recommendations', []))
                combined_insights['risk_factors'].extend(insights.get('risk_factors', []))
        
        # Calculate average confidence
        confidence_scores = [
            result.get('confidence_score', 0) 
            for result in results.values() 
            if 'confidence_score' in result
        ]
        
        if confidence_scores:
            combined_insights['confidence_score'] = sum(confidence_scores) / len(confidence_scores)
        
        return combined_insights
    
    def _extract_alerts_from_results(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract alerts from agent results"""
        alerts = []
        
        for task_id, result in results.items():
            if 'alerts' in result:
                alerts.extend(result['alerts'])
            elif 'generated_alerts' in result:
                alerts.extend(result['generated_alerts'])
        
        return alerts
    
    def _update_performance_metrics(self, workflow: Workflow):
        """Update performance metrics after workflow completion"""
        if workflow.total_duration:
            # Update average execution time
            total_workflows = self.metrics['workflows_executed']
            current_avg = self.metrics['average_execution_time']
            
            new_avg = ((current_avg * (total_workflows - 1)) + workflow.total_duration) / total_workflows
            self.metrics['average_execution_time'] = new_avg
        
        # Update agent utilization
        for task in workflow.tasks:
            if task.assigned_agent:
                if task.assigned_agent not in self.metrics['agent_utilization']:
                    self.metrics['agent_utilization'][task.assigned_agent] = 0
                self.metrics['agent_utilization'][task.assigned_agent] += 1
    
    def get_coordination_metrics(self) -> Dict[str, Any]:
        """Get coordination performance metrics"""
        return {
            'workflows_executed': self.metrics['workflows_executed'],
            'workflows_successful': self.metrics['workflows_successful'],
            'workflows_failed': self.metrics['workflows_failed'],
            'success_rate': (self.metrics['workflows_successful'] / max(self.metrics['workflows_executed'], 1)) * 100,
            'average_execution_time': self.metrics['average_execution_time'],
            'active_workflows': len(self.active_workflows),
            'agent_utilization': self.metrics['agent_utilization'],
            'timestamp': datetime.now().isoformat()
        }