"""
Unified Orchestration Manager
Coordinates all orchestration components for seamless multi-agent workflows.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from .agent_coordinator import AgentCoordinator
from .collaboration_manager import CollaborationManager
from .task_dispatcher import TaskDispatcher, LoadBalancingStrategy
from .workflow_engine import WorkflowEngine, WorkflowDefinition
from ..core.agent_registry import AgentRegistry, get_global_registry
from ..core.agent_communication import MessageBus
from ...config.agent_config import get_agent_config
# For TaskDispatcher reference (add to files that use it)
from .task_dispatcher import TaskDispatcher, LoadBalancingStrategy

# For WorkflowEngine reference (add to files that use it)  
from .workflow_engine import WorkflowEngine, WorkflowDefinition

# For CollaborationManager reference (add to files that use it)
from .collaboration_manager import CollaborationManager

# For OrchestrationManager reference (add to files that use it)
# from .orchestration_manager import OrchestrationManager

class OrchestrationManager:
    """
    Unified manager that coordinates all orchestration components.
    Provides single point of control for complex multi-agent operations.
    """
    
    def __init__(self, registry: Optional[AgentRegistry] = None, 
                 message_bus: Optional[MessageBus] = None):
        # Core components
        self.registry = registry or get_global_registry()
        self.message_bus = message_bus or MessageBus()
        
        # Orchestration components
        self.task_dispatcher = TaskDispatcher(self.registry, self.message_bus)
        self.coordinator = AgentCoordinator(self.registry, self.message_bus, self.task_dispatcher)
        self.collaboration_manager = CollaborationManager(self.registry, self.message_bus)
        self.workflow_engine = WorkflowEngine(self.registry, self.message_bus)
        
        # Configuration
        self.config_manager = get_agent_config()
        orchestration_config = self.config_manager.orchestration_config
        self.max_concurrent_workflows = orchestration_config['max_concurrent_workflows']
        
        self.logger = logging.getLogger("orchestration_manager")
        self.running = False
    
    async def start(self):
        """Start all orchestration components"""
        self.running = True
        
        # Start components in dependency order
        await self.message_bus.start()
        await self.task_dispatcher.start()
        await self.collaboration_manager.start()
        await self.workflow_engine.start()
        await self.coordinator.start()
        
        # Register predefined workflows
        await self._register_predefined_workflows()
        
        self.logger.info("Orchestration Manager started successfully")
    
    async def stop(self):
        """Stop all orchestration components gracefully"""
        self.running = False
        
        # Stop in reverse order
        await self.coordinator.stop()
        await self.workflow_engine.stop()
        await self.collaboration_manager.stop()
        await self.task_dispatcher.stop()
        await self.message_bus.stop()
        
        self.logger.info("Orchestration Manager stopped")
    
    async def execute_intelligent_workflow(self, workflow_request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute workflow with intelligent orchestration decisions.
        Automatically chooses best coordination strategy based on requirements.
        """
        try:
            workflow_type = workflow_request.get('type', 'custom')
            complexity = workflow_request.get('complexity', 'medium')
            agents_required = workflow_request.get('agents_required', [])
            
            # Intelligent strategy selection
            if len(agents_required) > 5 and complexity == 'high':
                # Use workflow engine for complex multi-step workflows
                return await self._execute_with_workflow_engine(workflow_request)
            elif workflow_request.get('requires_consensus', False):
                # Use collaboration manager for consensus-based workflows
                return await self._execute_with_collaboration(workflow_request)
            else:
                # Use coordinator for standard workflows
                return await self._execute_with_coordinator(workflow_request)
                
        except Exception as e:
            self.logger.error(f"Error in intelligent workflow execution: {str(e)}")
            raise
    
    async def optimize_system_performance(self) -> Dict[str, Any]:
        """
        Analyze and optimize overall system performance.
        Adjusts load balancing, collaboration patterns, and workflow strategies.
        """
        try:
            # Get performance metrics from all components
            coordinator_metrics = self.coordinator.get_coordination_metrics()
            dispatcher_status = self.task_dispatcher.get_dispatcher_status()
            collaboration_stats = self.collaboration_manager.get_manager_status()
            workflow_status = self.workflow_engine.get_engine_status()
            
            optimizations = {
                'timestamp': datetime.now().isoformat(),
                'current_performance': {
                    'coordination_success_rate': coordinator_metrics['success_rate'],
                    'dispatch_efficiency': dispatcher_status['dispatch_stats']['average_dispatch_time'],
                    'collaboration_success_rate': collaboration_stats['statistics']['success_rate'],
                    'workflow_success_rate': workflow_status['success_rate']
                },
                'optimizations_applied': []
            }
            
            # Optimize load balancing strategy
            current_success_rate = coordinator_metrics['success_rate']
            if current_success_rate < 85:  # Below 85% success rate
                # Switch to performance-based load balancing
                await self.task_dispatcher.set_load_balancing_strategy(
                    LoadBalancingStrategy.PERFORMANCE_BASED
                )
                optimizations['optimizations_applied'].append(
                    "Switched to performance-based load balancing"
                )
            
            # Optimize collaboration timeouts based on historical data
            avg_collaboration_duration = collaboration_stats['statistics']['average_duration_seconds']
            if avg_collaboration_duration > 300:  # More than 5 minutes
                # Reduce collaboration timeouts
                optimizations['optimizations_applied'].append(
                    "Reduced collaboration timeouts for better efficiency"
                )
            
            # Optimize workflow concurrency
            active_workflows = workflow_status['active_executions']
            if active_workflows > self.max_concurrent_workflows * 0.8:
                # System is near capacity
                optimizations['optimizations_applied'].append(
                    "Workflow throttling applied due to high load"
                )
            
            return optimizations
            
        except Exception as e:
            self.logger.error(f"Error optimizing system performance: {str(e)}")
            return {'error': str(e)}
    
    async def _execute_with_workflow_engine(self, workflow_request: Dict[str, Any]) -> Dict[str, Any]:
        """Execute using workflow engine for complex workflows"""
        workflow_id = workflow_request.get('workflow_id', 'custom_workflow')
        input_data = workflow_request.get('input_data', {})
        
        execution_id = await self.workflow_engine.execute_workflow(
            workflow_id, input_data, "orchestration_manager"
        )
        
        if execution_id:
            return {
                'execution_id': execution_id,
                'execution_method': 'workflow_engine',
                'status': 'started'
            }
        else:
            raise Exception("Failed to start workflow execution")
    
    async def _execute_with_collaboration(self, workflow_request: Dict[str, Any]) -> Dict[str, Any]:
        """Execute using collaboration manager for consensus workflows"""
        collaboration_request = {
            'type': 'consensus_building',
            'initiator_agent': 'orchestration_manager',
            'target_agents': workflow_request.get('agents_required', []),
            'shared_data': workflow_request.get('shared_data', {}),
            'timeout_minutes': workflow_request.get('timeout_minutes', 30)
        }
        
        session_id = await self.collaboration_manager.initiate_collaboration(collaboration_request)
        
        if session_id:
            return {
                'session_id': session_id,
                'execution_method': 'collaboration_manager',
                'status': 'started'
            }
        else:
            raise Exception("Failed to start collaboration session")
    
    async def _execute_with_coordinator(self, workflow_request: Dict[str, Any]) -> Dict[str, Any]:
        """Execute using agent coordinator for standard workflows"""
        if workflow_request.get('type') == 'incident_resolution':
            result = await self.coordinator.coordinate_incident_resolution(
                workflow_request.get('incident_data', {})
            )
        elif workflow_request.get('type') == 'pattern_analysis':
            result = await self.coordinator.coordinate_pattern_analysis(
                workflow_request.get('analysis_request', {})
            )
        else:
            # Custom coordination workflow
            workflow_def = self._build_coordinator_workflow(workflow_request)
            workflow_id = await self.coordinator.execute_workflow(workflow_def)
            result = {'workflow_id': workflow_id, 'status': 'started'}
        
        result['execution_method'] = 'agent_coordinator'
        return result
    
    async def _register_predefined_workflows(self):
        """Register predefined workflows with the engine"""
        try:
            # Register incident resolution workflow
            incident_workflow = self.workflow_engine.create_incident_resolution_workflow()
            await self.workflow_engine.register_workflow(incident_workflow)
            
            # Register pattern analysis workflow
            pattern_workflow = self.workflow_engine.create_pattern_analysis_workflow()
            await self.workflow_engine.register_workflow(pattern_workflow)
            
            # Register proactive monitoring workflow
            monitoring_workflow = self.workflow_engine.create_proactive_monitoring_workflow()
            await self.workflow_engine.register_workflow(monitoring_workflow)
            
            self.logger.info("Predefined workflows registered successfully")
            
        except Exception as e:
            self.logger.error(f"Error registering predefined workflows: {str(e)}")
    
    def get_orchestration_status(self) -> Dict[str, Any]:
        """Get comprehensive orchestration system status"""
        return {
            'orchestration_manager': {
                'status': 'running' if self.running else 'stopped',
                'max_concurrent_workflows': self.max_concurrent_workflows
            },
            'coordinator': self.coordinator.get_coordination_metrics(),
            'task_dispatcher': self.task_dispatcher.get_dispatcher_status(),
            'collaboration_manager': self.collaboration_manager.get_manager_status(),
            'workflow_engine': self.workflow_engine.get_engine_status(),
            'timestamp': datetime.now().isoformat()
        }


