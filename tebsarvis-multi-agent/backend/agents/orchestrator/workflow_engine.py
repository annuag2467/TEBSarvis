"""
Workflow Engine for TEBSarvis Multi-Agent System
Manages complex multi-agent workflows with branching, conditions, and error handling.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Callable, Union
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, asdict, field
from abc import ABC, abstractmethod
import uuid
import json

from ..core.base_agent import BaseAgent, AgentCapability
from ..core.agent_registry import AgentRegistry
from ..core.agent_communication import MessageBus, AgentCommunicator
from ..core.message_types import TaskType, Priority

class WorkflowState(Enum):
    DRAFT = "draft"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"

class StepStatus(Enum):
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRYING = "retrying"

class TriggerType(Enum):
    MANUAL = "manual"
    SCHEDULED = "scheduled"
    EVENT = "event"
    CONDITION = "condition"
    WEBHOOK = "webhook"

@dataclass
class WorkflowContext:
    """Workflow execution context and shared data"""
    workflow_id: str
    variables: Dict[str, Any] = field(default_factory=dict)
    step_results: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_history: List[Dict[str, Any]] = field(default_factory=list)
    
    def set_variable(self, key: str, value: Any):
        """Set a workflow variable"""
        self.variables[key] = value
        self._log_action(f"Set variable: {key}")
    
    def get_variable(self, key: str, default: Any = None) -> Any:
        """Get a workflow variable"""
        return self.variables.get(key, default)
    
    def set_step_result(self, step_id: str, result: Any):
        """Set result from a workflow step"""
        self.step_results[step_id] = result
        self._log_action(f"Step result set: {step_id}")
    
    def get_step_result(self, step_id: str, default: Any = None) -> Any:
        """Get result from a workflow step"""
        return self.step_results.get(step_id, default)
    
    def _log_action(self, action: str):
        """Log an action in execution history"""
        self.execution_history.append({
            'timestamp': datetime.now().isoformat(),
            'action': action
        })

class WorkflowStep(ABC):
    """Abstract base class for workflow steps"""
    
    def __init__(self, step_id: str, name: str, description: str = ""):
        self.step_id = step_id
        self.name = name
        self.description = description
        self.status = StepStatus.PENDING
        self.dependencies: List[str] = []
        self.conditions: List[str] = []
        self.timeout_seconds: int = 300
        self.retry_count: int = 0
        self.max_retries: int = 2
        self.on_failure: str = "fail"  # fail, skip, retry
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.error_message: Optional[str] = None
        self.result: Optional[Any] = None
    
    @abstractmethod
    async def execute(self, context: WorkflowContext, 
                     registry: AgentRegistry, 
                     communicator: AgentCommunicator) -> bool:
        """Execute the workflow step"""
        pass
    
    def can_execute(self, context: WorkflowContext) -> bool:
        """Check if step can be executed based on dependencies and conditions"""
        # Check dependencies
        for dep in self.dependencies:
            if dep not in context.step_results:
                return False
        
        # Check conditions
        for condition in self.conditions:
            if not self._evaluate_condition(condition, context):
                return False
        
        return True
    
    def _evaluate_condition(self, condition: str, context: WorkflowContext) -> bool:
        """Evaluate a condition string"""
        try:
            # Simple condition evaluation
            # In production, use a proper expression evaluator
            if "==" in condition:
                left, right = condition.split("==")
                left_val = self._get_value(left.strip(), context)
                right_val = self._get_value(right.strip(), context)
                return left_val == right_val
            elif "!=" in condition:
                left, right = condition.split("!=")
                left_val = self._get_value(left.strip(), context)
                right_val = self._get_value(right.strip(), context)
                return left_val != right_val
            elif ">" in condition:
                left, right = condition.split(">")
                left_val = float(self._get_value(left.strip(), context))
                right_val = float(self._get_value(right.strip(), context))
                return left_val > right_val
            else:
                # Boolean condition
                return bool(self._get_value(condition, context))
        except Exception:
            return False
    
    def _get_value(self, expression: str, context: WorkflowContext) -> Any:
        """Get value from expression (variable, literal, or step result)"""
        expression = expression.strip()
        
        # Check for step result reference
        if expression.startswith("steps."):
            step_id = expression[6:]  # Remove "steps."
            return context.get_step_result(step_id)
        
        # Check for variable reference
        if expression.startswith("vars."):
            var_name = expression[5:]  # Remove "vars."
            return context.get_variable(var_name)
        
        # Try to parse as literal
        try:
            # Try number
            if "." in expression:
                return float(expression)
            else:
                return int(expression)
        except ValueError:
            # Try boolean
            if expression.lower() in ["true", "false"]:
                return expression.lower() == "true"
            # Return as string
            return expression.strip('"\'')

class AgentTaskStep(WorkflowStep):
    """Workflow step that executes a task on an agent"""
    
    def __init__(self, step_id: str, name: str, agent_type: str, 
                 task_type: str, task_data: Dict[str, Any], **kwargs):
        super().__init__(step_id, name, kwargs.get('description', ''))
        self.agent_type = agent_type
        self.task_type = task_type
        self.task_data = task_data
        self.output_mapping: Dict[str, str] = kwargs.get('output_mapping', {})
    
    async def execute(self, context: WorkflowContext, 
                     registry: AgentRegistry, 
                     communicator: AgentCommunicator) -> bool:
        """Execute task on agent"""
        try:
            self.status = StepStatus.RUNNING
            self.started_at = datetime.now()
            
            # Find suitable agent
            capable_agents = registry.find_agents_by_type(self.agent_type)
            if not capable_agents:
                self.error_message = f"No agents of type {self.agent_type} available"
                self.status = StepStatus.FAILED
                return False
            
            # Get best agent
            best_agent = registry.get_best_agent_for_capability(self.task_type)
            if not best_agent:
                self.error_message = f"No agent capable of {self.task_type}"
                self.status = StepStatus.FAILED
                return False
            
            # Prepare task data with context variables
            resolved_task_data = self._resolve_task_data(self.task_data, context)
            
            # Send task to agent
            response = await communicator.send_task_request(
                recipient_id=best_agent.agent_id,
                task_type=self.task_type,
                task_data=resolved_task_data,
                timeout_seconds=self.timeout_seconds
            )
            
            if response.success:
                self.result = response.result_data
                self.status = StepStatus.COMPLETED
                self.completed_at = datetime.now()
                
                # Store result in context
                context.set_step_result(self.step_id, self.result)
                
                # Apply output mapping
                self._apply_output_mapping(context)
                
                return True
            else:
                self.error_message = response.error_message
                self.status = StepStatus.FAILED
                return False
                
        except Exception as e:
            self.error_message = str(e)
            self.status = StepStatus.FAILED
            return False
    
    def _resolve_task_data(self, task_data: Dict[str, Any], context: WorkflowContext) -> Dict[str, Any]:
        """Resolve variables in task data"""
        resolved_data = {}
        
        for key, value in task_data.items():
            if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                # Variable reference
                var_ref = value[2:-1]  # Remove ${ and }
                resolved_data[key] = self._get_value(var_ref, context)
            elif isinstance(value, dict):
                resolved_data[key] = self._resolve_task_data(value, context)
            else:
                resolved_data[key] = value
        
        return resolved_data
    
    def _apply_output_mapping(self, context: WorkflowContext):
        """Apply output mapping to store results in variables"""
        if not self.result or not self.output_mapping:
            return
        
        for source_path, target_var in self.output_mapping.items():
            value = self._extract_value_by_path(self.result, source_path)
            if value is not None:
                context.set_variable(target_var, value)
    
    def _extract_value_by_path(self, data: Any, path: str) -> Any:
        """Extract value from nested data using dot notation"""
        parts = path.split('.')
        current = data
        
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            elif isinstance(current, list) and part.isdigit():
                index = int(part)
                if 0 <= index < len(current):
                    current = current[index]
                else:
                    return None
            else:
                return None
        
        return current

class DecisionStep(WorkflowStep):
    """Workflow step that makes decisions based on conditions"""
    
    def __init__(self, step_id: str, name: str, conditions: Dict[str, str], 
                 default_path: str = "continue", **kwargs):
        super().__init__(step_id, name, kwargs.get('description', ''))
        self.decision_conditions = conditions  # condition -> next_step_id
        self.default_path = default_path
    
    async def execute(self, context: WorkflowContext, 
                     registry: AgentRegistry, 
                     communicator: AgentCommunicator) -> bool:
        """Execute decision logic"""
        try:
            self.status = StepStatus.RUNNING
            self.started_at = datetime.now()
            
            # Evaluate conditions
            for condition, next_step in self.decision_conditions.items():
                if self._evaluate_condition(condition, context):
                    self.result = {'decision': next_step, 'condition': condition}
                    context.set_variable(f"{self.step_id}_decision", next_step)
                    break
            else:
                # No condition matched, use default
                self.result = {'decision': self.default_path, 'condition': 'default'}
                context.set_variable(f"{self.step_id}_decision", self.default_path)
            
            self.status = StepStatus.COMPLETED
            self.completed_at = datetime.now()
            context.set_step_result(self.step_id, self.result)
            
            return True
            
        except Exception as e:
            self.error_message = str(e)
            self.status = StepStatus.FAILED
            return False

class ParallelStep(WorkflowStep):
    """Workflow step that executes multiple sub-steps in parallel"""
    
    def __init__(self, step_id: str, name: str, sub_steps: List[WorkflowStep], 
                 wait_for_all: bool = True, **kwargs):
        super().__init__(step_id, name, kwargs.get('description', ''))
        self.sub_steps = sub_steps
        self.wait_for_all = wait_for_all
    
    async def execute(self, context: WorkflowContext, 
                     registry: AgentRegistry, 
                     communicator: AgentCommunicator) -> bool:
        """Execute sub-steps in parallel"""
        try:
            self.status = StepStatus.RUNNING
            self.started_at = datetime.now()
            
            # Execute all sub-steps in parallel
            tasks = []
            for sub_step in self.sub_steps:
                task = asyncio.create_task(
                    sub_step.execute(context, registry, communicator)
                )
                tasks.append((sub_step, task))
            
            # Wait for completion
            results = {}
            if self.wait_for_all:
                # Wait for all to complete
                for sub_step, task in tasks:
                    try:
                        success = await task
                        results[sub_step.step_id] = {
                            'success': success,
                            'result': sub_step.result,
                            'error': sub_step.error_message
                        }
                    except Exception as e:
                        results[sub_step.step_id] = {
                            'success': False,
                            'result': None,
                            'error': str(e)
                        }
            else:
                # Wait for first completion
                done, pending = await asyncio.wait(
                    [task for _, task in tasks],
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                # Cancel pending tasks
                for task in pending:
                    task.cancel()
                
                # Get results from completed tasks
                for sub_step, task in tasks:
                    if task in done:
                        try:
                            success = await task
                            results[sub_step.step_id] = {
                                'success': success,
                                'result': sub_step.result,
                                'error': sub_step.error_message
                            }
                        except Exception as e:
                            results[sub_step.step_id] = {
                                'success': False,
                                'result': None,
                                'error': str(e)
                            }
            
            self.result = results
            context.set_step_result(self.step_id, self.result)
            
            # Determine overall success
            if self.wait_for_all:
                success = all(r['success'] for r in results.values())
            else:
                success = any(r['success'] for r in results.values())
            
            self.status = StepStatus.COMPLETED if success else StepStatus.FAILED
            self.completed_at = datetime.now()
            
            return success
            
        except Exception as e:
            self.error_message = str(e)
            self.status = StepStatus.FAILED
            return False

class DelayStep(WorkflowStep):
    """Workflow step that introduces a delay"""
    
    def __init__(self, step_id: str, name: str, delay_seconds: int, **kwargs):
        super().__init__(step_id, name, kwargs.get('description', ''))
        self.delay_seconds = delay_seconds
    
    async def execute(self, context: WorkflowContext, 
                     registry: AgentRegistry, 
                     communicator: AgentCommunicator) -> bool:
        """Execute delay"""
        try:
            self.status = StepStatus.RUNNING
            self.started_at = datetime.now()
            
            await asyncio.sleep(self.delay_seconds)
            
            self.result = {'delayed_seconds': self.delay_seconds}
            self.status = StepStatus.COMPLETED
            self.completed_at = datetime.now()
            context.set_step_result(self.step_id, self.result)
            
            return True
            
        except Exception as e:
            self.error_message = str(e)
            self.status = StepStatus.FAILED
            return False

@dataclass
class WorkflowDefinition:
    """Complete workflow definition"""
    workflow_id: str
    name: str
    description: str
    version: str
    steps: List[WorkflowStep]
    triggers: List[Dict[str, Any]]
    variables: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timeout_minutes: int = 60
    max_retries: int = 3
    created_at: datetime = field(default_factory=datetime.now)
    created_by: str = "system"

@dataclass
class WorkflowExecution:
    """Workflow execution instance"""
    execution_id: str
    workflow_id: str
    workflow_definition: WorkflowDefinition
    context: WorkflowContext
    state: WorkflowState
    current_step: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    triggered_by: str = "manual"
    error_message: Optional[str] = None

class WorkflowEngine:
    """
    Workflow engine that manages and executes complex multi-agent workflows.
    """
    
    def __init__(self, registry: AgentRegistry, message_bus: MessageBus):
        self.registry = registry
        self.message_bus = message_bus
        self.communicator = AgentCommunicator("workflow_engine", message_bus)
        
        # Workflow storage
        self.workflow_definitions: Dict[str, WorkflowDefinition] = {}
        self.active_executions: Dict[str, WorkflowExecution] = {}
        self.execution_history: List[WorkflowExecution] = []
        
        # Engine configuration
        self.max_concurrent_executions = 50
        self.execution_cleanup_hours = 24
        
        self.logger = logging.getLogger("workflow_engine")
        self.running = False
        self.background_tasks = []
    
    async def start(self):
        """Start the workflow engine"""
        self.running = True
        
        # Start background tasks
        self.background_tasks = [
            asyncio.create_task(self._monitor_executions()),
            asyncio.create_task(self._cleanup_old_executions()),
            asyncio.create_task(self._process_scheduled_workflows())
        ]
        
        self.logger.info("Workflow Engine started")
    
    async def stop(self):
        """Stop the workflow engine gracefully"""
        self.running = False
        
        # Cancel all active executions
        for execution in list(self.active_executions.values()):
            await self.cancel_execution(execution.execution_id)
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        if self.background_tasks:
            await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        self.logger.info("Workflow Engine stopped")
    
    async def register_workflow(self, workflow_definition: WorkflowDefinition) -> bool:
        """
        Register a new workflow definition.
        
        Args:
            workflow_definition: Workflow to register
            
        Returns:
            True if registration successful
        """
        try:
            # Validate workflow
            validation_result = self._validate_workflow(workflow_definition)
            if not validation_result['valid']:
                self.logger.error(f"Invalid workflow: {validation_result['errors']}")
                return False
            
            self.workflow_definitions[workflow_definition.workflow_id] = workflow_definition
            self.logger.info(f"Workflow registered: {workflow_definition.workflow_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error registering workflow: {str(e)}")
            return False
    
    async def execute_workflow(self, workflow_id: str, 
                             input_data: Optional[Dict[str, Any]] = None,
                             triggered_by: str = "manual") -> Optional[str]:
        """
        Start execution of a workflow.
        
        Args:
            workflow_id: ID of workflow to execute
            input_data: Input data for workflow
            triggered_by: Who/what triggered the execution
            
        Returns:
            Execution ID if started successfully
        """
        try:
            # Check if workflow exists
            if workflow_id not in self.workflow_definitions:
                self.logger.error(f"Workflow not found: {workflow_id}")
                return None
            
            # Check capacity
            if len(self.active_executions) >= self.max_concurrent_executions:
                self.logger.warning("Maximum concurrent executions reached")
                return None
            
            workflow_def = self.workflow_definitions[workflow_id]
            execution_id = str(uuid.uuid4())
            
            # Create execution context
            context = WorkflowContext(
                workflow_id=workflow_id,
                variables=workflow_def.variables.copy(),
                metadata={'triggered_by': triggered_by}
            )
            
            # Add input data to context
            if input_data:
                context.variables.update(input_data)
            
            # Create execution instance
            execution = WorkflowExecution(
                execution_id=execution_id,
                workflow_id=workflow_id,
                workflow_definition=workflow_def,
                context=context,
                state=WorkflowState.READY,
                triggered_by=triggered_by
            )
            
            self.active_executions[execution_id] = execution
            
            # Start execution
            asyncio.create_task(self._execute_workflow_async(execution))
            
            self.logger.info(f"Workflow execution started: {execution_id}")
            return execution_id
            
        except Exception as e:
            self.logger.error(f"Error starting workflow execution: {str(e)}")
            return None
    
    async def get_execution_status(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Get status of workflow execution"""
        try:
            # Check active executions
            if execution_id in self.active_executions:
                execution = self.active_executions[execution_id]
                return self._serialize_execution_status(execution)
            
            # Check history
            for execution in self.execution_history:
                if execution.execution_id == execution_id:
                    return self._serialize_execution_status(execution)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting execution status: {str(e)}")
            return None
    
    async def cancel_execution(self, execution_id: str) -> bool:
        """Cancel an active workflow execution"""
        try:
            if execution_id in self.active_executions:
                execution = self.active_executions[execution_id]
                execution.state = WorkflowState.CANCELLED
                self.logger.info(f"Workflow execution cancelled: {execution_id}")
                return True
            return False
            
        except Exception as e:
            self.logger.error(f"Error cancelling execution: {str(e)}")
            return False
    
    async def pause_execution(self, execution_id: str) -> bool:
        """Pause an active workflow execution"""
        try:
            if execution_id in self.active_executions:
                execution = self.active_executions[execution_id]
                if execution.state == WorkflowState.RUNNING:
                    execution.state = WorkflowState.PAUSED
                    self.logger.info(f"Workflow execution paused: {execution_id}")
                    return True
            return False
            
        except Exception as e:
            self.logger.error(f"Error pausing execution: {str(e)}")
            return False
    
    async def resume_execution(self, execution_id: str) -> bool:
        """Resume a paused workflow execution"""
        try:
            if execution_id in self.active_executions:
                execution = self.active_executions[execution_id]
                if execution.state == WorkflowState.PAUSED:
                    execution.state = WorkflowState.RUNNING
                    self.logger.info(f"Workflow execution resumed: {execution_id}")
                    return True
            return False
            
        except Exception as e:
            self.logger.error(f"Error resuming execution: {str(e)}")
            return False
    
    async def _execute_workflow_async(self, execution: WorkflowExecution):
        """Execute workflow asynchronously"""
        try:
            execution.state = WorkflowState.RUNNING
            execution.started_at = datetime.now()
            
            workflow_def = execution.workflow_definition
            timeout_deadline = datetime.now() + timedelta(minutes=workflow_def.timeout_minutes)
            
            # Execute steps in order
            for step in workflow_def.steps:
                # Check for cancellation/pause
                if execution.state == WorkflowState.CANCELLED:
                    break
                
                while execution.state == WorkflowState.PAUSED:
                    await asyncio.sleep(1)
                
                # Check timeout
                if datetime.now() > timeout_deadline:
                    execution.state = WorkflowState.TIMEOUT
                    execution.error_message = "Workflow execution timed out"
                    break
                
                # Check if step can be executed
                if not step.can_execute(execution.context):
                    continue
                
                execution.current_step = step.step_id
                
                # Execute step with retries
                success = await self._execute_step_with_retries(
                    step, execution.context, step.max_retries
                )
                
                if not success:
                    if step.on_failure == "fail":
                        execution.state = WorkflowState.FAILED
                        execution.error_message = f"Step {step.step_id} failed: {step.error_message}"
                        break
                    elif step.on_failure == "skip":
                        step.status = StepStatus.SKIPPED
                        continue
                    # "retry" is handled in _execute_step_with_retries
            
            # Determine final state
            if execution.state == WorkflowState.RUNNING:
                execution.state = WorkflowState.COMPLETED
            
            execution.completed_at = datetime.now()
            execution.current_step = None
            
            # Move to history
            self.execution_history.append(execution)
            del self.active_executions[execution.execution_id]
            
            self.logger.info(f"Workflow execution completed: {execution.execution_id} ({execution.state.value})")
            
        except Exception as e:
            execution.state = WorkflowState.FAILED
            execution.error_message = str(e)
            execution.completed_at = datetime.now()
            self.logger.error(f"Workflow execution error: {str(e)}")
    
    async def _execute_step_with_retries(self, step: WorkflowStep, 
                                       context: WorkflowContext, 
                                       max_retries: int) -> bool:
        """Execute step with retry logic"""
        for attempt in range(max_retries + 1):
            try:
                success = await step.execute(context, self.registry, self.communicator)
                if success:
                    return True
                
                if attempt < max_retries:
                    step.status = StepStatus.RETRYING
                    step.retry_count += 1
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                
            except Exception as e:
                step.error_message = str(e)
                if attempt < max_retries:
                    step.status = StepStatus.RETRYING
                    step.retry_count += 1
                    await asyncio.sleep(2 ** attempt)
                else:
                    step.status = StepStatus.FAILED
        
        return False
    
    def _validate_workflow(self, workflow_def: WorkflowDefinition) -> Dict[str, Any]:
        """Validate workflow definition"""
        errors = []
        warnings = []
        
        # Check for duplicate step IDs
        step_ids = [step.step_id for step in workflow_def.steps]
        if len(step_ids) != len(set(step_ids)):
            errors.append("Duplicate step IDs found")
        
        # Validate step dependencies
        for step in workflow_def.steps:
            for dep in step.dependencies:
                if dep not in step_ids:
                    errors.append(f"Step {step.step_id} has invalid dependency: {dep}")
        
        # Check for circular dependencies
        if self._has_circular_dependencies(workflow_def.steps):
            errors.append("Circular dependencies detected")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }
    
    def _has_circular_dependencies(self, steps: List[WorkflowStep]) -> bool:
        """Check for circular dependencies in workflow steps"""
        # Build dependency graph
        graph = {step.step_id: step.dependencies for step in steps}
        
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
        
        for step_id in graph:
            if dfs(step_id):
                return True
        
        return False
    
    async def _monitor_executions(self):
        """Monitor active workflow executions"""
        while self.running:
            try:
                current_time = datetime.now()
                
                for execution in list(self.active_executions.values()):
                    # Check for timeout
                    if execution.started_at:
                        timeout_minutes = execution.workflow_definition.timeout_minutes
                        if (current_time - execution.started_at).total_seconds() > timeout_minutes * 60:
                            execution.state = WorkflowState.TIMEOUT
                            execution.error_message = "Workflow execution timed out"
                            execution.completed_at = current_time
                            
                            # Move to history
                            self.execution_history.append(execution)
                            del self.active_executions[execution.execution_id]
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error monitoring executions: {str(e)}")
                await asyncio.sleep(60)
    
    async def _cleanup_old_executions(self):
        """Clean up old execution history"""
        while self.running:
            try:
                cutoff_time = datetime.now() - timedelta(hours=self.execution_cleanup_hours)
                
                # Remove old executions from history
                self.execution_history = [
                    execution for execution in self.execution_history
                    if execution.completed_at and execution.completed_at > cutoff_time
                ]
                
                await asyncio.sleep(3600)  # Run every hour
                
            except Exception as e:
                self.logger.error(f"Error cleaning up executions: {str(e)}")
                await asyncio.sleep(300)
    
    async def _process_scheduled_workflows(self):
        """Process scheduled workflow triggers"""
        while self.running:
            try:
                current_time = datetime.now()
                
                for workflow_def in self.workflow_definitions.values():
                    for trigger in workflow_def.triggers:
                        if trigger.get('type') == TriggerType.SCHEDULED.value:
                            # Simple scheduled trigger processing
                            # In production, use a proper scheduler like APScheduler
                            
                            schedule = trigger.get('schedule', {})
                            interval_minutes = schedule.get('interval_minutes', 60)
                            
                            # Check if it's time to trigger
                            last_run = trigger.get('last_run')
                            if not last_run:
                                trigger['last_run'] = current_time
                                continue
                            
                            if isinstance(last_run, str):
                                last_run = datetime.fromisoformat(last_run)
                            
                            if (current_time - last_run).total_seconds() >= interval_minutes * 60:
                                # Trigger workflow
                                await self.execute_workflow(
                                    workflow_def.workflow_id,
                                    input_data=trigger.get('input_data', {}),
                                    triggered_by="scheduler"
                                )
                                trigger['last_run'] = current_time
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Error processing scheduled workflows: {str(e)}")
                await asyncio.sleep(300)
    
    def _serialize_execution_status(self, execution: WorkflowExecution) -> Dict[str, Any]:
        """Serialize execution status for API response"""
        step_statuses = []
        for step in execution.workflow_definition.steps:
            step_status = {
                'step_id': step.step_id,
                'name': step.name,
                'status': step.status.value,
                'started_at': step.started_at.isoformat() if step.started_at else None,
                'completed_at': step.completed_at.isoformat() if step.completed_at else None,
                'error_message': step.error_message,
                'retry_count': step.retry_count
            }
            step_statuses.append(step_status)
        
        return {
            'execution_id': execution.execution_id,
            'workflow_id': execution.workflow_id,
            'workflow_name': execution.workflow_definition.name,
            'state': execution.state.value,
            'current_step': execution.current_step,
            'started_at': execution.started_at.isoformat() if execution.started_at else None,
            'completed_at': execution.completed_at.isoformat() if execution.completed_at else None,
            'triggered_by': execution.triggered_by,
            'error_message': execution.error_message,
            'steps': step_statuses,
            'context_variables': execution.context.variables,
            'step_results': execution.context.step_results,
            'execution_history': execution.context.execution_history
        }
    
    def get_engine_status(self) -> Dict[str, Any]:
        """Get workflow engine status"""
        active_count = len(self.active_executions)
        history_count = len(self.execution_history)
        
        # Calculate success rate
        completed_executions = [
            e for e in self.execution_history 
            if e.state in [WorkflowState.COMPLETED, WorkflowState.FAILED]
        ]
        
        if completed_executions:
            successful = len([e for e in completed_executions if e.state == WorkflowState.COMPLETED])
            success_rate = (successful / len(completed_executions)) * 100
        else:
            success_rate = 0.0
        
        return {
            'status': 'running' if self.running else 'stopped',
            'registered_workflows': len(self.workflow_definitions),
            'active_executions': active_count,
            'execution_history_count': history_count,
            'success_rate': success_rate,
            'max_concurrent_executions': self.max_concurrent_executions,
            'timestamp': datetime.now().isoformat()
        }
    
    def create_incident_resolution_workflow(self) -> WorkflowDefinition:
        """Create a predefined incident resolution workflow"""
        steps = [
            AgentTaskStep(
                step_id="enrich_context",
                name="Enrich Incident Context",
                agent_type="context",
                task_type="metadata_enrichment",
                task_data={
                    "incident_data": "${vars.incident_data}",
                    "enrichment_level": "comprehensive"
                },
                output_mapping={
                    "enriched_metadata": "enriched_context"
                }
            ),
            AgentTaskStep(
                step_id="search_similar",
                name="Search Similar Incidents",
                agent_type="search",
                task_type="semantic_search",
                task_data={
                    "query": "${vars.incident_summary}",
                    "max_results": 5,
                    "filters": {
                        "category": "${vars.incident_category}"
                    }
                },
                output_mapping={
                    "results": "similar_incidents"
                }
            ),
            AgentTaskStep(
                step_id="generate_resolution",
                name="Generate Resolution Recommendations",
                agent_type="resolution",
                task_type="incident_resolution",
                task_data={
                    "incident_data": "${vars.incident_data}",
                    "similar_incidents": "${vars.similar_incidents}"
                },
                dependencies=["enrich_context", "search_similar"],
                output_mapping={
                    "solutions": "resolution_options"
                }
            ),
            DecisionStep(
                step_id="check_confidence",
                name="Check Resolution Confidence",
                conditions={
                    "steps.generate_resolution.overall_confidence > 0.8": "high_confidence",
                    "steps.generate_resolution.overall_confidence > 0.5": "medium_confidence"
                },
                default_path="low_confidence"
            ),
            AgentTaskStep(
                step_id="generate_response",
                name="Generate User Response",
                agent_type="conversation",
                task_type="response_generation",
                task_data={
                    "user_query": "How to resolve: ${vars.incident_summary}",
                    "knowledge_sources": ["incidents", "resolutions"],
                    "resolution_data": "${vars.resolution_options}"
                },
                dependencies=["generate_resolution", "check_confidence"],
                output_mapping={
                    "response": "final_response"
                }
            )
        ]
        
        workflow_def = WorkflowDefinition(
            workflow_id="incident_resolution_v1",
            name="Incident Resolution Workflow",
            description="Complete incident resolution using multi-agent collaboration",
            version="1.0.0",
            steps=steps,
            triggers=[],
            variables={
                "incident_data": {},
                "incident_summary": "",
                "incident_category": ""
            },
            timeout_minutes=30
        )
        
        return workflow_def
    
    def create_pattern_analysis_workflow(self) -> WorkflowDefinition:
        """Create a predefined pattern analysis workflow"""
        clustering_step = AgentTaskStep(
            step_id="incident_clustering",
            name="Cluster Incidents",
            agent_type="pattern_detection",
            task_type="incident_clustering",
            task_data={
                "time_range": "${vars.analysis_time_range}",
                "clustering_method": "mixed"
            },
            output_mapping={
                "clusters": "incident_clusters"
            }
        )
        
        trend_step = AgentTaskStep(
            step_id="trend_analysis",
            name="Analyze Trends",
            agent_type="pattern_detection",
            task_type="trend_analysis",
            task_data={
                "analysis_period": "${vars.analysis_time_range}",
                "trend_types": ["volume", "category", "severity"]
            },
            output_mapping={
                "trends": "trend_data"
            }
        )
        
        anomaly_step = AgentTaskStep(
            step_id="anomaly_detection",
            name="Detect Anomalies",
            agent_type="pattern_detection",
            task_type="anomaly_detection",
            task_data={
                "detection_window": {"days": 7},
                "baseline_period": "${vars.analysis_time_range}",
                "sensitivity": "medium"
            },
            output_mapping={
                "anomalies": "detected_anomalies"
            }
        )
        
        parallel_analysis = ParallelStep(
            step_id="parallel_analysis",
            name="Parallel Pattern Analysis",
            sub_steps=[clustering_step, trend_step, anomaly_step],
            wait_for_all=True
        )
        
        alert_evaluation = AgentTaskStep(
            step_id="evaluate_alerts",
            name="Evaluate Alert Conditions",
            agent_type="alerting",
            task_type="evaluate_conditions",
            task_data={
                "conditions": "${vars.alert_conditions}",
                "data": {
                    "clusters": "${vars.incident_clusters}",
                    "trends": "${vars.trend_data}",
                    "anomalies": "${vars.detected_anomalies}"
                }
            },
            dependencies=["parallel_analysis"],
            output_mapping={
                "alerts": "generated_alerts"
            }
        )
        
        workflow_def = WorkflowDefinition(
            workflow_id="pattern_analysis_v1",
            name="Pattern Analysis Workflow",
            description="Comprehensive pattern analysis with clustering, trends, and anomaly detection",
            version="1.0.0",
            steps=[parallel_analysis, alert_evaluation],
            triggers=[],
            variables={
                "analysis_time_range": {"days": 30},
                "alert_conditions": {}
            },
            timeout_minutes=45
        )
        
        return workflow_def
    
    def create_proactive_monitoring_workflow(self) -> WorkflowDefinition:
        """Create a predefined proactive monitoring workflow"""
        steps = [
            AgentTaskStep(
                step_id="real_time_monitoring",
                name="Real-time Monitoring",
                agent_type="alerting",
                task_type="real_time_monitoring",
                task_data={
                    "monitoring_window": {"minutes": 15},
                    "rule_ids": "${vars.monitoring_rules}"
                },
                output_mapping={
                    "generated_alerts": "monitoring_alerts"
                }
            ),
            DecisionStep(
                step_id="check_alerts",
                name="Check for Alerts",
                conditions={
                    "vars.monitoring_alerts != null and len(vars.monitoring_alerts) > 0": "alerts_found"
                },
                default_path="no_alerts"
            ),
            AgentTaskStep(
                step_id="dispatch_notifications",
                name="Dispatch Alert Notifications",
                agent_type="alerting",
                task_type="notification_dispatch",
                task_data={
                    "alerts": "${vars.monitoring_alerts}",
                    "notification_channels": "${vars.notification_channels}"
                },
                dependencies=["check_alerts"],
                conditions=["steps.check_alerts.decision == 'alerts_found'"],
                output_mapping={
                    "notifications_sent": "notification_results"
                }
            ),
            DelayStep(
                step_id="monitoring_interval",
                name="Wait for Next Monitoring Cycle",
                delay_seconds=300  # 5 minutes
            )
        ]
        
        workflow_def = WorkflowDefinition(
            workflow_id="proactive_monitoring_v1",
            name="Proactive Monitoring Workflow",
            description="Continuous monitoring with alert generation and notification",
            version="1.0.0",
            steps=steps,
            triggers=[
                {
                    "type": TriggerType.SCHEDULED.value,
                    "schedule": {"interval_minutes": 5},
                    "input_data": {
                        "monitoring_rules": ["volume_spike", "pattern_anomaly"],
                        "notification_channels": ["email", "teams"]
                    }
                }
            ],
            variables={
                "monitoring_rules": [],
                "notification_channels": ["email"]
            },
            timeout_minutes=10
        )
        
        return workflow_def