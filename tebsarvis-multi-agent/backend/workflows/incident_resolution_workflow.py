"""
Incident Resolution Workflow - Complete resolution pipeline using multiple agents
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from enum import Enum

from ..agents.core.base_agent import BaseAgent
from ..agents.core.agent_registry import get_global_registry
from ..agents.core.agent_communication import MessageBus, AgentCommunicator
from ..agents.core.message_types import TaskType, Priority, create_task_request
from ..agents.orchestrator.agent_coordinator import AgentCoordinator
from ..agents.orchestrator.collaboration_manager import CollaborationManager
from ..config.agent_config import get_agent_config, AgentType

logger = logging.getLogger(__name__)

class WorkflowStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class IncidentResolutionWorkflow:
    """
    Orchestrates the complete incident resolution pipeline:
    1. Context enrichment
    2. Similar incident search
    3. Resolution generation
    4. Response generation
    5. Quality validation
    """
    
    def __init__(self):
        self.registry = get_global_registry()
        self.message_bus = MessageBus()
        self.coordinator = AgentCoordinator(self.registry, self.message_bus)
        self.collaboration_manager = CollaborationManager(self.registry, self.message_bus)
        self.communicator = AgentCommunicator("incident_resolution_workflow", self.message_bus)
        self.config = get_agent_config()
        
        self.active_workflows: Dict[str, Dict[str, Any]] = {}
        self.workflow_history: List[Dict[str, Any]] = []
        
        self.logger = logging.getLogger("workflows.incident_resolution")
    
    async def start(self):
        """Start the workflow system"""
        await self.message_bus.start()
        await self.coordinator.start()
        await self.collaboration_manager.start()
        self.logger.info("Incident Resolution Workflow system started")
    
    async def stop(self):
        """Stop the workflow system"""
        await self.collaboration_manager.stop()
        await self.coordinator.stop()
        await self.message_bus.stop()
        self.logger.info("Incident Resolution Workflow system stopped")
    
    async def execute_incident_resolution(self, incident_data: Dict[str, Any], 
                                        workflow_options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute complete incident resolution workflow.
        
        Args:
            incident_data: Incident information to resolve
            workflow_options: Optional workflow configuration
            
        Returns:
            Complete resolution results with all agent outputs
        """
        try:
            workflow_id = f"incident_resolution_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            
            # Initialize workflow tracking
            workflow_state = {
                'workflow_id': workflow_id,
                'status': WorkflowStatus.PENDING,
                'incident_data': incident_data,
                'started_at': datetime.now(),
                'steps_completed': [],
                'results': {},
                'errors': [],
                'options': workflow_options or {}
            }
            
            self.active_workflows[workflow_id] = workflow_state
            
            self.logger.info(f"Starting incident resolution workflow {workflow_id}")
            workflow_state['status'] = WorkflowStatus.RUNNING
            
            # Step 1: Context Enrichment
            context_result = await self._enrich_incident_context(workflow_id, incident_data)
            workflow_state['results']['context_enrichment'] = context_result
            workflow_state['steps_completed'].append('context_enrichment')
            
            # Step 2: Similar Incident Search
            search_result = await self._search_similar_incidents(
                workflow_id, incident_data, context_result
            )
            workflow_state['results']['similar_search'] = search_result
            workflow_state['steps_completed'].append('similar_search')
            
            # Step 3: Resolution Generation
            resolution_result = await self._generate_resolution_recommendations(
                workflow_id, incident_data, context_result, search_result
            )
            workflow_state['results']['resolution_generation'] = resolution_result
            workflow_state['steps_completed'].append('resolution_generation')
            
            # Step 4: Response Generation
            response_result = await self._generate_user_response(
                workflow_id, incident_data, resolution_result
            )
            workflow_state['results']['response_generation'] = response_result
            workflow_state['steps_completed'].append('response_generation')
            
            # Step 5: Quality Validation (Optional)
            if workflow_options and workflow_options.get('enable_validation', True):
                validation_result = await self._validate_resolution_quality(
                    workflow_id, incident_data, resolution_result, response_result
                )
                workflow_state['results']['quality_validation'] = validation_result
                workflow_state['steps_completed'].append('quality_validation')
            
            # Step 6: Collaboration (if needed)
            if workflow_options and workflow_options.get('enable_collaboration', False):
                collaboration_result = await self._coordinate_agent_collaboration(
                    workflow_id, incident_data, workflow_state['results']
                )
                workflow_state['results']['collaboration'] = collaboration_result
                workflow_state['steps_completed'].append('collaboration')
            
            # Finalize workflow
            workflow_state['status'] = WorkflowStatus.COMPLETED
            workflow_state['completed_at'] = datetime.now()
            workflow_state['duration'] = (
                workflow_state['completed_at'] - workflow_state['started_at']
            ).total_seconds()
            
            # Generate final response
            final_result = await self._compile_final_result(workflow_state)
            
            # Move to history
            self.workflow_history.append(workflow_state)
            del self.active_workflows[workflow_id]
            
            self.logger.info(f"Incident resolution workflow {workflow_id} completed successfully")
            return final_result
            
        except Exception as e:
            self.logger.error(f"Error in incident resolution workflow: {str(e)}")
            
            # Update workflow state on error
            if workflow_id in self.active_workflows:
                workflow_state = self.active_workflows[workflow_id]
                workflow_state['status'] = WorkflowStatus.FAILED
                workflow_state['errors'].append(str(e))
                workflow_state['completed_at'] = datetime.now()
                
                # Move to history
                self.workflow_history.append(workflow_state)
                del self.active_workflows[workflow_id]
            
            raise
    
    async def _enrich_incident_context(self, workflow_id: str, 
                                     incident_data: Dict[str, Any]) -> Dict[str, Any]:
        """Step 1: Enrich incident context using Context Agent"""
        try:
            self.logger.info(f"[{workflow_id}] Step 1: Enriching incident context")
            
            # Find Context Agent
            context_agents = self.registry.find_agents_by_type('context')
            if not context_agents:
                raise RuntimeError("No Context Agent available")
            
            best_agent = self.registry.get_best_agent_for_capability('metadata_enrichment')
            if not best_agent:
                raise RuntimeError("No agent capable of metadata enrichment")
            
            # Prepare task data
            task_data = {
                'incident_data': incident_data,
                'enrichment_level': 'comprehensive',
                'context_types': ['technical', 'business', 'user', 'historical']
            }
            
            # Execute context enrichment
            response = await self.communicator.send_task_request(
                recipient_id=best_agent.agent_id,
                task_type='metadata_enrichment',
                task_data=task_data,
                timeout_seconds=180
            )
            
            if not response.success:
                raise RuntimeError(f"Context enrichment failed: {response.error_message}")
            
            self.logger.info(f"[{workflow_id}] Context enrichment completed")
            return response.result_data
            
        except Exception as e:
            self.logger.error(f"[{workflow_id}] Context enrichment failed: {str(e)}")
            raise
    
    async def _search_similar_incidents(self, workflow_id: str, 
                                      incident_data: Dict[str, Any],
                                      context_result: Dict[str, Any]) -> Dict[str, Any]:
        """Step 2: Search for similar incidents using Search Agent"""
        try:
            self.logger.info(f"[{workflow_id}] Step 2: Searching for similar incidents")
            
            # Find Search Agent
            search_agent = self.registry.get_best_agent_for_capability('semantic_search')
            if not search_agent:
                raise RuntimeError("No Search Agent available")
            
            # Prepare search query
            summary = incident_data.get('summary', '')
            description = incident_data.get('description', '')
            category = incident_data.get('category', '')
            
            # Enhanced query with context
            enriched_metadata = context_result.get('enriched_metadata', {})
            entities = enriched_metadata.get('entities', {})
            
            # Build comprehensive search query
            search_query = f"{summary} {description}"
            if entities.get('systems'):
                search_query += f" {' '.join(entities['systems'])}"
            if entities.get('error_types'):
                search_query += f" {' '.join(entities['error_types'])}"
            
            task_data = {
                'query': search_query,
                'max_results': 5,
                'filters': {
                    'category': category,
                    'has_resolution': True
                },
                'include_metadata': True,
                'boost_recent': True
            }
            
            # Execute semantic search
            response = await self.communicator.send_task_request(
                recipient_id=search_agent.agent_id,
                task_type='semantic_search',
                task_data=task_data,
                timeout_seconds=120
            )
            
            if not response.success:
                raise RuntimeError(f"Similar incident search failed: {response.error_message}")
            
            self.logger.info(f"[{workflow_id}] Similar incident search completed")
            return response.result_data
            
        except Exception as e:
            self.logger.error(f"[{workflow_id}] Similar incident search failed: {str(e)}")
            raise
    
    async def _generate_resolution_recommendations(self, workflow_id: str,
                                                 incident_data: Dict[str, Any],
                                                 context_result: Dict[str, Any],
                                                 search_result: Dict[str, Any]) -> Dict[str, Any]:
        """Step 3: Generate resolution recommendations using Resolution Agent"""
        try:
            self.logger.info(f"[{workflow_id}] Step 3: Generating resolution recommendations")
            
            # Find Resolution Agent
            resolution_agent = self.registry.get_best_agent_for_capability('incident_resolution')
            if not resolution_agent:
                raise RuntimeError("No Resolution Agent available")
            
            # Prepare task data with enriched context and similar incidents
            task_data = {
                'incident_data': incident_data,
                'enriched_context': context_result.get('enriched_metadata', {}),
                'similar_incidents': search_result.get('results', []),
                'resolution_options': {
                    'max_solutions': 3,
                    'include_implementation_details': True,
                    'include_validation_steps': True,
                    'consider_complexity': True
                }
            }
            
            # Execute resolution generation
            response = await self.communicator.send_task_request(
                recipient_id=resolution_agent.agent_id,
                task_type='incident_resolution',
                task_data=task_data,
                timeout_seconds=300
            )
            
            if not response.success:
                raise RuntimeError(f"Resolution generation failed: {response.error_message}")
            
            self.logger.info(f"[{workflow_id}] Resolution generation completed")
            return response.result_data
            
        except Exception as e:
            self.logger.error(f"[{workflow_id}] Resolution generation failed: {str(e)}")
            raise
    
    async def _generate_user_response(self, workflow_id: str,
                                    incident_data: Dict[str, Any],
                                    resolution_result: Dict[str, Any]) -> Dict[str, Any]:
        """Step 4: Generate user-friendly response using Conversation Agent"""
        try:
            self.logger.info(f"[{workflow_id}] Step 4: Generating user response")
            
            # Find Conversation Agent
            conversation_agent = self.registry.get_best_agent_for_capability('response_generation')
            if not conversation_agent:
                raise RuntimeError("No Conversation Agent available")
            
            # Prepare user query and knowledge context
            user_query = f"How to resolve: {incident_data.get('summary', '')}"
            
            task_data = {
                'user_query': user_query,
                'knowledge_sources': ['incidents', 'resolutions'],
                'resolution_data': resolution_result,
                'incident_context': incident_data,
                'response_style': 'helpful_and_detailed',
                'include_citations': True,
                'max_context_items': 5
            }
            
            # Execute response generation
            response = await self.communicator.send_task_request(
                recipient_id=conversation_agent.agent_id,
                task_type='response_generation',
                task_data=task_data,
                timeout_seconds=120
            )
            
            if not response.success:
                raise RuntimeError(f"Response generation failed: {response.error_message}")
            
            self.logger.info(f"[{workflow_id}] User response generation completed")
            return response.result_data
            
        except Exception as e:
            self.logger.error(f"[{workflow_id}] User response generation failed: {str(e)}")
            raise
    
    async def _validate_resolution_quality(self, workflow_id: str,
                                         incident_data: Dict[str, Any],
                                         resolution_result: Dict[str, Any],
                                         response_result: Dict[str, Any]) -> Dict[str, Any]:
        """Step 5: Validate resolution quality using Resolution Agent"""
        try:
            self.logger.info(f"[{workflow_id}] Step 5: Validating resolution quality")
            
            # Find Resolution Agent for validation
            resolution_agent = self.registry.get_best_agent_for_capability('resolution_validation')
            if not resolution_agent:
                self.logger.warning(f"[{workflow_id}] No validation capability available, skipping")
                return {'validation_skipped': True, 'reason': 'No validation agent available'}
            
            # Prepare validation data
            proposed_solution = {
                'solutions': resolution_result.get('solutions', []),
                'overall_confidence': resolution_result.get('overall_confidence', 0.0),
                'similar_incidents_used': resolution_result.get('similar_incidents_used', 0)
            }
            
            task_data = {
                'proposed_solution': proposed_solution,
                'incident_data': incident_data,
                'validation_criteria': {
                    'check_completeness': True,
                    'check_feasibility': True,
                    'check_safety': True,
                    'minimum_confidence': 0.7
                }
            }
            
            # Execute validation
            response = await self.communicator.send_task_request(
                recipient_id=resolution_agent.agent_id,
                task_type='resolution_validation',
                task_data=task_data,
                timeout_seconds=180
            )
            
            if not response.success:
                self.logger.warning(f"[{workflow_id}] Validation failed: {response.error_message}")
                return {'validation_failed': True, 'error': response.error_message}
            
            self.logger.info(f"[{workflow_id}] Resolution quality validation completed")
            return response.result_data
            
        except Exception as e:
            self.logger.error(f"[{workflow_id}] Resolution validation failed: {str(e)}")
            return {'validation_error': True, 'error': str(e)}
    
    async def _coordinate_agent_collaboration(self, workflow_id: str,
                                            incident_data: Dict[str, Any],
                                            workflow_results: Dict[str, Any]) -> Dict[str, Any]:
        """Step 6: Coordinate agent collaboration for consensus building"""
        try:
            self.logger.info(f"[{workflow_id}] Step 6: Initiating agent collaboration")
            
            # Prepare collaboration request
            collaboration_request = {
                'type': 'consensus_building',
                'initiator_agent': 'incident_resolution_workflow',
                'target_agents': self._get_collaboration_agents(),
                'shared_data': {
                    'incident_data': incident_data,
                    'workflow_results': workflow_results,
                    'collaboration_goal': 'resolution_consensus'
                },
                'timeout_minutes': 15,
                'metadata': {
                    'workflow_id': workflow_id,
                    'collaboration_type': 'incident_resolution_consensus'
                }
            }
            
            # Initiate collaboration
            session_id = await self.collaboration_manager.initiate_collaboration(collaboration_request)
            
            if not session_id:
                self.logger.warning(f"[{workflow_id}] Failed to initiate collaboration")
                return {'collaboration_failed': True, 'reason': 'Could not start collaboration session'}
            
            # Wait for collaboration to complete
            collaboration_result = await self._wait_for_collaboration_completion(session_id, timeout=900)
            
            self.logger.info(f"[{workflow_id}] Agent collaboration completed")
            return collaboration_result
            
        except Exception as e:
            self.logger.error(f"[{workflow_id}] Agent collaboration failed: {str(e)}")
            return {'collaboration_error': True, 'error': str(e)}
    
    def _get_collaboration_agents(self) -> List[str]:
        """Get list of agents for collaboration"""
        collaboration_agents = []
        
        # Get one agent of each type for collaboration
        agent_types = ['resolution', 'search', 'context']
        
        for agent_type in agent_types:
            agents = self.registry.find_agents_by_type(agent_type)
            if agents:
                collaboration_agents.append(agents[0].agent_id)
        
        return collaboration_agents
    
    async def _wait_for_collaboration_completion(self, session_id: str, timeout: int = 900) -> Dict[str, Any]:
        """Wait for collaboration session to complete"""
        start_time = datetime.now()
        
        while (datetime.now() - start_time).total_seconds() < timeout:
            status = await self.collaboration_manager.get_collaboration_status(session_id)
            
            if not status:
                raise RuntimeError(f"Collaboration session {session_id} not found")
            
            if status['status'] in ['completed', 'failed', 'cancelled']:
                return status
            
            await asyncio.sleep(5)  # Check every 5 seconds
        
        # Timeout reached
        await self.collaboration_manager.cancel_collaboration(session_id)
        raise TimeoutError(f"Collaboration session {session_id} timed out")
    
    async def _compile_final_result(self, workflow_state: Dict[str, Any]) -> Dict[str, Any]:
        """Compile final workflow result"""
        try:
            results = workflow_state['results']
            
            # Extract key information from each step
            context_metadata = results.get('context_enrichment', {}).get('enriched_metadata', {})
            similar_incidents = results.get('similar_search', {}).get('results', [])
            solutions = results.get('resolution_generation', {}).get('solutions', [])
            user_response = results.get('response_generation', {}).get('response', '')
            validation = results.get('quality_validation', {})
            collaboration = results.get('collaboration', {})
            
            # Determine overall confidence
            resolution_confidence = results.get('resolution_generation', {}).get('overall_confidence', 0.0)
            validation_confidence = validation.get('confidence_score', resolution_confidence)
            overall_confidence = min(validation_confidence, 1.0)
            
            # Determine quality score
            quality_score = self._calculate_quality_score(workflow_state)
            
            final_result = {
                'workflow_id': workflow_state['workflow_id'],
                'status': workflow_state['status'].value,
                'incident_id': workflow_state['incident_data'].get('id'),
                'execution_summary': {
                    'duration_seconds': workflow_state.get('duration', 0),
                    'steps_completed': workflow_state['steps_completed'],
                    'total_steps': len(workflow_state['steps_completed']),
                    'success': workflow_state['status'] == WorkflowStatus.COMPLETED
                },
                'resolution_output': {
                    'primary_response': user_response,
                    'solution_recommendations': solutions,
                    'confidence_score': overall_confidence,
                    'quality_score': quality_score,
                    'similar_incidents_count': len(similar_incidents),
                    'context_enriched': bool(context_metadata)
                },
                'metadata': {
                    'context_entities': context_metadata.get('entities', {}),
                    'similar_incidents_used': similar_incidents[:3] if similar_incidents else [],
                    'validation_results': validation,
                    'collaboration_results': collaboration,
                    'processing_agents': self._get_processing_agents_summary(workflow_state)
                },
                'recommendations': self._generate_workflow_recommendations(workflow_state),
                'timestamp': datetime.now().isoformat()
            }
            
            return final_result
            
        except Exception as e:
            self.logger.error(f"Error compiling final result: {str(e)}")
            raise
    
    def _calculate_quality_score(self, workflow_state: Dict[str, Any]) -> float:
        """Calculate overall quality score for the workflow execution"""
        score_components = []
        
        # Context enrichment quality
        context_result = workflow_state['results'].get('context_enrichment', {})
        if context_result:
            enrichment_confidence = context_result.get('enrichment_metadata', {}).get('confidence_score', 0.5)
            score_components.append(enrichment_confidence * 0.2)
        
        # Search quality
        search_result = workflow_state['results'].get('similar_search', {})
        if search_result and search_result.get('results'):
            avg_score = sum(r.get('score', 0) for r in search_result['results']) / len(search_result['results'])
            score_components.append(avg_score * 0.3)
        
        # Resolution quality
        resolution_result = workflow_state['results'].get('resolution_generation', {})
        if resolution_result:
            resolution_confidence = resolution_result.get('overall_confidence', 0.5)
            score_components.append(resolution_confidence * 0.4)
        
        # Validation quality
        validation_result = workflow_state['results'].get('quality_validation', {})
        if validation_result and not validation_result.get('validation_skipped'):
            validation_score = validation_result.get('validation_score', 0.7)
            score_components.append(validation_score * 0.1)
        
        return sum(score_components) if score_components else 0.5
    
    def _get_processing_agents_summary(self, workflow_state: Dict[str, Any]) -> Dict[str, str]:
        """Get summary of agents that processed this workflow"""
        return {
            'context_agent': 'context_agent',
            'search_agent': 'search_agent',
            'resolution_agent': 'resolution_agent',
            'conversation_agent': 'conversation_agent',
            'workflow_coordinator': 'incident_resolution_workflow'
        }
    
    def _generate_workflow_recommendations(self, workflow_state: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on workflow execution"""
        recommendations = []
        
        # Check execution time
        duration = workflow_state.get('duration', 0)
        if duration > 300:  # 5 minutes
            recommendations.append("Consider optimizing agent response times for faster resolution")
        
        # Check confidence scores
        resolution_confidence = workflow_state['results'].get('resolution_generation', {}).get('overall_confidence', 0)
        if resolution_confidence < 0.7:
            recommendations.append("Low confidence in resolution - consider manual review")
        
        # Check similar incidents usage
        similar_count = workflow_state['results'].get('similar_search', {}).get('total_count', 0)
        if similar_count == 0:
            recommendations.append("No similar incidents found - this may be a new issue type")
        
        # Check validation results
        validation = workflow_state['results'].get('quality_validation', {})
        if validation.get('validation_failed'):
            recommendations.append("Resolution validation failed - manual review recommended")
        
        return recommendations
    
    async def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a workflow"""
        # Check active workflows
        if workflow_id in self.active_workflows:
            workflow = self.active_workflows[workflow_id]
            return {
                'workflow_id': workflow_id,
                'status': workflow['status'].value,
                'steps_completed': workflow['steps_completed'],
                'current_step': self._get_current_step(workflow),
                'started_at': workflow['started_at'].isoformat(),
                'duration': (datetime.now() - workflow['started_at']).total_seconds()
            }
        
        # Check history
        for workflow in self.workflow_history:
            if workflow['workflow_id'] == workflow_id:
                return {
                    'workflow_id': workflow_id,
                    'status': workflow['status'].value,
                    'steps_completed': workflow['steps_completed'],
                    'started_at': workflow['started_at'].isoformat(),
                    'completed_at': workflow.get('completed_at', {}).isoformat() if workflow.get('completed_at') else None,
                    'duration': workflow.get('duration', 0)
                }
        
        return None
    
    def _get_current_step(self, workflow_state: Dict[str, Any]) -> str:
        """Determine current step based on completed steps"""
        completed = workflow_state['steps_completed']
        
        if 'context_enrichment' not in completed:
            return 'context_enrichment'
        elif 'similar_search' not in completed:
            return 'similar_search'
        elif 'resolution_generation' not in completed:
            return 'resolution_generation'
        elif 'response_generation' not in completed:
            return 'response_generation'
        elif 'quality_validation' not in completed:
            return 'quality_validation'
        elif 'collaboration' not in completed:
            return 'collaboration'
        else:
            return 'finalizing'
    
    def get_workflow_statistics(self) -> Dict[str, Any]:
        """Get workflow execution statistics"""
        total_workflows = len(self.workflow_history)
        if total_workflows == 0:
            return {'message': 'No workflows executed yet'}
        
        successful_workflows = len([w for w in self.workflow_history if w['status'] == WorkflowStatus.COMPLETED])
        failed_workflows = len([w for w in self.workflow_history if w['status'] == WorkflowStatus.FAILED])
        
        durations = [w.get('duration', 0) for w in self.workflow_history if w.get('duration')]
        avg_duration = sum(durations) / len(durations) if durations else 0
        
        return {
            'total_workflows': total_workflows,
            'successful_workflows': successful_workflows,
            'failed_workflows': failed_workflows,
            'success_rate': (successful_workflows / total_workflows) * 100,
            'average_duration_seconds': avg_duration,
            'active_workflows': len(self.active_workflows),
            'last_updated': datetime.now().isoformat()
        }