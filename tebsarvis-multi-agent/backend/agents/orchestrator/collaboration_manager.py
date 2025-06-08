"""
Collaboration Manager for TEBSarvis Multi-Agent System
Manages agent collaboration protocols, shared context, and collective intelligence.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Set, Tuple, Callable
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, asdict, field
from collections import defaultdict, Counter
import json
import uuid

from ..core.base_agent import BaseAgent, AgentCapability
from ..core.agent_registry import AgentRegistry
from ..core.agent_communication import MessageBus, AgentCommunicator
from ..core.message_types import create_collaboration_request, Priority
from ...config.agent_config import get_agent_config
# For TaskDispatcher reference (add to files that use it)
from .task_dispatcher import TaskDispatcher, LoadBalancingStrategy

# For WorkflowEngine reference (add to files that use it)  
from .workflow_engine import WorkflowEngine, WorkflowDefinition

# For CollaborationManager reference (add to files that use it)
# from .collaboration_manager import CollaborationManager

# For OrchestrationManager reference (add to files that use it)
from .orchestration_manager import OrchestrationManager

class CollaborationType(Enum):
    CONSENSUS_BUILDING = "consensus_building"
    KNOWLEDGE_SYNTHESIS = "knowledge_synthesis"
    PARALLEL_PROCESSING = "parallel_processing"
    SEQUENTIAL_HANDOFF = "sequential_handoff"
    EXPERT_CONSULTATION = "expert_consultation"
    PEER_REVIEW = "peer_review"

class CollaborationStatus(Enum):
    PENDING = "pending"
    ACTIVE = "active"
    COLLECTING_RESPONSES = "collecting_responses"
    SYNTHESIZING = "synthesizing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"

@dataclass
class SharedContext:
    """Shared context for collaboration sessions"""
    context_id: str
    session_id: str
    data: Dict[str, Any]
    version: int
    last_updated: datetime
    updated_by: str
    access_permissions: Set[str]
    
    def update_data(self, updates: Dict[str, Any], updated_by: str):
        """Update shared data with version control"""
        self.data.update(updates)
        self.version += 1
        self.last_updated = datetime.now()
        self.updated_by = updated_by

@dataclass
class CollaborationSession:
    """Represents an active collaboration session between agents"""
    session_id: str
    collaboration_type: CollaborationType
    initiator_agent: str
    participating_agents: Set[str]
    shared_context: SharedContext
    status: CollaborationStatus
    created_at: datetime
    timeout_minutes: int
    required_responses: int
    collected_responses: Dict[str, Any]
    final_result: Optional[Dict[str, Any]]
    metadata: Dict[str, Any]
    
    def add_response(self, agent_id: str, response: Dict[str, Any]):
        """Add agent response to collaboration"""
        self.collected_responses[agent_id] = {
            'response': response,
            'timestamp': datetime.now(),
            'agent_id': agent_id
        }

@dataclass
class CollaborationPattern:
    """Template for common collaboration patterns"""
    pattern_id: str
    name: str
    description: str
    collaboration_type: CollaborationType
    required_agent_types: List[str]
    workflow_steps: List[Dict[str, Any]]
    timeout_minutes: int
    success_criteria: Dict[str, Any]

class CollaborationManager:
    """
    Manages agent collaboration protocols and orchestrates multi-agent workflows.
    Provides consensus building, knowledge synthesis, and collective intelligence.
    """
    
    def __init__(self, registry: AgentRegistry, message_bus: MessageBus):
        self.registry = registry
        self.message_bus = message_bus
        self.communicator = AgentCommunicator("collaboration_manager", message_bus)
        
        # Active collaboration sessions
        self.active_sessions: Dict[str, CollaborationSession] = {}
        self.shared_contexts: Dict[str, SharedContext] = {}
        self.collaboration_history: List[CollaborationSession] = []
        
        # Collaboration patterns
        self.collaboration_patterns: Dict[str, CollaborationPattern] = {}
        self._load_collaboration_patterns()
        
        # Configuration
        config_manager = get_agent_config()
        collaboration_config = config_manager.collaboration_config
        self.max_concurrent_sessions = collaboration_config['max_concurrent_sessions']
        self.default_timeout_minutes = collaboration_config['default_timeout_minutes']
        self.context_cleanup_hours = collaboration_config['context_cleanup_hours']
        
        # Statistics
        self.stats = {
            'sessions_created': 0,
            'sessions_completed': 0,
            'sessions_failed': 0,
            'average_session_duration': 0.0,
            'collaboration_types': defaultdict(int)
        }
        
        self.logger = logging.getLogger("collaboration_manager")
        self.running = False
        self.background_tasks = []
    
    async def start(self):
        """Start the collaboration manager"""
        self.running = True
        
        # Start background monitoring tasks
        self.background_tasks = [
            asyncio.create_task(self._monitor_sessions()),
            asyncio.create_task(self._cleanup_expired_contexts()),
            asyncio.create_task(self._update_collaboration_metrics())
        ]
        
        self.logger.info("Collaboration Manager started")
    
    async def stop(self):
        """Stop the collaboration manager gracefully"""
        self.running = False
        
        # Cancel all active sessions
        for session in list(self.active_sessions.values()):
            await self.cancel_collaboration(session.session_id)
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        if self.background_tasks:
            await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        self.logger.info("Collaboration Manager stopped")
    
    async def initiate_collaboration(self, collaboration_request: Dict[str, Any]) -> Optional[str]:
        """
        Initiate a new collaboration session between agents.
        
        Args:
            collaboration_request: Request containing collaboration parameters
            
        Returns:
            Session ID if successful, None otherwise
        """
        try:
            # Extract request parameters
            collaboration_type = CollaborationType(collaboration_request['type'])
            initiator_agent = collaboration_request['initiator_agent']
            target_agents = set(collaboration_request.get('target_agents', []))
            shared_data = collaboration_request.get('shared_data', {})
            timeout_minutes = collaboration_request.get('timeout_minutes', self.default_timeout_minutes)
            
            # Validate agents
            if not await self._validate_collaboration_agents(initiator_agent, target_agents):
                self.logger.error("Invalid agents for collaboration")
                return None
            
            # Check capacity
            if len(self.active_sessions) >= self.max_concurrent_sessions:
                self.logger.warning("Maximum concurrent collaboration sessions reached")
                return None
            
            # Create session
            session_id = str(uuid.uuid4())
            context_id = f"ctx_{session_id}"
            
            # Create shared context
            shared_context = SharedContext(
                context_id=context_id,
                session_id=session_id,
                data=shared_data,
                version=1,
                last_updated=datetime.now(),
                updated_by=initiator_agent,
                access_permissions={initiator_agent}.union(target_agents)
            )
            
            # Create collaboration session
            session = CollaborationSession(
                session_id=session_id,
                collaboration_type=collaboration_type,
                initiator_agent=initiator_agent,
                participating_agents={initiator_agent}.union(target_agents),
                shared_context=shared_context,
                status=CollaborationStatus.PENDING,
                created_at=datetime.now(),
                timeout_minutes=timeout_minutes,
                required_responses=len(target_agents),
                collected_responses={},
                final_result=None,
                metadata=collaboration_request.get('metadata', {})
            )
            
            # Store session and context
            self.active_sessions[session_id] = session
            self.shared_contexts[context_id] = shared_context
            
            # Invite agents to collaborate
            success = await self._invite_agents_to_collaborate(session)
            
            if success:
                session.status = CollaborationStatus.ACTIVE
                self.stats['sessions_created'] += 1
                self.stats['collaboration_types'][collaboration_type.value] += 1
                
                self.logger.info(f"Collaboration session {session_id} initiated")
                return session_id
            else:
                # Clean up failed session
                await self._cleanup_session(session_id)
                return None
                
        except Exception as e:
            self.logger.error(f"Error notifying consensus result: {str(e)}")
    
    async def _notify_collaboration_completed(self, session: CollaborationSession, 
                                            final_result: Dict[str, Any]):
        """Notify all agents that collaboration is completed"""
        try:
            notification_data = {
                'session_id': session.session_id,
                'collaboration_type': session.collaboration_type.value,
                'final_result': final_result,
                'duration_minutes': (datetime.now() - session.created_at).total_seconds() / 60
            }
            
            for agent_id in session.participating_agents:
                try:
                    await self.communicator.send_collaboration_request(
                        recipient_id=agent_id,
                        collaboration_type='collaboration_completed',
                        shared_data=notification_data
                    )
                except Exception as e:
                    self.logger.warning(f"Failed to notify agent {agent_id} of completion: {str(e)}")
                    
        except Exception as e:
            self.logger.error(f"Error notifying collaboration completed: {str(e)}")
    
    async def _notify_collaboration_cancelled(self, session: CollaborationSession):
        """Notify all agents that collaboration was cancelled"""
        try:
            notification_data = {
                'session_id': session.session_id,
                'collaboration_type': session.collaboration_type.value,
                'reason': 'Collaboration cancelled by manager'
            }
            
            for agent_id in session.participating_agents:
                try:
                    await self.communicator.send_collaboration_request(
                        recipient_id=agent_id,
                        collaboration_type='collaboration_cancelled',
                        shared_data=notification_data
                    )
                except Exception as e:
                    self.logger.warning(f"Failed to notify agent {agent_id} of cancellation: {str(e)}")
                    
        except Exception as e:
            self.logger.error(f"Error notifying collaboration cancelled: {str(e)}")
    
    async def _cleanup_session(self, session_id: str):
        """Clean up session and associated resources"""
        try:
            if session_id in self.active_sessions:
                session = self.active_sessions[session_id]
                
                # Remove from active sessions
                del self.active_sessions[session_id]
                
                # Move to history
                self.collaboration_history.append(session)
                
                # Shared context will be cleaned up by background task
                
                self.logger.debug(f"Session {session_id} cleaned up")
                
        except Exception as e:
            self.logger.error(f"Error cleaning up session {session_id}: {str(e)}")
    
    async def _monitor_sessions(self):
        """Monitor active collaboration sessions for timeouts"""
        while self.running:
            try:
                current_time = datetime.now()
                
                for session_id, session in list(self.active_sessions.items()):
                    # Check for timeout
                    session_duration = (current_time - session.created_at).total_seconds()
                    timeout_seconds = session.timeout_minutes * 60
                    
                    if session_duration > timeout_seconds:
                        self.logger.warning(f"Session {session_id} timed out")
                        session.status = CollaborationStatus.TIMEOUT
                        
                        # Update statistics
                        self.stats['sessions_failed'] += 1
                        
                        # Notify agents and clean up
                        await self._notify_collaboration_cancelled(session)
                        await self._cleanup_session(session_id)
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error monitoring sessions: {str(e)}")
                await asyncio.sleep(60)
    
    async def _cleanup_expired_contexts(self):
        """Clean up old shared contexts"""
        while self.running:
            try:
                current_time = datetime.now()
                cleanup_threshold = current_time - timedelta(hours=self.context_cleanup_hours)
                
                # Find expired contexts
                expired_contexts = []
                for context_id, context in self.shared_contexts.items():
                    if context.last_updated < cleanup_threshold:
                        # Check if session is still active
                        session_active = any(
                            session.shared_context.context_id == context_id 
                            for session in self.active_sessions.values()
                        )
                        
                        if not session_active:
                            expired_contexts.append(context_id)
                
                # Clean up expired contexts
                for context_id in expired_contexts:
                    del self.shared_contexts[context_id]
                    self.logger.debug(f"Cleaned up expired context: {context_id}")
                
                await asyncio.sleep(3600)  # Run every hour
                
            except Exception as e:
                self.logger.error(f"Error cleaning up contexts: {str(e)}")
                await asyncio.sleep(300)
    
    async def _update_collaboration_metrics(self):
        """Update collaboration performance metrics"""
        while self.running:
            try:
                # Calculate success rate
                total_sessions = self.stats['sessions_completed'] + self.stats['sessions_failed']
                if total_sessions > 0:
                    success_rate = (self.stats['sessions_completed'] / total_sessions) * 100
                    self.stats['success_rate'] = success_rate
                
                await asyncio.sleep(300)  # Update every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Error updating metrics: {str(e)}")
                await asyncio.sleep(60)
    
    def _load_collaboration_patterns(self):
        """Load predefined collaboration patterns"""
        self.collaboration_patterns = {
            'incident_resolution_consensus': CollaborationPattern(
                pattern_id='incident_resolution_consensus',
                name='Incident Resolution Consensus',
                description='Build consensus on incident resolution approach',
                collaboration_type=CollaborationType.CONSENSUS_BUILDING,
                required_agent_types=['resolution', 'search', 'context'],
                workflow_steps=[
                    {'step': 'context_enrichment', 'agent_type': 'context'},
                    {'step': 'similarity_search', 'agent_type': 'search'},
                    {'step': 'resolution_generation', 'agent_type': 'resolution'},
                    {'step': 'consensus_building', 'agent_type': 'all'}
                ],
                timeout_minutes=20,
                success_criteria={'required_agreement': 0.7}
            ),
            'knowledge_synthesis': CollaborationPattern(
                pattern_id='knowledge_synthesis',
                name='Multi-Agent Knowledge Synthesis',
                description='Synthesize knowledge from multiple agent perspectives',
                collaboration_type=CollaborationType.KNOWLEDGE_SYNTHESIS,
                required_agent_types=['resolution', 'search', 'pattern_detection'],
                workflow_steps=[
                    {'step': 'parallel_analysis', 'agent_type': 'all'},
                    {'step': 'knowledge_integration', 'agent_type': 'manager'}
                ],
                timeout_minutes=30,
                success_criteria={'min_confidence': 0.6}
            ),
            'expert_consultation': CollaborationPattern(
                pattern_id='expert_consultation',
                name='Expert Agent Consultation',
                description='Consult expert agents for complex decisions',
                collaboration_type=CollaborationType.EXPERT_CONSULTATION,
                required_agent_types=['resolution', 'pattern_detection'],
                workflow_steps=[
                    {'step': 'expert_analysis', 'agent_type': 'experts'},
                    {'step': 'opinion_synthesis', 'agent_type': 'manager'}
                ],
                timeout_minutes=25,
                success_criteria={'expert_agreement': 0.8}
            )
        }
    
    def _serialize_session_status(self, session: CollaborationSession) -> Dict[str, Any]:
        """Serialize collaboration session status"""
        return {
            'session_id': session.session_id,
            'collaboration_type': session.collaboration_type.value,
            'status': session.status.value,
            'initiator_agent': session.initiator_agent,
            'participating_agents': list(session.participating_agents),
            'created_at': session.created_at.isoformat(),
            'timeout_minutes': session.timeout_minutes,
            'responses_collected': len(session.collected_responses),
            'required_responses': session.required_responses,
            'shared_context_version': session.shared_context.version,
            'final_result': session.final_result,
            'metadata': session.metadata
        }
    
    def _update_average_duration(self, duration_seconds: float):
        """Update average session duration"""
        total_completed = self.stats['sessions_completed']
        if total_completed > 1:
            current_avg = self.stats['average_session_duration']
            new_avg = ((current_avg * (total_completed - 1)) + duration_seconds) / total_completed
            self.stats['average_session_duration'] = new_avg
        else:
            self.stats['average_session_duration'] = duration_seconds
    
    def get_manager_status(self) -> Dict[str, Any]:
        """Get collaboration manager status and metrics"""
        return {
            'status': 'running' if self.running else 'stopped',
            'active_sessions': len(self.active_sessions),
            'shared_contexts': len(self.shared_contexts),
            'collaboration_patterns': len(self.collaboration_patterns),
            'statistics': {
                'sessions_created': self.stats['sessions_created'],
                'sessions_completed': self.stats['sessions_completed'],
                'sessions_failed': self.stats['sessions_failed'],
                'success_rate': self.stats.get('success_rate', 0.0),
                'average_duration_seconds': self.stats['average_session_duration'],
                'collaboration_types': dict(self.stats['collaboration_types'])
            },
            'timestamp': datetime.now().isoformat()
        }
            
    
    async def join_collaboration(self, session_id: str, agent_id: str, 
                               contribution: Dict[str, Any]) -> bool:
        """
        Agent joins an active collaboration session with their contribution.
        
        Args:
            session_id: Collaboration session ID
            agent_id: Agent joining the collaboration
            contribution: Agent's contribution to the collaboration
            
        Returns:
            True if successfully joined
        """
        try:
            if session_id not in self.active_sessions:
                self.logger.warning(f"Collaboration session {session_id} not found")
                return False
            
            session = self.active_sessions[session_id]
            
            # Validate agent participation
            if agent_id not in session.participating_agents:
                self.logger.warning(f"Agent {agent_id} not authorized for session {session_id}")
                return False
            
            if session.status != CollaborationStatus.ACTIVE:
                self.logger.warning(f"Session {session_id} is not active")
                return False
            
            # Add agent's response
            session.add_response(agent_id, contribution)
            
            # Update shared context if contribution includes updates
            if 'context_updates' in contribution:
                session.shared_context.update_data(
                    contribution['context_updates'], 
                    agent_id
                )
            
            # Check if all responses collected
            if len(session.collected_responses) >= session.required_responses:
                session.status = CollaborationStatus.COLLECTING_RESPONSES
                # Trigger synthesis if applicable
                if session.collaboration_type in [CollaborationType.CONSENSUS_BUILDING, 
                                                CollaborationType.KNOWLEDGE_SYNTHESIS]:
                    await self._trigger_synthesis(session)
            
            # Notify other agents of the contribution
            await self._notify_agent_joined(session, agent_id, contribution)
            
            self.logger.info(f"Agent {agent_id} joined collaboration {session_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error joining collaboration: {str(e)}")
            return False
    
    async def build_consensus(self, session_id: str, 
                            consensus_request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Build consensus among participating agents on a specific topic.
        
        Args:
            session_id: Collaboration session ID
            consensus_request: Request containing consensus parameters
            
        Returns:
            Consensus result or None if failed
        """
        try:
            if session_id not in self.active_sessions:
                return None
            
            session = self.active_sessions[session_id]
            
            if session.collaboration_type != CollaborationType.CONSENSUS_BUILDING:
                self.logger.warning(f"Session {session_id} is not a consensus building session")
                return None
            
            # Extract consensus parameters
            topic = consensus_request.get('topic', 'general')
            options = consensus_request.get('options', [])
            required_agreement = consensus_request.get('required_agreement', 0.6)  # 60%
            
            # Collect votes from all participating agents
            votes = {}
            for agent_id in session.participating_agents:
                if agent_id == session.initiator_agent:
                    continue  # Initiator doesn't vote
                
                try:
                    # Request vote from agent
                    vote_request = {
                        'topic': topic,
                        'options': options,
                        'context': session.shared_context.data,
                        'session_id': session_id
                    }
                    
                    response = await self.communicator.send_collaboration_request(
                        recipient_id=agent_id,
                        collaboration_type='consensus_vote',
                        shared_data=vote_request
                    )
                    
                    if response:
                        votes[agent_id] = vote_request.get('vote', 'abstain')
                    
                except Exception as e:
                    self.logger.warning(f"Failed to get vote from agent {agent_id}: {str(e)}")
                    votes[agent_id] = 'abstain'
            
            # Analyze votes
            vote_counts = Counter(votes.values())
            total_votes = len(votes)
            
            # Determine consensus
            consensus_result = {
                'topic': topic,
                'votes': votes,
                'vote_counts': dict(vote_counts),
                'total_votes': total_votes,
                'consensus_reached': False,
                'winning_option': None,
                'agreement_percentage': 0.0,
                'timestamp': datetime.now().isoformat()
            }
            
            if total_votes > 0:
                # Find the most voted option
                most_voted = vote_counts.most_common(1)[0]
                winning_option, vote_count = most_voted
                
                agreement_percentage = vote_count / total_votes
                
                if agreement_percentage >= required_agreement:
                    consensus_result['consensus_reached'] = True
                    consensus_result['winning_option'] = winning_option
                    consensus_result['agreement_percentage'] = agreement_percentage
            
            # Store consensus result in shared context
            session.shared_context.update_data(
                {'consensus_result': consensus_result},
                'collaboration_manager'
            )
            
            # Notify all agents of consensus result
            await self._notify_consensus_result(session, consensus_result)
            
            self.logger.info(f"Consensus building completed for session {session_id}")
            return consensus_result
            
        except Exception as e:
            self.logger.error(f"Error building consensus: {str(e)}")
            return None
    
    async def synthesize_knowledge(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Synthesize knowledge from all agent contributions in a collaboration.
        
        Args:
            session_id: Collaboration session ID
            
        Returns:
            Synthesized knowledge or None if failed
        """
        try:
            if session_id not in self.active_sessions:
                return None
            
            session = self.active_sessions[session_id]
            session.status = CollaborationStatus.SYNTHESIZING
            
            # Collect all contributions
            contributions = []
            for agent_id, response_data in session.collected_responses.items():
                contribution = {
                    'agent_id': agent_id,
                    'agent_type': self._get_agent_type(agent_id),
                    'response': response_data['response'],
                    'timestamp': response_data['timestamp'],
                    'confidence': response_data['response'].get('confidence', 0.5)
                }
                contributions.append(contribution)
            
            # Synthesize knowledge based on collaboration type
            if session.collaboration_type == CollaborationType.KNOWLEDGE_SYNTHESIS:
                synthesis_result = await self._synthesize_agent_knowledge(contributions, session)
            elif session.collaboration_type == CollaborationType.EXPERT_CONSULTATION:
                synthesis_result = await self._synthesize_expert_opinions(contributions, session)
            elif session.collaboration_type == CollaborationType.PEER_REVIEW:
                synthesis_result = await self._synthesize_peer_reviews(contributions, session)
            else:
                synthesis_result = await self._synthesize_general_contributions(contributions, session)
            
            # Store synthesis result
            session.final_result = synthesis_result
            session.shared_context.update_data(
                {'synthesis_result': synthesis_result},
                'collaboration_manager'
            )
            
            # Mark session as completed
            await self.complete_collaboration(session_id, synthesis_result)
            
            self.logger.info(f"Knowledge synthesis completed for session {session_id}")
            return synthesis_result
            
        except Exception as e:
            self.logger.error(f"Error synthesizing knowledge: {str(e)}")
            return None
    
    async def complete_collaboration(self, session_id: str, 
                                   final_result: Dict[str, Any]) -> bool:
        """
        Complete a collaboration session and store results.
        
        Args:
            session_id: Collaboration session ID
            final_result: Final collaboration result
            
        Returns:
            True if completed successfully
        """
        try:
            if session_id not in self.active_sessions:
                return False
            
            session = self.active_sessions[session_id]
            session.status = CollaborationStatus.COMPLETED
            session.final_result = final_result
            
            # Calculate session duration
            duration = (datetime.now() - session.created_at).total_seconds()
            
            # Update statistics
            self.stats['sessions_completed'] += 1
            self._update_average_duration(duration)
            
            # Notify all participating agents
            await self._notify_collaboration_completed(session, final_result)
            
            # Move to history
            self.collaboration_history.append(session)
            del self.active_sessions[session_id]
            
            # Keep shared context for a while for reference
            # It will be cleaned up by background task
            
            self.logger.info(f"Collaboration session {session_id} completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error completing collaboration: {str(e)}")
            return False
    
    async def cancel_collaboration(self, session_id: str) -> bool:
        """Cancel an active collaboration session"""
        try:
            if session_id not in self.active_sessions:
                return False
            
            session = self.active_sessions[session_id]
            session.status = CollaborationStatus.CANCELLED
            
            # Notify participating agents
            await self._notify_collaboration_cancelled(session)
            
            # Update statistics
            self.stats['sessions_failed'] += 1
            
            # Clean up
            await self._cleanup_session(session_id)
            
            self.logger.info(f"Collaboration session {session_id} cancelled")
            return True
            
        except Exception as e:
            self.logger.error(f"Error cancelling collaboration: {str(e)}")
            return False
    
    async def get_collaboration_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a collaboration session"""
        try:
            # Check active sessions
            if session_id in self.active_sessions:
                session = self.active_sessions[session_id]
                return self._serialize_session_status(session)
            
            # Check history
            for session in self.collaboration_history:
                if session.session_id == session_id:
                    return self._serialize_session_status(session)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting collaboration status: {str(e)}")
            return None
    
    async def _validate_collaboration_agents(self, initiator: str, targets: Set[str]) -> bool:
        """Validate that all agents exist and are available"""
        all_agents = {initiator}.union(targets)
        
        for agent_id in all_agents:
            agent_reg = self.registry.get_agent_by_id(agent_id)
            if not agent_reg or agent_reg.status.value != 'active':
                return False
        
        return True
    
    async def _invite_agents_to_collaborate(self, session: CollaborationSession) -> bool:
        """Invite agents to join collaboration session"""
        try:
            invitation_data = {
                'session_id': session.session_id,
                'collaboration_type': session.collaboration_type.value,
                'initiator': session.initiator_agent,
                'shared_context': session.shared_context.data,
                'timeout_minutes': session.timeout_minutes
            }
            
            success_count = 0
            for agent_id in session.participating_agents:
                if agent_id == session.initiator_agent:
                    success_count += 1  # Initiator is automatically included
                    continue
                
                try:
                    success = await self.communicator.send_collaboration_request(
                        recipient_id=agent_id,
                        collaboration_type='session_invitation',
                        shared_data=invitation_data
                    )
                    
                    if success:
                        success_count += 1
                        
                except Exception as e:
                    self.logger.warning(f"Failed to invite agent {agent_id}: {str(e)}")
            
            # Require at least 50% success rate
            return success_count >= len(session.participating_agents) * 0.5
            
        except Exception as e:
            self.logger.error(f"Error inviting agents: {str(e)}")
            return False
    
    async def _trigger_synthesis(self, session: CollaborationSession):
        """Trigger knowledge synthesis when all responses are collected"""
        try:
            # Small delay to ensure all responses are processed
            await asyncio.sleep(1)
            
            # Start synthesis
            await self.synthesize_knowledge(session.session_id)
            
        except Exception as e:
            self.logger.error(f"Error triggering synthesis: {str(e)}")
    
    async def _synthesize_agent_knowledge(self, contributions: List[Dict[str, Any]], 
                                        session: CollaborationSession) -> Dict[str, Any]:
        """Synthesize knowledge from multiple agent contributions"""
        try:
            # Group contributions by agent type
            by_agent_type = defaultdict(list)
            for contrib in contributions:
                by_agent_type[contrib['agent_type']].append(contrib)
            
            # Extract key insights from each agent type
            synthesis = {
                'synthesis_type': 'knowledge_integration',
                'agent_contributions': {},
                'key_insights': [],
                'confidence_score': 0.0,
                'recommendations': [],
                'timestamp': datetime.now().isoformat()
            }
            
            total_confidence = 0.0
            insight_count = 0
            
            for agent_type, contribs in by_agent_type.items():
                # Combine insights from same agent type
                combined_insights = []
                combined_confidence = 0.0
                
                for contrib in contribs:
                    response = contrib['response']
                    if 'insights' in response:
                        combined_insights.extend(response['insights'])
                    if 'recommendations' in response:
                        synthesis['recommendations'].extend(response['recommendations'])
                    
                    combined_confidence += contrib['confidence']
                
                if contribs:
                    avg_confidence = combined_confidence / len(contribs)
                    synthesis['agent_contributions'][agent_type] = {
                        'insights': combined_insights,
                        'confidence': avg_confidence,
                        'contribution_count': len(contribs)
                    }
                    
                    total_confidence += avg_confidence
                    insight_count += len(combined_insights)
            
            # Calculate overall confidence
            if by_agent_type:
                synthesis['confidence_score'] = total_confidence / len(by_agent_type)
            
            # Extract most important insights
            all_insights = []
            for agent_type, data in synthesis['agent_contributions'].items():
                for insight in data['insights']:
                    all_insights.append({
                        'insight': insight,
                        'source_agent_type': agent_type,
                        'confidence': data['confidence']
                    })
            
            # Sort by confidence and take top insights
            all_insights.sort(key=lambda x: x['confidence'], reverse=True)
            synthesis['key_insights'] = all_insights[:10]  # Top 10
            
            return synthesis
            
        except Exception as e:
            self.logger.error(f"Error synthesizing agent knowledge: {str(e)}")
            return {'error': str(e)}
    
    async def _synthesize_expert_opinions(self, contributions: List[Dict[str, Any]], 
                                        session: CollaborationSession) -> Dict[str, Any]:
        """Synthesize expert opinions with weighted importance"""
        try:
            synthesis = {
                'synthesis_type': 'expert_consultation',
                'expert_opinions': [],
                'consensus_view': None,
                'conflicting_views': [],
                'confidence_score': 0.0,
                'timestamp': datetime.now().isoformat()
            }
            
            # Weight experts by their historical performance
            weighted_opinions = []
            for contrib in contributions:
                # Get agent's expertise level (simplified)
                agent_reg = self.registry.get_agent_by_id(contrib['agent_id'])
                expertise_weight = 1.0
                if agent_reg:
                    # Higher weight for better performing agents
                    success_rate = agent_reg.health_status.get('success_rate', 50.0)
                    expertise_weight = success_rate / 100.0
                
                weighted_opinion = {
                    'agent_id': contrib['agent_id'],
                    'opinion': contrib['response'],
                    'confidence': contrib['confidence'],
                    'weight': expertise_weight,
                    'timestamp': contrib['timestamp']
                }
                weighted_opinions.append(weighted_opinion)
            
            synthesis['expert_opinions'] = weighted_opinions
            
            # Find consensus if opinions are similar
            # Simplified consensus detection
            opinions = [op['opinion'] for op in weighted_opinions]
            if len(opinions) > 1:
                # Check for consensus (simplified)
                consensus_score = self._calculate_opinion_consensus(opinions)
                if consensus_score > 0.7:
                    synthesis['consensus_view'] = {
                        'description': 'High consensus among experts',
                        'consensus_score': consensus_score
                    }
            
            # Calculate weighted confidence
            total_weight = sum(op['weight'] for op in weighted_opinions)
            if total_weight > 0:
                weighted_confidence = sum(
                    op['confidence'] * op['weight'] for op in weighted_opinions
                ) / total_weight
                synthesis['confidence_score'] = weighted_confidence
            
            return synthesis
            
        except Exception as e:
            self.logger.error(f"Error synthesizing expert opinions: {str(e)}")
            return {'error': str(e)}
    
    async def _synthesize_peer_reviews(self, contributions: List[Dict[str, Any]], 
                                     session: CollaborationSession) -> Dict[str, Any]:
        """Synthesize peer review contributions"""
        try:
            synthesis = {
                'synthesis_type': 'peer_review',
                'reviews': [],
                'overall_assessment': None,
                'improvement_suggestions': [],
                'approval_rate': 0.0,
                'timestamp': datetime.now().isoformat()
            }
            
            approvals = 0
            total_reviews = len(contributions)
            
            for contrib in contributions:
                review = contrib['response']
                synthesis['reviews'].append({
                    'reviewer': contrib['agent_id'],
                    'review': review,
                    'confidence': contrib['confidence'],
                    'timestamp': contrib['timestamp']
                })
                
                # Count approvals
                if review.get('approved', False):
                    approvals += 1
                
                # Collect suggestions
                if 'suggestions' in review:
                    synthesis['improvement_suggestions'].extend(review['suggestions'])
            
            # Calculate approval rate
            if total_reviews > 0:
                synthesis['approval_rate'] = approvals / total_reviews
            
            # Overall assessment
            if synthesis['approval_rate'] >= 0.8:
                synthesis['overall_assessment'] = 'Approved with high confidence'
            elif synthesis['approval_rate'] >= 0.6:
                synthesis['overall_assessment'] = 'Approved with moderate confidence'
            else:
                synthesis['overall_assessment'] = 'Requires revision'
            
            return synthesis
            
        except Exception as e:
            self.logger.error(f"Error synthesizing peer reviews: {str(e)}")
            return {'error': str(e)}
    
    async def _synthesize_general_contributions(self, contributions: List[Dict[str, Any]], 
                                              session: CollaborationSession) -> Dict[str, Any]:
        """Synthesize general contributions"""
        try:
            synthesis = {
                'synthesis_type': 'general_collaboration',
                'contributions': contributions,
                'summary': None,
                'confidence_score': 0.0,
                'timestamp': datetime.now().isoformat()
            }
            
            # Calculate average confidence
            if contributions:
                avg_confidence = sum(c['confidence'] for c in contributions) / len(contributions)
                synthesis['confidence_score'] = avg_confidence
            
            # Create summary
            synthesis['summary'] = f"Collaboration completed with {len(contributions)} agent contributions"
            
            return synthesis
            
        except Exception as e:
            self.logger.error(f"Error synthesizing general contributions: {str(e)}")
            return {'error': str(e)}
    
    def _calculate_opinion_consensus(self, opinions: List[Dict[str, Any]]) -> float:
        """Calculate consensus score between opinions (simplified)"""
        # Simplified consensus calculation
        # In production, use more sophisticated similarity measures
        return 0.8  # Placeholder
    
    def _get_agent_type(self, agent_id: str) -> str:
        """Get agent type from registry"""
        agent_reg = self.registry.get_agent_by_id(agent_id)
        return agent_reg.agent_type if agent_reg else 'unknown'
    
    async def _notify_agent_joined(self, session: CollaborationSession, 
                                 agent_id: str, contribution: Dict[str, Any]):
        """Notify other agents that an agent joined with contribution"""
        try:
            notification_data = {
                'session_id': session.session_id,
                'joined_agent': agent_id,
                'contribution_summary': contribution.get('summary', 'No summary'),
                'participant_count': len(session.collected_responses)
            }
            
            for participant in session.participating_agents:
                if participant != agent_id:  # Don't notify the agent who just joined
                    try:
                        await self.communicator.send_collaboration_request(
                            recipient_id=participant,
                            collaboration_type='agent_joined_notification',
                            shared_data=notification_data
                        )
                    except Exception as e:
                        self.logger.warning(f"Failed to notify agent {participant}: {str(e)}")
                        
        except Exception as e:
            self.logger.error(f"Error notifying agent joined: {str(e)}")
    
    async def _notify_consensus_result(self, session: CollaborationSession, 
                                 consensus_result: Dict[str, Any]):
        """Notify all agents of consensus result"""
        try:
            for agent_id in session.participating_agents:
                try:
                    await self.communicator.send_collaboration_request(
                        recipient_id=agent_id,
                        collaboration_type='consensus_result',
                        shared_data={
                            'session_id': session.session_id,
                            'consensus_result': consensus_result
                        }
                    )
                except Exception as e:
                    self.logger.warning(f"Failed to notify agent {agent_id} of consensus: {str(e)}")
                    
        except Exception as e:
            self.logger.error(f"Error notifying consensus result: {str(e)}")