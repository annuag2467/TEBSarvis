"""
Conversation Workflow - Chat + search + resolve chain for natural language interactions
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum
import uuid
import json

from ..agents.core.base_agent import BaseAgent
from ..agents.core.agent_registry import get_global_registry
from ..agents.core.agent_communication import MessageBus, AgentCommunicator
from ..agents.core.message_types import TaskType, Priority, create_task_request
from ..agents.orchestrator.agent_coordinator import AgentCoordinator
from ..agents.orchestrator.collaboration_manager import CollaborationManager
from ..config.agent_config import get_agent_config, AgentType

logger = logging.getLogger(__name__)

class ConversationState(Enum):
    INITIATED = "initiated"
    PROCESSING = "processing"
    SEARCHING = "searching"
    RESOLVING = "resolving"
    RESPONDING = "responding"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"

class IntentType(Enum):
    QUESTION = "question"
    PROBLEM_REPORT = "problem_report"
    HOW_TO_REQUEST = "how_to_request"
    STATUS_INQUIRY = "status_inquiry"
    GENERAL_CHAT = "general_chat"
    ESCALATION_REQUEST = "escalation_request"

class ConversationWorkflow:
    """
    Orchestrates natural language conversation workflow:
    1. Intent recognition and entity extraction
    2. Context-aware knowledge search
    3. Resolution generation (if applicable)
    4. Response synthesis and generation
    5. Follow-up suggestion generation
    6. Conversation state management
    """
    
    def __init__(self):
        self.registry = get_global_registry()
        self.message_bus = MessageBus()
        self.coordinator = AgentCoordinator(self.registry, self.message_bus)
        self.collaboration_manager = CollaborationManager(self.registry, self.message_bus)
        self.communicator = AgentCommunicator("conversation_workflow", self.message_bus)
        self.config = get_agent_config()
        
        # Conversation management
        self.active_conversations: Dict[str, Dict[str, Any]] = {}
        self.conversation_history: List[Dict[str, Any]] = []
        
        # Configuration
        self.max_conversation_duration = 3600  # 1 hour
        self.max_active_conversations = 100
        self.context_window_size = 10  # Number of previous exchanges to consider
        self.session_cleanup_interval = 1800  # 30 minutes
        
        # Background tasks
        self.background_tasks: List[asyncio.Task] = []
        
        self.logger = logging.getLogger("workflows.conversation")
    
    async def start(self):
        """Start the conversation workflow system"""
        try:
            await self.message_bus.start()
            await self.coordinator.start()
            await self.collaboration_manager.start()
            
            # Start background cleanup task
            self.background_tasks = [
                asyncio.create_task(self._cleanup_expired_conversations())
            ]
            
            self.logger.info("Conversation Workflow system started")
            
        except Exception as e:
            self.logger.error(f"Failed to start conversation workflow: {str(e)}")
            raise
    
    async def stop(self):
        """Stop the conversation workflow system"""
        try:
            # Cancel background tasks
            for task in self.background_tasks:
                if not task.done():
                    task.cancel()
            
            if self.background_tasks:
                await asyncio.gather(*self.background_tasks, return_exceptions=True)
            
            await self.collaboration_manager.stop()
            await self.coordinator.stop()
            await self.message_bus.stop()
            
            self.logger.info("Conversation Workflow system stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping conversation workflow: {str(e)}")
    
    async def process_user_message(self, user_message: str, 
                                 session_id: Optional[str] = None,
                                 user_context: Optional[Dict[str, Any]] = None,
                                 conversation_options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a user message through the complete conversation workflow.
        
        Args:
            user_message: User's input message
            session_id: Optional conversation session ID
            user_context: Optional user context information
            conversation_options: Optional conversation configuration
            
        Returns:
            Complete conversation response with all workflow outputs
        """
        try:
            # Initialize or retrieve conversation session
            if not session_id:
                session_id = f"conv_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
            
            conversation_state = self._get_or_create_conversation_session(
                session_id, user_context, conversation_options
            )
            
            # Update conversation state
            conversation_state['state'] = ConversationState.PROCESSING
            conversation_state['current_message'] = user_message
            conversation_state['last_activity'] = datetime.now()
            
            # Add user message to history
            conversation_state['messages'].append({
                'role': 'user',
                'content': user_message,
                'timestamp': datetime.now().isoformat()
            })
            
            self.logger.info(f"Processing message in conversation {session_id}")
            
            # Step 1: Intent Recognition and Entity Extraction
            intent_result = await self._recognize_intent_and_entities(
                session_id, user_message, conversation_state
            )
            
            # Step 2: Context-Aware Knowledge Search
            search_result = await self._perform_contextual_search(
                session_id, user_message, intent_result, conversation_state
            )
            
            # Step 3: Generate Resolution (if needed)
            resolution_result = None
            if self._requires_resolution(intent_result):
                resolution_result = await self._generate_contextual_resolution(
                    session_id, user_message, intent_result, search_result, conversation_state
                )
            
            # Step 4: Response Synthesis and Generation
            response_result = await self._synthesize_response(
                session_id, user_message, intent_result, search_result, 
                resolution_result, conversation_state
            )
            
            # Step 5: Generate Follow-up Suggestions
            followup_result = await self._generate_followup_suggestions(
                session_id, intent_result, response_result, conversation_state
            )
            
            # Step 6: Update Conversation State
            conversation_state['state'] = ConversationState.COMPLETED
            conversation_state['turn_count'] += 1
            
            # Add assistant response to history
            conversation_state['messages'].append({
                'role': 'assistant',
                'content': response_result.get('response', ''),
                'timestamp': datetime.now().isoformat(),
                'metadata': {
                    'intent': intent_result.get('intent', {}),
                    'sources_used': len(search_result.get('results', [])),
                    'resolution_provided': resolution_result is not None
                }
            })
            
            # Compile final response
            final_response = await self._compile_conversation_response(
                session_id, user_message, intent_result, search_result, 
                resolution_result, response_result, followup_result, conversation_state
            )
            
            self.logger.info(f"Conversation {session_id} processed successfully")
            return final_response
            
        except Exception as e:
            self.logger.error(f"Error processing conversation: {str(e)}")
            
            # Update conversation state on error
            if session_id in self.active_conversations:
                self.active_conversations[session_id]['state'] = ConversationState.FAILED
                self.active_conversations[session_id]['error'] = str(e)
            
            # Return error response
            return {
                'session_id': session_id,
                'response': "I apologize, but I encountered an error while processing your message. Please try again.",
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def _recognize_intent_and_entities(self, session_id: str, user_message: str,
                                           conversation_state: Dict[str, Any]) -> Dict[str, Any]:
        """Step 1: Recognize user intent and extract entities"""
        try:
            self.logger.info(f"[{session_id}] Step 1: Recognizing intent and entities")
            
            conversation_state['state'] = ConversationState.PROCESSING
            
            # Find Conversation Agent for intent recognition
            conversation_agent = self.registry.get_best_agent_for_capability('intent_recognition')
            if not conversation_agent:
                raise RuntimeError("No Conversation Agent available for intent recognition")
            
            # Prepare context from conversation history
            conversation_context = self._build_conversation_context(conversation_state)
            
            task_data = {
                'user_input': user_message,
                'context': {
                    'conversation_history': conversation_context,
                    'user_context': conversation_state.get('user_context', {}),
                    'session_metadata': {
                        'turn_count': conversation_state['turn_count'],
                        'session_duration': self._calculate_session_duration(conversation_state)
                    }
                },
                'extract_entities': True,
                'confidence_threshold': 0.6
            }
            
            response = await self.communicator.send_task_request(
                recipient_id=conversation_agent.agent_id,
                task_type='intent_recognition',
                task_data=task_data,
                timeout_seconds=60
            )
            
            if not response.success:
                raise RuntimeError(f"Intent recognition failed: {response.error_message}")
            
            # Store intent result in conversation state
            conversation_state['last_intent'] = response.result_data
            
            self.logger.info(f"[{session_id}] Intent recognition completed")
            return response.result_data
            
        except Exception as e:
            self.logger.error(f"[{session_id}] Intent recognition failed: {str(e)}")
            # Return default intent on failure
            return {
                'intent': {'type': 'general_chat', 'confidence': 0.5},
                'entities': {},
                'error': str(e)
            }
    
    async def _perform_contextual_search(self, session_id: str, user_message: str,
                                       intent_result: Dict[str, Any],
                                       conversation_state: Dict[str, Any]) -> Dict[str, Any]:
        """Step 2: Perform context-aware knowledge search"""
        try:
            self.logger.info(f"[{session_id}] Step 2: Performing contextual search")
            
            conversation_state['state'] = ConversationState.SEARCHING
            
            # Find Search Agent
            search_agent = self.registry.get_best_agent_for_capability('contextual_search')
            if not search_agent:
                # Fallback to semantic search
                search_agent = self.registry.get_best_agent_for_capability('semantic_search')
            
            if not search_agent:
                raise RuntimeError("No Search Agent available")
            
            # Build enhanced search query
            search_query = await self._build_enhanced_search_query(
                user_message, intent_result, conversation_state
            )
            
            # Prepare search context
            search_context = {
                'user_intent': intent_result.get('intent', {}),
                'entities': intent_result.get('entities', {}),
                'conversation_context': self._build_conversation_context(conversation_state),
                'user_level': conversation_state.get('user_context', {}).get('experience_level', 'intermediate'),
                'urgency': self._determine_urgency(intent_result),
                'prefer_recent': True
            }
            
            task_data = {
                'query': search_query,
                'context': search_context,
                'max_results': 5,
                'filters': self._build_search_filters(intent_result, conversation_state),
                'include_metadata': True,
                'boost_recent': True
            }
            
            # Use contextual search if available, otherwise semantic search
            task_type = 'contextual_search' if search_agent and hasattr(search_agent, 'contextual_search') else 'semantic_search'
            
            response = await self.communicator.send_task_request(
                recipient_id=search_agent.agent_id,
                task_type=task_type,
                task_data=task_data,
                timeout_seconds=90
            )
            
            if not response.success:
                self.logger.warning(f"[{session_id}] Search failed: {response.error_message}")
                return {'results': [], 'error': response.error_message}
            
            self.logger.info(f"[{session_id}] Contextual search completed")
            return response.result_data
            
        except Exception as e:
            self.logger.error(f"[{session_id}] Contextual search failed: {str(e)}")
            return {'results': [], 'error': str(e)}
    
    async def _generate_contextual_resolution(self, session_id: str, user_message: str,
                                            intent_result: Dict[str, Any],
                                            search_result: Dict[str, Any],
                                            conversation_state: Dict[str, Any]) -> Dict[str, Any]:
        """Step 3: Generate resolution for problem-solving intents"""
        try:
            self.logger.info(f"[{session_id}] Step 3: Generating contextual resolution")
            
            conversation_state['state'] = ConversationState.RESOLVING
            
            # Find Resolution Agent
            resolution_agent = self.registry.get_best_agent_for_capability('incident_resolution')
            if not resolution_agent:
                raise RuntimeError("No Resolution Agent available")
            
            # Extract problem context from intent and entities
            incident_data = self._extract_incident_data_from_conversation(
                user_message, intent_result, conversation_state
            )
            
            task_data = {
                'incident_data': incident_data,
                'conversation_context': self._build_conversation_context(conversation_state),
                'search_results': search_result.get('results', []),
                'user_context': conversation_state.get('user_context', {}),
                'resolution_style': 'conversational',
                'max_solutions': 3,
                'include_explanations': True
            }
            
            response = await self.communicator.send_task_request(
                recipient_id=resolution_agent.agent_id,
                task_type='incident_resolution',
                task_data=task_data,
                timeout_seconds=180
            )
            
            if not response.success:
                self.logger.warning(f"[{session_id}] Resolution generation failed: {response.error_message}")
                return {'error': response.error_message}
            
            self.logger.info(f"[{session_id}] Contextual resolution completed")
            return response.result_data
            
        except Exception as e:
            self.logger.error(f"[{session_id}] Resolution generation failed: {str(e)}")
            return {'error': str(e)}
    
    async def _synthesize_response(self, session_id: str, user_message: str,
                                 intent_result: Dict[str, Any],
                                 search_result: Dict[str, Any],
                                 resolution_result: Optional[Dict[str, Any]],
                                 conversation_state: Dict[str, Any]) -> Dict[str, Any]:
        """Step 4: Synthesize final response"""
        try:
            self.logger.info(f"[{session_id}] Step 4: Synthesizing response")
            
            conversation_state['state'] = ConversationState.RESPONDING
            
            # Find Conversation Agent for response generation
            conversation_agent = self.registry.get_best_agent_for_capability('response_generation')
            if not conversation_agent:
                raise RuntimeError("No Conversation Agent available for response generation")
            
            # Prepare comprehensive context for response generation
            knowledge_context = {
                'search_results': search_result.get('results', []),
                'resolution_data': resolution_result,
                'intent_context': intent_result,
                'conversation_history': self._build_conversation_context(conversation_state)
            }
            
            # Determine response style based on intent and user context
            response_style = self._determine_response_style(intent_result, conversation_state)
            
            task_data = {
                'user_query': user_message,
                'knowledge_context': knowledge_context,
                'response_style': response_style,
                'conversation_context': conversation_state,
                'include_citations': True,
                'max_response_length': 500,
                'personalization': conversation_state.get('user_context', {})
            }
            
            response = await self.communicator.send_task_request(
                recipient_id=conversation_agent.agent_id,
                task_type='response_generation',
                task_data=task_data,
                timeout_seconds=120
            )
            
            if not response.success:
                raise RuntimeError(f"Response synthesis failed: {response.error_message}")
            
            self.logger.info(f"[{session_id}] Response synthesis completed")
            return response.result_data
            
        except Exception as e:
            self.logger.error(f"[{session_id}] Response synthesis failed: {str(e)}")
            # Return fallback response
            return {
                'response': self._generate_fallback_response(user_message, intent_result),
                'confidence': 0.3,
                'error': str(e)
            }
    
    async def _generate_followup_suggestions(self, session_id: str,
                                           intent_result: Dict[str, Any],
                                           response_result: Dict[str, Any],
                                           conversation_state: Dict[str, Any]) -> Dict[str, Any]:
        """Step 5: Generate follow-up suggestions"""
        try:
            self.logger.info(f"[{session_id}] Step 5: Generating follow-up suggestions")
            
            # Use Conversation Agent for follow-up generation
            conversation_agent = self.registry.get_best_agent_for_capability('conversation_management')
            if not conversation_agent:
                # Generate basic follow-ups locally
                return self._generate_basic_followups(intent_result, response_result, conversation_state)
            
            task_data = {
                'conversation_history': self._build_conversation_context(conversation_state),
                'current_intent': intent_result.get('intent', {}),
                'response_provided': response_result.get('response', ''),
                'user_context': conversation_state.get('user_context', {}),
                'max_suggestions': 3
            }
            
            response = await self.communicator.send_task_request(
                recipient_id=conversation_agent.agent_id,
                task_type='conversation_management',
                task_data=task_data,
                timeout_seconds=60
            )
            
            if response.success:
                return response.result_data
            else:
                return self._generate_basic_followups(intent_result, response_result, conversation_state)
                
        except Exception as e:
            self.logger.error(f"[{session_id}] Follow-up generation failed: {str(e)}")
            return self._generate_basic_followups(intent_result, response_result, conversation_state)
    
    def _get_or_create_conversation_session(self, session_id: str, 
                                          user_context: Optional[Dict[str, Any]],
                                          options: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Get existing conversation session or create new one"""
        if session_id in self.active_conversations:
            return self.active_conversations[session_id]
        
        # Check capacity
        if len(self.active_conversations) >= self.max_active_conversations:
            # Remove oldest session
            oldest_session = min(
                self.active_conversations.values(),
                key=lambda x: x['created_at']
            )
            del self.active_conversations[oldest_session['session_id']]
        
        # Create new conversation session
        conversation_state = {
            'session_id': session_id,
            'created_at': datetime.now(),
            'last_activity': datetime.now(),
            'state': ConversationState.INITIATED,
            'user_context': user_context or {},
            'options': options or {},
            'messages': [],
            'turn_count': 0,
            'context_summary': '',
            'entities_mentioned': set(),
            'topics_discussed': []
        }
        
        self.active_conversations[session_id] = conversation_state
        return conversation_state
    
    def _build_conversation_context(self, conversation_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build conversation context for agents"""
        messages = conversation_state.get('messages', [])
        
        # Return last N messages within context window
        return messages[-self.context_window_size:] if messages else []
    
    def _calculate_session_duration(self, conversation_state: Dict[str, Any]) -> float:
        """Calculate session duration in seconds"""
        created_at = conversation_state.get('created_at', datetime.now())
        return (datetime.now() - created_at).total_seconds()
    
    async def _build_enhanced_search_query(self, user_message: str,
                                         intent_result: Dict[str, Any],
                                         conversation_state: Dict[str, Any]) -> str:
        """Build enhanced search query using intent and context"""
        base_query = user_message
        
        # Add entity context
        entities = intent_result.get('entities', {})
        entity_terms = []
        
        for entity_type, entity_values in entities.items():
            if entity_values:
                entity_terms.extend(entity_values)
        
        if entity_terms:
            base_query += f" {' '.join(entity_terms)}"
        
        # Add conversation context
        recent_topics = conversation_state.get('topics_discussed', [])
        if recent_topics:
            base_query += f" {' '.join(recent_topics[-2:])}"  # Last 2 topics
        
        return base_query
    
    def _build_search_filters(self, intent_result: Dict[str, Any],
                            conversation_state: Dict[str, Any]) -> Dict[str, Any]:
        """Build search filters based on intent and context"""
        filters = {}
        
        # Add intent-based filters
        intent = intent_result.get('intent', {})
        intent_type = intent.get('type', '')
        
        if intent_type == 'problem_report':
            filters['has_resolution'] = True
        elif intent_type == 'how_to_request':
            filters['content_type'] = 'instructional'
        
        # Add time-based filters
        user_context = conversation_state.get('user_context', {})
        if user_context.get('prefer_recent', True):
            filters['recency_boost'] = True
        
        return filters
    
    def _determine_urgency(self, intent_result: Dict[str, Any]) -> str:
        """Determine urgency level from intent"""
        intent = intent_result.get('intent', {})
        
        if intent.get('type') == 'escalation_request':
            return 'high'
        elif intent.get('type') == 'problem_report':
            # Check for urgency indicators in entities or confidence
            entities = intent_result.get('entities', {})
            if any('urgent' in str(v).lower() or 'critical' in str(v).lower() 
                   for v in entities.values()):
                return 'high'
            return 'medium'
        else:
            return 'normal'
    
    def _requires_resolution(self, intent_result: Dict[str, Any]) -> bool:
        """Check if intent requires resolution generation"""
        intent_type = intent_result.get('intent', {}).get('type', '')
        return intent_type in ['problem_report', 'how_to_request']
    
    def _extract_incident_data_from_conversation(self, user_message: str,
                                               intent_result: Dict[str, Any],
                                               conversation_state: Dict[str, Any]) -> Dict[str, Any]:
        """Extract incident data from conversation for resolution"""
        entities = intent_result.get('entities', {})
        
        # Build incident data structure
        incident_data = {
            'id': f"conv_{conversation_state['session_id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'summary': user_message[:100],  # First 100 chars as summary
            'description': user_message,
            'category': entities.get('systems', ['General'])[0] if entities.get('systems') else 'General',
            'severity': self._determine_severity_from_intent(intent_result),
            'priority': self._determine_priority_from_urgency(self._determine_urgency(intent_result)),
            'date_submitted': datetime.now().strftime('%d-%m-%Y %H:%M'),
            'reporter': conversation_state.get('user_context', {}).get('user_id', 'chat_user')
        }
        
        return incident_data
    
    def _determine_severity_from_intent(self, intent_result: Dict[str, Any]) -> str:
        """Determine severity from intent analysis"""
        urgency = self._determine_urgency(intent_result)
        
        if urgency == 'high':
            return 'Severity 3'
        elif urgency == 'medium':
            return 'Severity 2'
        else:
            return 'Severity 1'
    
    def _determine_priority_from_urgency(self, urgency: str) -> str:
        """Determine priority from urgency level"""
        if urgency == 'high':
            return 'High'
        elif urgency == 'medium':
            return 'Medium'
        else:
            return 'Low'
    
    def _determine_response_style(self, intent_result: Dict[str, Any],
                                conversation_state: Dict[str, Any]) -> str:
        """Determine appropriate response style"""
        intent_type = intent_result.get('intent', {}).get('type', '')
        user_level = conversation_state.get('user_context', {}).get('experience_level', 'intermediate')
        
        if intent_type == 'general_chat':
            return 'conversational'
        elif intent_type == 'problem_report':
            return 'solution_focused'
        elif intent_type == 'how_to_request':
            return 'instructional'
        elif user_level == 'beginner':
            return 'detailed_explanatory'
        elif user_level == 'expert':
            return 'concise_technical'
        else:
            return 'helpful_and_clear'
    
    def _generate_fallback_response(self, user_message: str,
                                  intent_result: Dict[str, Any]) -> str:
        """Generate fallback response when synthesis fails"""
        intent_type = intent_result.get('intent', {}).get('type', 'general_chat')
        
        if intent_type == 'problem_report':
            return "I understand you're experiencing an issue. Let me search for relevant solutions and get back to you with some recommendations."
        elif intent_type == 'how_to_request':
            return "I can help you with that. Let me find the appropriate instructions or guidance for your request."
        elif intent_type == 'question':
            return "That's a good question. Let me search our knowledge base to provide you with accurate information."
        else:
            return "I'm here to help! Could you provide a bit more detail about what you'd like assistance with?"
    
    def _generate_basic_followups(self, intent_result: Dict[str, Any],
                                response_result: Dict[str, Any],
                                conversation_state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate basic follow-up suggestions"""
        intent_type = intent_result.get('intent', {}).get('type', '')
        
        followups = []
        
        if intent_type == 'problem_report':
            followups = [
                "Did this solution help resolve your issue?",
                "Would you like me to explain any of these steps in more detail?",
                "Do you need assistance with anything else?"
            ]
        elif intent_type == 'how_to_request':
            followups = [
                "Would you like more detailed instructions for any of these steps?",
                "Do you have questions about implementing this solution?",
                "Is there a related process you'd like to learn about?"
            ]
        elif intent_type == 'question':
            followups = [
                "Does this answer your question completely?",
                "Would you like me to elaborate on any part of this explanation?",
                "Do you have any related questions?"
            ]
        else:
            followups = [
                "Is there anything else I can help you with?",
                "Would you like more information about this topic?",
                "Do you have any other questions?"
            ]
        
        return {
            'follow_up_suggestions': followups,
            'conversation_state': 'awaiting_user_response',
            'suggested_actions': self._suggest_actions_based_on_intent(intent_type)
        }
    
    def _suggest_actions_based_on_intent(self, intent_type: str) -> List[str]:
        """Suggest actions based on intent type"""
        if intent_type == 'problem_report':
            return ['escalate_to_support', 'request_callback', 'search_more_solutions']
        elif intent_type == 'how_to_request':
            return ['request_demo', 'schedule_training', 'get_documentation']
        else:
            return ['ask_another_question', 'browse_topics', 'contact_support']
    
    async def _compile_conversation_response(self, session_id: str, user_message: str,
                                           intent_result: Dict[str, Any],
                                           search_result: Dict[str, Any],
                                           resolution_result: Optional[Dict[str, Any]],
                                           response_result: Dict[str, Any],
                                           followup_result: Dict[str, Any],
                                           conversation_state: Dict[str, Any]) -> Dict[str, Any]:
        """Compile final conversation response"""
        try:
            # Extract key information
            primary_response = response_result.get('response', '')
            intent = intent_result.get('intent', {})
            entities = intent_result.get('entities', {})
            search_results = search_result.get('results', [])
            confidence = response_result.get('confidence', 0.7)
            
            # Calculate response quality metrics
            quality_metrics = self._calculate_response_quality(
                intent_result, search_result, resolution_result, response_result
            )
            
            # Build comprehensive response
            final_response = {
                'session_id': session_id,
                'response': primary_response,
                'success': True,
                'conversation_metadata': {
                    'intent_recognized': intent,
                    'entities_extracted': entities,
                    'confidence_score': confidence,
                    'turn_count': conversation_state['turn_count'],
                    'session_duration': self._calculate_session_duration(conversation_state),
                    'processing_time': (datetime.now() - conversation_state['last_activity']).total_seconds()
                },
                'knowledge_sources': {
                    'search_results_count': len(search_results),
                    'top_sources': search_results[:3] if search_results else [],
                    'resolution_provided': resolution_result is not None,
                    'resolution_confidence': resolution_result.get('overall_confidence', 0) if resolution_result else 0
                },
                'interaction_guidance': {
                    'follow_up_suggestions': followup_result.get('follow_up_suggestions', []),
                    'suggested_actions': followup_result.get('suggested_actions', []),
                    'conversation_state': followup_result.get('conversation_state', 'active')
                },
                'quality_metrics': quality_metrics,
                'workflow_summary': {
                    'steps_completed': ['intent_recognition', 'contextual_search', 'response_synthesis'],
                    'resolution_generated': resolution_result is not None,
                    'agents_involved': self._get_involved_agents_summary(),
                    'processing_pipeline': 'conversation_workflow'
                },
                'timestamp': datetime.now().isoformat()
            }
            
            # Add resolution details if available
            if resolution_result:
                final_response['resolution_details'] = {
                    'solutions_provided': len(resolution_result.get('solutions', [])),
                    'implementation_guidance': bool(resolution_result.get('solutions', [{}])[0].get('implementation')),
                    'estimated_resolution_time': resolution_result.get('solutions', [{}])[0].get('estimated_time', 'Unknown')
                }
                final_response['workflow_summary']['steps_completed'].append('resolution_generation')
            
            # Add followup details
            if followup_result.get('follow_up_suggestions'):
                final_response['workflow_summary']['steps_completed'].append('followup_generation')
            
            return final_response
            
        except Exception as e:
            self.logger.error(f"Error compiling conversation response: {str(e)}")
            return {
                'session_id': session_id,
                'response': primary_response or "I apologize, but I encountered an issue while preparing my response.",
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _calculate_response_quality(self, intent_result: Dict[str, Any],
                                  search_result: Dict[str, Any],
                                  resolution_result: Optional[Dict[str, Any]],
                                  response_result: Dict[str, Any]) -> Dict[str, float]:
        """Calculate quality metrics for the response"""
        metrics = {}
        
        # Intent recognition quality
        intent_confidence = intent_result.get('intent', {}).get('confidence', 0.5)
        metrics['intent_recognition_quality'] = intent_confidence
        
        # Search quality
        search_results = search_result.get('results', [])
        if search_results:
            avg_search_score = sum(r.get('score', 0) for r in search_results) / len(search_results)
            metrics['search_quality'] = avg_search_score
        else:
            metrics['search_quality'] = 0.0
        
        # Resolution quality (if applicable)
        if resolution_result:
            resolution_confidence = resolution_result.get('overall_confidence', 0.5)
            metrics['resolution_quality'] = resolution_confidence
        
        # Response synthesis quality
        response_confidence = response_result.get('confidence', 0.7)
        metrics['response_synthesis_quality'] = response_confidence
        
        # Overall quality (weighted average)
        weights = {'intent': 0.2, 'search': 0.3, 'resolution': 0.3, 'synthesis': 0.2}
        if not resolution_result:
            # Redistribute resolution weight to search and synthesis
            weights = {'intent': 0.2, 'search': 0.4, 'synthesis': 0.4}
        
        overall_quality = (
            metrics['intent_recognition_quality'] * weights['intent'] +
            metrics['search_quality'] * weights['search'] +
            metrics.get('resolution_quality', 0) * weights.get('resolution', 0) +
            metrics['response_synthesis_quality'] * weights['synthesis']
        )
        
        metrics['overall_quality'] = overall_quality
        
        return metrics
    
    def _get_involved_agents_summary(self) -> List[str]:
        """Get summary of agents involved in conversation processing"""
        return [
            'conversation_agent',
            'search_agent',
            'resolution_agent',  # if resolution was generated
            'conversation_workflow'
        ]
    
    async def _cleanup_expired_conversations(self):
        """Background task to clean up expired conversation sessions"""
        while True:
            try:
                current_time = datetime.now()
                expired_sessions = []
                
                for session_id, conversation in self.active_conversations.items():
                    # Check for expiration
                    last_activity = conversation.get('last_activity', current_time)
                    session_age = (current_time - last_activity).total_seconds()
                    
                    if session_age > self.max_conversation_duration:
                        expired_sessions.append(session_id)
                
                # Clean up expired sessions
                for session_id in expired_sessions:
                    conversation = self.active_conversations[session_id]
                    conversation['state'] = ConversationState.TIMEOUT
                    
                    # Move to history
                    self.conversation_history.append(conversation)
                    del self.active_conversations[session_id]
                    
                    self.logger.debug(f"Cleaned up expired conversation: {session_id}")
                
                # Limit history size
                if len(self.conversation_history) > 1000:
                    self.conversation_history = self.conversation_history[-500:]  # Keep last 500
                
                await asyncio.sleep(self.session_cleanup_interval)
                
            except Exception as e:
                self.logger.error(f"Error in conversation cleanup: {str(e)}")
                await asyncio.sleep(300)  # Retry in 5 minutes
    
    # Management and status methods
    
    async def get_conversation_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a conversation session"""
        if session_id in self.active_conversations:
            conversation = self.active_conversations[session_id]
            return {
                'session_id': session_id,
                'state': conversation['state'].value,
                'turn_count': conversation['turn_count'],
                'created_at': conversation['created_at'].isoformat(),
                'last_activity': conversation['last_activity'].isoformat(),
                'duration': self._calculate_session_duration(conversation),
                'message_count': len(conversation['messages']),
                'topics_discussed': conversation.get('topics_discussed', [])
            }
        
        # Check history
        for conversation in self.conversation_history:
            if conversation['session_id'] == session_id:
                return {
                    'session_id': session_id,
                    'state': conversation['state'].value,
                    'turn_count': conversation['turn_count'],
                    'created_at': conversation['created_at'].isoformat(),
                    'last_activity': conversation['last_activity'].isoformat(),
                    'duration': self._calculate_session_duration(conversation),
                    'message_count': len(conversation['messages']),
                    'status': 'archived'
                }
        
        return None
    
    def get_active_conversations(self) -> List[Dict[str, Any]]:
        """Get list of active conversation sessions"""
        return [
            {
                'session_id': session_id,
                'state': conv['state'].value,
                'turn_count': conv['turn_count'],
                'last_activity': conv['last_activity'].isoformat(),
                'user_id': conv.get('user_context', {}).get('user_id', 'unknown')
            }
            for session_id, conv in self.active_conversations.items()
        ]
    
    def get_conversation_statistics(self) -> Dict[str, Any]:
        """Get conversation workflow statistics"""
        total_conversations = len(self.conversation_history) + len(self.active_conversations)
        
        if total_conversations == 0:
            return {'message': 'No conversations processed yet'}
        
        # Calculate statistics from history
        completed_conversations = len([c for c in self.conversation_history if c['state'] == ConversationState.COMPLETED])
        failed_conversations = len([c for c in self.conversation_history if c['state'] == ConversationState.FAILED])
        
        # Calculate average session metrics
        session_durations = [
            self._calculate_session_duration(c) for c in self.conversation_history
            if c.get('created_at') and c.get('last_activity')
        ]
        avg_duration = sum(session_durations) / len(session_durations) if session_durations else 0
        
        turn_counts = [c.get('turn_count', 0) for c in self.conversation_history]
        avg_turns = sum(turn_counts) / len(turn_counts) if turn_counts else 0
        
        return {
            'total_conversations': total_conversations,
            'active_conversations': len(self.active_conversations),
            'completed_conversations': completed_conversations,
            'failed_conversations': failed_conversations,
            'success_rate': (completed_conversations / len(self.conversation_history)) * 100 if self.conversation_history else 0,
            'average_session_duration_seconds': avg_duration,
            'average_turns_per_conversation': avg_turns,
            'conversation_history_size': len(self.conversation_history),
            'system_limits': {
                'max_active_conversations': self.max_active_conversations,
                'max_conversation_duration': self.max_conversation_duration,
                'context_window_size': self.context_window_size
            },
            'last_updated': datetime.now().isoformat()
        }
    
    async def end_conversation(self, session_id: str) -> bool:
        """Manually end a conversation session"""
        if session_id in self.active_conversations:
            conversation = self.active_conversations[session_id]
            conversation['state'] = ConversationState.COMPLETED
            conversation['last_activity'] = datetime.now()
            
            # Move to history
            self.conversation_history.append(conversation)
            del self.active_conversations[session_id]
            
            self.logger.info(f"Conversation {session_id} ended manually")
            return True
        
        return False
    
    def export_conversation_history(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Export conversation history for a session"""
        # Check active conversations
        if session_id in self.active_conversations:
            return self.active_conversations[session_id]
        
        # Check history
        for conversation in self.conversation_history:
            if conversation['session_id'] == session_id:
                return conversation
        
        return None