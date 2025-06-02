"""
Conversation Agent for TEBSarvis Multi-Agent System
Handles natural language interactions and Q&A with context management.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import json
import re

from ..core.base_agent import BaseAgent, AgentCapability
from ..shared.azure_clients import AzureClientManager

class ConversationAgent(BaseAgent):
    """
    Conversation Agent that handles natural language interactions with users.
    Provides contextual responses, intent recognition, and multi-turn conversations.
    """
    
    def __init__(self, agent_id: str = "conversation_agent"):
        capabilities = [
            AgentCapability(
                name="natural_language_qa",
                description="Answer questions in natural language",
                input_types=["user_question", "conversation_context"],
                output_types=["natural_response", "follow_up_suggestions"],
                dependencies=["azure_openai", "search_agent"]
            ),
            AgentCapability(
                name="intent_recognition",
                description="Recognize user intent and extract entities",
                input_types=["user_input"],
                output_types=["intent", "entities"],
                dependencies=["azure_openai"]
            ),
            AgentCapability(
                name="conversation_management",
                description="Manage multi-turn conversation state",
                input_types=["conversation_history"],
                output_types=["conversation_state", "context_summary"],
                dependencies=[]
            ),
            AgentCapability(
                name="response_generation",
                description="Generate contextual responses with citations",
                input_types=["user_query", "knowledge_context"],
                output_types=["contextual_response"],
                dependencies=["azure_openai", "search_agent"]
            )
        ]
        
        super().__init__(agent_id, "conversation", capabilities)
        
        self.azure_manager = AzureClientManager()
        self.conversation_sessions = {}  # session_id -> conversation_data
        self.session_timeout = 3600  # 1 hour
        self.max_context_length = 4000  # tokens
        self.intent_patterns = self._load_intent_patterns()
        
        # Initialize agent
        asyncio.create_task(self._initialize())
    
    async def _initialize(self):
        """Initialize the conversation agent"""
        try:
            await self.azure_manager.initialize()
            # Start session cleanup task
            asyncio.create_task(self._cleanup_expired_sessions())
            self.logger.info("Conversation Agent initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize Conversation Agent: {str(e)}")
            raise
    
    def get_capabilities(self) -> List[AgentCapability]:
        """Return agent capabilities"""
        return self.capabilities
    
    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process conversation tasks.
        
        Args:
            task: Task data containing conversation parameters
            
        Returns:
            Conversation response with context
        """
        task_type = task.get('type', 'natural_language_qa')
        
        if task_type == 'natural_language_qa':
            return await self._handle_qa_request(task)
        elif task_type == 'intent_recognition':
            return await self._recognize_intent(task)
        elif task_type == 'conversation_management':
            return await self._manage_conversation_state(task)
        elif task_type == 'response_generation':
            return await self._generate_contextual_response(task)
        elif task_type == 'conversation_analysis':
            return await self._analyze_conversation(task)
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    async def _handle_qa_request(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle natural language Q&A requests.
        
        Args:
            task: Task containing user question and context
            
        Returns:
            Dictionary with response and metadata
        """
        try:
            user_question = task.get('question', '')
            session_id = task.get('session_id', 'default')
            user_context = task.get('user_context', {})
            conversation_history = task.get('conversation_history', [])
            
            if not user_question.strip():
                return {
                    'response': 'I need a question to help you with.',
                    'error': 'Empty question'
                }
            
            # Get or create conversation session
            session = self._get_or_create_session(session_id, user_context)
            
            # Update conversation history
            session['history'].append({
                'role': 'user',
                'content': user_question,
                'timestamp': datetime.now().isoformat()
            })
            
            # Recognize intent and extract entities
            intent_result = await self._recognize_user_intent(user_question)
            
            # Generate response based on intent
            response_data = await self._generate_intent_based_response(
                user_question, intent_result, session, conversation_history
            )
            
            # Add response to session history
            session['history'].append({
                'role': 'assistant',
                'content': response_data['response'],
                'timestamp': datetime.now().isoformat(),
                'metadata': response_data.get('metadata', {})
            })
            
            # Update session metadata
            session['last_activity'] = datetime.now()
            session['turn_count'] += 1
            
            return {
                'response': response_data['response'],
                'intent': intent_result,
                'sources': response_data.get('sources', []),
                'follow_up_suggestions': response_data.get('follow_up_suggestions', []),
                'session_id': session_id,
                'conversation_metadata': {
                    'agent_id': self.agent_id,
                    'timestamp': datetime.now().isoformat(),
                    'turn_count': session['turn_count'],
                    'session_duration': self._calculate_session_duration(session)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error handling Q&A request: {str(e)}")
            raise

    async def _cleanup_expired_sessions(self):
        """Clean up expired conversation sessions"""
        while True:
            try:
                current_time = datetime.now()
                expired_sessions = []
                
                for session_id, session in self.conversation_sessions.items():
                    last_activity = session['last_activity']
                    if (current_time - last_activity).total_seconds() > self.session_timeout:
                        expired_sessions.append(session_id)
                
                for session_id in expired_sessions:
                    del self.conversation_sessions[session_id]
                    self.logger.debug(f"Cleaned up expired session: {session_id}")
                
                # Clean up every 30 minutes
                await asyncio.sleep(1800)
                
            except Exception as e:
                self.logger.error(f"Error in session cleanup: {str(e)}")
                await asyncio.sleep(300)  # Retry in 5 minutes

    async def _recognize_intent(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recognize user intent and extract entities.
        
        Args:
            task: Task containing user input
            
        Returns:
            Dictionary with intent and entities
        """
        try:
            user_input = task.get('user_input', '')
            context = task.get('context', {})
            
            # Use both pattern matching and AI for intent recognition
            pattern_intent = self._match_intent_patterns(user_input)
            ai_intent = await self._recognize_intent_with_ai(user_input, context)
            
            # Combine results
            final_intent = self._combine_intent_results(pattern_intent, ai_intent)
            
            return {
                'intent': final_intent,
                'confidence': final_intent.get('confidence', 0.0),
                'entities': final_intent.get('entities', {}),
                'processing_metadata': {
                    'agent_id': self.agent_id,
                    'timestamp': datetime.now().isoformat(),
                    'methods_used': ['pattern_matching', 'ai_classification']
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error recognizing intent: {str(e)}")
            raise
    
    async def _generate_contextual_response(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate contextual response with knowledge retrieval.
        
        Args:
            task: Task containing query and context requirements
            
        Returns:
            Dictionary with contextual response
        """
        try:
            user_query = task.get('user_query', '')
            knowledge_sources = task.get('knowledge_sources', ['incidents', 'resolutions'])
            max_context_items = task.get('max_context_items', 5)
            response_style = task.get('response_style', 'helpful')
            
            # Retrieve relevant knowledge
            knowledge_context = await self._retrieve_knowledge_context(
                user_query, knowledge_sources, max_context_items
            )
            
            # Generate response with retrieved context
            response = await self._generate_response_with_rag(
                user_query, knowledge_context, response_style
            )
            
            return {
                "response" : response
            }
        except Exception as e:
            self.logger.error(f"Error generating contextual response: {str(e)}")
            raise