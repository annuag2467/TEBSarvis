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
from ...azure-functions.shared.azure_clients import AzureClientManager

class ConversationAgent(BaseAgent):
    """
    Conversation Agent that handles natural language interactions with users.
    Provides contextual responses, intent recognition, and multi-turn conversations.
    """
    
    def __init__(self, agent_id: str = "conversation_agent", agent_system=None):
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


        config_manager = get_agent_config()
        agent_config = config_manager.get_agent_config(AgentType.CONVERSATION)
        capabilities = agent_config.get('capabilities', [])
        super().__init__(agent_id, "conversation", capabilities, agent_system)
        performance_config = config_manager.get_agent_performance_config(AgentType.CONVERSATION)
        self.max_concurrent_tasks = performance_config.max_concurrent_tasks
        self.task_timeout = performance_config.task_timeout
        self.azure_manager = AzureClientManager()
        self.conversation_sessions = {}  # session_id -> conversation_data
        self.session_timeout = 3600  # 1 hour
        self.max_context_length = 4000  # tokens
        self.intent_patterns = self._load_intent_patterns()
        
        # Initialize agent
        asyncio.create_task(self._initialize())
    
    
    async def _initialize(self):
    """Initialize the agent with proper error handling"""
    try:
        # Initialize Azure manager
        await self.azure_manager.initialize()
        asyncio.create_task(self._cleanup_expired_sessions())
        # Verify Azure services are accessible
        health_status = await self.azure_manager.get_health_status()
        
        unhealthy_services = [
            service for service, status in health_status.items() 
            if status != 'healthy' and service != 'timestamp'
        ]
        
        if unhealthy_services:
            self.logger.warning(f"Some Azure services are unhealthy: {unhealthy_services}")
        
        self.logger.info(f"{self.agent_type.title()} Agent initialized successfully")
        
    except Exception as e:
        self.logger.error(f"Failed to initialize {self.agent_type} Agent: {str(e)}")
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
                'response': response['response'],
                'sources': knowledge_context.get('sources', []),
                'confidence': response.get('confidence', 0.7),
                'context_used': len(knowledge_context.get('documents', [])),
                'generation_metadata': {
                    'agent_id': self.agent_id,
                    'timestamp': datetime.now().isoformat(),
                    'response_style': response_style,
                    'knowledge_sources': knowledge_sources
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error generating contextual response: {str(e)}")
            raise

    async def _manage_conversation_state(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Manage conversation state and context.
        
        Args:
            task: Task containing conversation management parameters
            
        Returns:
            Dictionary with conversation state
        """
        try:
            session_id = task.get('session_id', 'default')
            action = task.get('action', 'get_state')  # get_state, update_state, clear_state
            
            session = self._get_or_create_session(session_id, {})
            
            if action == 'get_state':
                return {
                    'session_id': session_id,
                    'conversation_state': {
                        'turn_count': session['turn_count'],
                        'last_activity': session['last_activity'].isoformat(),
                        'history_length': len(session['history']),
                        'context_summary': self._generate_context_summary(session['history'])
                    },
                    'metadata': {
                        'agent_id': self.agent_id,
                        'timestamp': datetime.now().isoformat()
                    }
                }
            
            elif action == 'update_state':
                updates = task.get('updates', {})
                session.update(updates)
                session['last_activity'] = datetime.now()
                
                return {
                    'session_id': session_id,
                    'updated': True,
                    'new_state': {
                        'turn_count': session['turn_count'],
                        'last_activity': session['last_activity'].isoformat()
                    }
                }
            
            elif action == 'clear_state':
                if session_id in self.conversation_sessions:
                    del self.conversation_sessions[session_id]
                
                return {
                    'session_id': session_id,
                    'cleared': True,
                    'timestamp': datetime.now().isoformat()
                }
            
            else:
                raise ValueError(f"Unknown action: {action}")
                
        except Exception as e:
            self.logger.error(f"Error managing conversation state: {str(e)}")
            raise

    async def _analyze_conversation(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze conversation patterns and metrics.
        
        Args:
            task: Task containing analysis parameters
            
        Returns:
            Dictionary with conversation analysis
        """
        try:
            session_id = task.get('session_id')
            analysis_type = task.get('analysis_type', 'summary')  # summary, sentiment, topics
            
            if session_id and session_id in self.conversation_sessions:
                session = self.conversation_sessions[session_id]
                conversation_history = session['history']
            else:
                conversation_history = task.get('conversation_history', [])
            
            analysis_results = {}
            
            if analysis_type in ['summary', 'all']:
                analysis_results['summary'] = self._analyze_conversation_summary(conversation_history)
            
            if analysis_type in ['sentiment', 'all']:
                analysis_results['sentiment'] = await self._analyze_conversation_sentiment(conversation_history)
            
            if analysis_type in ['topics', 'all']:
                analysis_results['topics'] = self._analyze_conversation_topics(conversation_history)
            
            if analysis_type in ['patterns', 'all']:
                analysis_results['patterns'] = self._analyze_conversation_patterns(conversation_history)
            
            return {
                'analysis_type': analysis_type,
                'session_id': session_id,
                'results': analysis_results,
                'conversation_length': len(conversation_history),
                'analysis_metadata': {
                    'agent_id': self.agent_id,
                    'timestamp': datetime.now().isoformat(),
                    'analysis_methods': list(analysis_results.keys())
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing conversation: {str(e)}")
            raise




    def _get_or_create_session(self, session_id: str, user_context: Dict[str, Any]) -> Dict[str, Any]:
        """Get existing session or create new one"""
        if session_id not in self.conversation_sessions:
            self.conversation_sessions[session_id] = {
                'session_id': session_id,
                'created_at': datetime.now(),
                'last_activity': datetime.now(),
                'turn_count': 0,
                'history': [],
                'user_context': user_context,
                'session_metadata': {}
            }
        
        return self.conversation_sessions[session_id]

    async def _recognize_user_intent(self, user_input: str) -> Dict[str, Any]:
        """Recognize user intent from input"""
        try:
            # Use pattern matching first
            pattern_intent = self._match_intent_patterns(user_input)
            
            # If pattern matching gives good confidence, use it
            if pattern_intent.get('confidence', 0) > 0.8:
                return pattern_intent
            
            # Otherwise, use AI for intent recognition
            return await self._recognize_intent_with_ai(user_input, {})
            
        except Exception as e:
            self.logger.error(f"Error recognizing intent: {str(e)}")
            return {'intent': 'general_query', 'confidence': 0.5, 'entities': {}}

    def _match_intent_patterns(self, user_input: str) -> Dict[str, Any]:
        """Match user input against predefined intent patterns"""
        user_input_lower = user_input.lower()
        
        # Define intent patterns
        patterns = {
            'incident_resolution': {
                'patterns': ['how to fix', 'resolve', 'solution', 'problem with'],
                'confidence': 0.9
            },
            'search_similar': {
                'patterns': ['similar', 'like this', 'same issue', 'related'],
                'confidence': 0.8
            },
            'status_inquiry': {
                'patterns': ['status', 'what happened', 'update on'],
                'confidence': 0.7
            },
            'help_request': {
                'patterns': ['help', 'assist', 'support', 'can you'],
                'confidence': 0.6
            }
        }
        
        best_match = {'intent': 'general_query', 'confidence': 0.3, 'entities': {}}
        
        for intent, config in patterns.items():
            for pattern in config['patterns']:
                if pattern in user_input_lower:
                    if config['confidence'] > best_match['confidence']:
                        best_match = {
                            'intent': intent,
                            'confidence': config['confidence'],
                            'entities': self._extract_simple_entities(user_input),
                            'matched_pattern': pattern
                        }
        
        return best_match

    async def _recognize_intent_with_ai(self, user_input: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Use AI to recognize intent"""
        try:
            intent_prompt = f"""
            Analyze this user input and identify the intent:
            
            Input: "{user_input}"
            
            Possible intents:
            - incident_resolution: User wants help resolving a technical issue
            - search_similar: User wants to find similar cases or incidents
            - status_inquiry: User wants to know the status of something
            - information_request: User wants general information
            - help_request: User needs help or assistance
            - general_query: General conversation
            
            Respond in JSON format:
            {{
                "intent": "intent_name",
                "confidence": 0.85,
                "entities": {{
                    "system": "extracted_system_name",
                    "issue_type": "extracted_issue_type"
                }},
                "reasoning": "brief explanation"
            }}
            """
            
            response = await self.azure_manager.get_chat_completion(
                messages=[{"role": "user", "content": intent_prompt}],
                temperature=0.3,
                max_tokens=200
            )
            
            try:
                return json.loads(response)
            except json.JSONDecodeError:
                return {'intent': 'general_query', 'confidence': 0.5, 'entities': {}}
                
        except Exception as e:
            self.logger.error(f"Error in AI intent recognition: {str(e)}")
            return {'intent': 'general_query', 'confidence': 0.5, 'entities': {}}

    def _extract_simple_entities(self, text: str) -> Dict[str, Any]:
        """Extract simple entities from text"""
        entities = {}
        
        # System names
        systems = ['lms', 'email', 'network', 'database', 'server', 'application']
        for system in systems:
            if system in text.lower():
                entities['system'] = system
                break
        
        # Issue types
        issues = ['error', 'timeout', 'crash', 'slow', 'login', 'access', 'connection']
        for issue in issues:
            if issue in text.lower():
                entities['issue_type'] = issue
                break
        
        return entities

    async def _generate_intent_based_response(self, user_question: str, intent_result: Dict[str, Any], 
                                            session: Dict[str, Any], conversation_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate response based on recognized intent"""
        try:
            intent = intent_result.get('intent', 'general_query')
            
            if intent == 'incident_resolution':
                return await self._handle_incident_resolution_intent(user_question, intent_result, session)
            elif intent == 'search_similar':
                return await self._handle_search_similar_intent(user_question, intent_result, session)
            elif intent == 'status_inquiry':
                return await self._handle_status_inquiry_intent(user_question, intent_result, session)
            elif intent == 'help_request':
                return await self._handle_help_request_intent(user_question, intent_result, session)
            else:
                return await self._handle_general_query_intent(user_question, intent_result, session)
                
        except Exception as e:
            self.logger.error(f"Error generating intent-based response: {str(e)}")
            return {
                'response': "I apologize, but I encountered an error while processing your request. Could you please try rephrasing your question?",
                'sources': [],
                'follow_up_suggestions': [],
                'metadata': {'error': str(e)}
            }

    async def _handle_incident_resolution_intent(self, user_question: str, intent_result: Dict[str, Any], 
                                            session: Dict[str, Any]) -> Dict[str, Any]:
        """Handle incident resolution requests"""
        try:
            # Extract incident details from the question
            incident_data = {
                'summary': user_question,
                'category': intent_result.get('entities', {}).get('system', 'General'),
                'description': user_question
            }
            
            # Use RAG to find relevant solutions
            knowledge_context = await self._retrieve_knowledge_context(
                user_question, ['incidents', 'resolutions'], 5
            )
            
            # Generate response with solutions
            response = await self._generate_response_with_rag(
                user_question, knowledge_context, 'solution_focused'
            )
            
            follow_up_suggestions = [
                "Would you like more detailed steps for any of these solutions?",
                "Do you need help with a specific error message?",
                "Should I search for similar resolved incidents?"
            ]
            
            return {
                'response': response['response'],
                'sources': knowledge_context.get('sources', []),
                'follow_up_suggestions': follow_up_suggestions,
                'metadata': {
                    'intent': 'incident_resolution',
                    'confidence': intent_result.get('confidence', 0.7),
                    'entities': intent_result.get('entities', {})
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error handling incident resolution intent: {str(e)}")
            return {
                'response': "I understand you're looking for help resolving an issue. Could you provide more details about the specific problem you're experiencing?",
                'sources': [],
                'follow_up_suggestions': []
            }

    async def _handle_search_similar_intent(self, user_question: str, intent_result: Dict[str, Any], 
                                        session: Dict[str, Any]) -> Dict[str, Any]:
        """Handle search for similar incidents"""
        try:
            # Search for similar incidents
            search_results = await self.azure_manager.search_similar_incidents(
                query_text=user_question,
                top_k=5
            )
            
            if search_results:
                response = "I found several similar incidents that might help:\n\n"
                for i, result in enumerate(search_results[:3], 1):
                    response += f"{i}. {result.get('content', '')[:150]}...\n"
                    if result.get('metadata', {}).get('resolution'):
                        response += f"   Resolution: {result['metadata']['resolution'][:100]}...\n"
                    response += "\n"
            else:
                response = "I couldn't find any closely similar incidents in our database. Would you like me to search for related topics or help you create a new incident report?"
            
            follow_up_suggestions = [
                "Would you like more details about any of these incidents?",
                "Should I search for incidents in a specific category?",
                "Do you want to see the complete resolution steps?"
            ]
            
            return {
                'response': response,
                'sources': search_results,
                'follow_up_suggestions': follow_up_suggestions,
                'metadata': {
                    'intent': 'search_similar',
                    'results_found': len(search_results)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error handling search similar intent: {str(e)}")
            return {
                'response': "I'll help you search for similar incidents. Could you describe the issue you're experiencing?",
                'sources': [],
                'follow_up_suggestions': []
            }

    async def _handle_general_query_intent(self, user_question: str, intent_result: Dict[str, Any], 
                                        session: Dict[str, Any]) -> Dict[str, Any]:
        """Handle general queries using RAG"""
        try:
            # Use RAG pipeline for general knowledge retrieval
            knowledge_context = await self._retrieve_knowledge_context(
                user_question, ['incidents', 'knowledge_base'], 3
            )
            
            response = await self._generate_response_with_rag(
                user_question, knowledge_context, 'conversational'
            )
            
            return {
                'response': response['response'],
                'sources': knowledge_context.get('sources', []),
                'follow_up_suggestions': self._generate_follow_up_suggestions(user_question),
                'metadata': {
                    'intent': 'general_query',
                    'knowledge_sources_used': len(knowledge_context.get('sources', []))
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error handling general query: {str(e)}")
            return {
                'response': "I'd be happy to help! Could you provide a bit more detail about what you're looking for?",
                'sources': [],
                'follow_up_suggestions': []
            }

    async def _retrieve_knowledge_context(self, query: str, knowledge_sources: List[str], 
                                        max_results: int) -> Dict[str, Any]:
        """Retrieve knowledge context for response generation"""
        try:
            # Use the search agent to find relevant information
            search_results = await self.azure_manager.search_similar_incidents(
                query_text=query,
                top_k=max_results
            )
            
            return {
                'sources': search_results,
                'documents': search_results,
                'query': query,
                'knowledge_sources': knowledge_sources
            }
            
        except Exception as e:
            self.logger.error(f"Error retrieving knowledge context: {str(e)}")
            return {'sources': [], 'documents': [], 'query': query}

    async def _generate_response_with_rag(self, user_query: str, knowledge_context: Dict[str, Any], 
                                        response_style: str) -> Dict[str, Any]:
        """Generate response using RAG pipeline"""
        try:
            # Prepare context from knowledge sources
            context_parts = []
            for doc in knowledge_context.get('documents', []):
                context_parts.append(f"- {doc.get('content', '')}")
            
            context_text = "\n".join(context_parts[:5])  # Limit context
            
            # Choose prompt template based on style
            if response_style == 'solution_focused':
                prompt = f"""Based on the following relevant information, provide a clear solution to this problem:

    Context:
    {context_text}

    Problem: {user_query}

    Please provide:
    1. A direct answer to the problem
    2. Step-by-step solution if applicable
    3. Any important considerations or warnings

    Answer:"""
            
            elif response_style == 'conversational':
                prompt = f"""You are a helpful IT support assistant. Use the following context to answer the user's question in a friendly, conversational manner:

    Context:
    {context_text}

    Question: {user_query}

    Answer:"""
            
            else:
                prompt = f"""Based on the relevant information below, please answer this question:

    Context:
    {context_text}

    Question: {user_query}

    Answer:"""
            
            # Generate response
            response = await self.azure_manager.get_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=500
            )
            
            return {
                'response': response,
                'confidence': 0.8,
                'context_used': len(knowledge_context.get('documents', []))
            }
            
        except Exception as e:
            self.logger.error(f"Error generating RAG response: {str(e)}")
            return {
                'response': "I apologize, but I'm having trouble accessing the relevant information right now. Could you try rephrasing your question?",
                'confidence': 0.3
            }

    def _generate_follow_up_suggestions(self, user_question: str) -> List[str]:
        """Generate contextual follow-up suggestions"""
        suggestions = []
        user_question_lower = user_question.lower()
        
        if any(word in user_question_lower for word in ['error', 'problem', 'issue', 'broken']):
            suggestions.extend([
                "Would you like me to search for similar issues?",
                "Do you need step-by-step troubleshooting help?",
                "Should I help you create an incident report?"
            ])
        elif any(word in user_question_lower for word in ['how', 'what', 'when', 'why']):
            suggestions.extend([
                "Would you like more detailed information?",
                "Do you need examples or use cases?",
                "Should I explain any technical terms?"
            ])
        else:
            suggestions.extend([
                "Is there anything specific you'd like me to clarify?",
                "Would you like me to search for related information?",
                "Do you have any follow-up questions?"
            ])
        
        return suggestions[:3]  # Return top 3 suggestions

    def _generate_context_summary(self, conversation_history: List[Dict[str, Any]]) -> str:
        """Generate a summary of conversation context"""
        if not conversation_history:
            return "No conversation history"
        
        total_messages = len(conversation_history)
        user_messages = [msg for msg in conversation_history if msg.get('role') == 'user']
        
        if user_messages:
            recent_topics = []
            for msg in user_messages[-3:]:  # Last 3 user messages
                content = msg.get('content', '')
                if len(content) > 50:
                    recent_topics.append(content[:50] + "...")
                else:
                    recent_topics.append(content)
            
            return f"Conversation with {total_messages} messages. Recent topics: {'; '.join(recent_topics)}"
        
        return f"Conversation with {total_messages} messages"

    def _calculate_session_duration(self, session: Dict[str, Any]) -> float:
        """Calculate session duration in minutes"""
        try:
            created_at = session.get('created_at', datetime.now())
            now = datetime.now()
            duration = (now - created_at).total_seconds() / 60
            return round(duration, 2)
        except:
            return 0.0

    # Additional helper methods for conversation analysis:

    def _analyze_conversation_summary(self, conversation_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze conversation for summary statistics"""
        if not conversation_history:
            return {'total_messages': 0, 'summary': 'No conversation data'}
        
        user_messages = [msg for msg in conversation_history if msg.get('role') == 'user']
        assistant_messages = [msg for msg in conversation_history if msg.get('role') == 'assistant']
        
        return {
            'total_messages': len(conversation_history),
            'user_messages': len(user_messages),
            'assistant_messages': len(assistant_messages),
            'average_message_length': sum(len(msg.get('content', '')) for msg in conversation_history) / len(conversation_history),
            'conversation_topics': self._extract_conversation_topics(conversation_history)
        }

    async def _analyze_conversation_sentiment(self, conversation_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze sentiment of conversation"""
        try:
            if not conversation_history:
                return {'sentiment': 'neutral', 'confidence': 0.5}
            
            # Simple sentiment analysis based on keywords
            positive_words = ['thank', 'great', 'helpful', 'good', 'excellent', 'solved']
            negative_words = ['problem', 'issue', 'error', 'broken', 'frustrated', 'urgent']
            
            positive_count = 0
            negative_count = 0
            
            for msg in conversation_history:
                if msg.get('role') == 'user':
                    content = msg.get('content', '').lower()
                    positive_count += sum(1 for word in positive_words if word in content)
                    negative_count += sum(1 for word in negative_words if word in content)
            
            if positive_count > negative_count:
                sentiment = 'positive'
                confidence = min(0.9, 0.5 + (positive_count - negative_count) * 0.1)
            elif negative_count > positive_count:
                sentiment = 'negative'
                confidence = min(0.9, 0.5 + (negative_count - positive_count) * 0.1)
            else:
                sentiment = 'neutral'
                confidence = 0.6
            
            return {
                'sentiment': sentiment,
                'confidence': confidence,
                'positive_indicators': positive_count,
                'negative_indicators': negative_count
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing sentiment: {str(e)}")
            return {'sentiment': 'neutral', 'confidence': 0.5}

    def _analyze_conversation_topics(self, conversation_history: List[Dict[str, Any]]) -> List[str]:
        """Extract main topics from conversation"""
        topics = set()
        
        # Technical terms that might indicate topics
        technical_terms = {
            'lms': 'Learning Management System',
            'email': 'Email System',
            'network': 'Network Issues',
            'database': 'Database Problems',
            'login': 'Authentication',
            'password': 'Password Issues',
            'server': 'Server Problems',
            'application': 'Application Issues'
        }
        
        for msg in conversation_history:
            if msg.get('role') == 'user':
                content = msg.get('content', '').lower()
                for term, topic in technical_terms.items():
                    if term in content:
                        topics.add(topic)
        
        return list(topics)

    def _analyze_conversation_patterns(self, conversation_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze patterns in conversation flow"""
        if not conversation_history:
            return {}
        
        patterns = {
            'question_count': 0,
            'solution_requests': 0,
            'clarification_requests': 0,
            'repeat_questions': 0
        }
        
        question_words = ['how', 'what', 'why', 'when', 'where', 'can', 'could', 'would']
        solution_words = ['fix', 'resolve', 'solution', 'help']
        clarification_words = ['mean', 'explain', 'clarify', 'understand']
        
        previous_questions = []
        
        for msg in conversation_history:
            if msg.get('role') == 'user':
                content = msg.get('content', '').lower()
                
                # Count questions
                if any(word in content for word in question_words) or content.endswith('?'):
                    patterns['question_count'] += 1
                    
                    # Check for repeat questions
                    if content in previous_questions:
                        patterns['repeat_questions'] += 1
                    previous_questions.append(content)
                
                # Count solution requests
                if any(word in content for word in solution_words):
                    patterns['solution_requests'] += 1
                
                # Count clarification requests
                if any(word in content for word in clarification_words):
                    patterns['clarification_requests'] += 1
        
        return patterns