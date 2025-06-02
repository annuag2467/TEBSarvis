"""
Azure Function App for Conversation Agent API
Endpoint: /ask-assistant
Provides natural language Q&A interface with context management.
"""

import azure.functions as func
import logging
import json
import asyncio
import sys
import os
from datetime import datetime
from typing import Dict, Any, Optional, List

# Add the backend path to sys.path to import our agents
backend_path = os.path.join(os.path.dirname(__file__), '..', '..', 'agents')
sys.path.append(backend_path)

from reactive.conversation_agent import ConversationAgent
from core.agent_registry import get_global_registry
from core.agent_communication import MessageBus
from shared.azure_clients import AzureClientManager

# Initialize global components
app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)
logger = logging.getLogger(__name__)

# Global variables for agent instances
conversation_agent = None
azure_manager = None
message_bus = None
registry = None

async def initialize_components():
    """Initialize agents and Azure components"""
    global conversation_agent, azure_manager, message_bus, registry
    
    try:
        if not conversation_agent:
            # Initialize Azure manager
            azure_manager = AzureClientManager()
            await azure_manager.initialize()
            
            # Initialize message bus and registry
            message_bus = MessageBus()
            await message_bus.start()
            
            registry = get_global_registry()
            await registry.start()
            
            # Initialize conversation agent
            conversation_agent = ConversationAgent()
            await conversation_agent.start()
            
            # Register agent
            await registry.register_agent(conversation_agent)
            
            logger.info("Conversation Agent components initialized successfully")
            
    except Exception as e:
        logger.error(f"Error initializing components: {str(e)}")
        raise

@app.route(route="ask-assistant", methods=["POST"])
async def ask_assistant(req: func.HttpRequest) -> func.HttpResponse:
    """
    Natural language Q&A interface for IT support agents.
    
    Expected request body:
    {
        "question": "Has this error occurred before?",
        "session_id": "user_123_session",
        "conversation_history": [
            {"role": "user", "content": "Previous question"},
            {"role": "assistant", "content": "Previous response"}
        ],
        "user_context": {
            "user_id": "agent_123",
            "role": "L1_support",
            "experience_level": "beginner"
        },
        "options": {
            "response_style": "helpful",
            "include_sources": true,
            "max_context_items": 5
        }
    }
    
    Returns:
    {
        "response": "Based on our incident history...",
        "intent": {
            "intent": "search_history",
            "confidence": 0.9,
            "entities": {...}
        },
        "sources": [...],
        "follow_up_suggestions": [...],
        "session_id": "...",
        "conversation_metadata": {...}
    }
    """
    
    logger.info("Ask assistant request received")
    
    try:
        # Initialize components if needed
        await initialize_components()
        
        # Validate request
        if not req.get_body():
            return func.HttpResponse(
                json.dumps({"error": "Request body is required"}),
                status_code=400,
                mimetype="application/json"
            )
        
        # Parse request body
        try:
            request_data = req.get_json()
        except ValueError as e:
            return func.HttpResponse(
                json.dumps({"error": f"Invalid JSON: {str(e)}"}),
                status_code=400,
                mimetype="application/json"
            )
        
        # Validate required fields
        if 'question' not in request_data:
            return func.HttpResponse(
                json.dumps({"error": "question is required"}),
                status_code=400,
                mimetype="application/json"
            )
        
        question = request_data['question']
        session_id = request_data.get('session_id', f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        conversation_history = request_data.get('conversation_history', [])
        user_context = request_data.get('user_context', {})
        options = request_data.get('options', {})
        
        # Prepare task for conversation agent
        task_data = {
            'type': 'natural_language_qa',
            'question': question,
            'session_id': session_id,
            'conversation_history': conversation_history,
            'user_context': user_context,
            'options': {
                'response_style': options.get('response_style', 'helpful'),
                'include_sources': options.get('include_sources', True),
                'max_context_items': options.get('max_context_items', 5),
                'follow_up_suggestions': options.get('follow_up_suggestions', True)
            }
        }
        
        # Process the conversation request
        start_time = datetime.now()
        result = await conversation_agent.process_task(task_data)
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Add processing metadata
        result['conversation_metadata'] = result.get('conversation_metadata', {})
        result['conversation_metadata'].update({
            'function_name': 'ask-assistant',
            'processing_time_seconds': processing_time,
            'timestamp': datetime.now().isoformat(),
            'api_version': '1.0.0'
        })
        
        # Add request tracking
        result['request_id'] = req.headers.get('x-request-id', 'unknown')
        
        logger.info(f"Conversation request processed successfully in {processing_time:.2f}s")
        
        return func.HttpResponse(
            json.dumps(result, indent=2),
            status_code=200,
            mimetype="application/json"
        )
        
    except Exception as e:
        logger.error(f"Error processing conversation request: {str(e)}")
        
        error_response = {
            "error": "Internal server error",
            "message": str(e),
            "timestamp": datetime.now().isoformat(),
            "function_name": "ask-assistant"
        }
        
        return func.HttpResponse(
            json.dumps(error_response),
            status_code=500,
            mimetype="application/json"
        )

@app.route(route="intent-recognition", methods=["POST"])
async def recognize_intent(req: func.HttpRequest) -> func.HttpResponse:
    """
    Recognize user intent and extract entities from natural language input.
    
    Expected request body:
    {
        "user_input": "How do I reset a user's password in the LMS?",
        "context": {
            "previous_intents": [],
            "user_role": "L1_support"
        }
    }
    """
    
    logger.info("Intent recognition request received")
    
    try:
        await initialize_components()
        
        # Parse request
        request_data = req.get_json()
        if not request_data:
            return func.HttpResponse(
                json.dumps({"error": "Request body is required"}),
                status_code=400,
                mimetype="application/json"
            )
        
        # Validate required fields
        if 'user_input' not in request_data:
            return func.HttpResponse(
                json.dumps({"error": "user_input is required"}),
                status_code=400,
                mimetype="application/json"
            )
        
        # Prepare intent recognition task
        task_data = {
            'type': 'intent_recognition',
            'user_input': request_data['user_input'],
            'context': request_data.get('context', {})
        }
        
        # Process intent recognition
        result = await conversation_agent.process_task(task_data)
        
        return func.HttpResponse(
            json.dumps(result, indent=2),
            status_code=200,
            mimetype="application/json"
        )
        
    except Exception as e:
        logger.error(f"Error in intent recognition: {str(e)}")
        
        error_response = {
            "error": "Intent recognition failed",
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }
        
        return func.HttpResponse(
            json.dumps(error_response),
            status_code=500,
            mimetype="application/json"
        )

@app.route(route="conversation-analysis", methods=["POST"])
async def analyze_conversation(req: func.HttpRequest) -> func.HttpResponse:
    """
    Analyze conversation patterns and provide insights.
    
    Expected request body:
    {
        "conversation_history": [...],
        "analysis_type": "sentiment",  // "sentiment", "satisfaction", "complexity"
        "user_context": {...}
    }
    """
    
    logger.info("Conversation analysis request received")
    
    try:
        await initialize_components()
        
        # Parse request
        request_data = req.get_json()
        if not request_data:
            return func.HttpResponse(
                json.dumps({"error": "Request body is required"}),
                status_code=400,
                mimetype="application/json"
            )
        
        # Validate required fields
        if 'conversation_history' not in request_data:
            return func.HttpResponse(
                json.dumps({"error": "conversation_history is required"}),
                status_code=400,
                mimetype="application/json"
            )
        
        # Prepare conversation analysis task
        task_data = {
            'type': 'conversation_analysis',
            'conversation_history': request_data['conversation_history'],
            'analysis_type': request_data.get('analysis_type', 'sentiment'),
            'user_context': request_data.get('user_context', {})
        }
        
        # Process conversation analysis
        result = await conversation_agent.process_task(task_data)
        
        return func.HttpResponse(
            json.dumps(result, indent=2),
            status_code=200,
            mimetype="application/json"
        )
        
    except Exception as e:
        logger.error(f"Error in conversation analysis: {str(e)}")
        
        error_response = {
            "error": "Conversation analysis failed",
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }
        
        return func.HttpResponse(
            json.dumps(error_response),
            status_code=500,
            mimetype="application/json"
        )

@app.route(route="conversation-health", methods=["GET"])
async def conversation_health(req: func.HttpRequest) -> func.HttpResponse:
    """
    Health check endpoint for the conversation service.
    """
    
    try:
        await initialize_components()
        
        # Check agent status
        agent_status = conversation_agent.get_status() if conversation_agent else {"status": "not_initialized"}
        
        # Check Azure services
        azure_health = await azure_manager.get_health_status() if azure_manager else {"status": "not_initialized"}
        
        # Test conversation functionality
        conversation_test = {"status": "not_tested"}
        if conversation_agent:
            try:
                test_task = {
                    'type': 'natural_language_qa',
                    'question': 'Hello, test question',
                    'session_id': 'health_check',
                    'conversation_history': [],
                    'user_context': {},
                    'options': {'response_style': 'helpful'}
                }
                test_result = await conversation_agent.process_task(test_task)
                conversation_test = {"status": "healthy", "test_result": "success"}
            except Exception as e:
                conversation_test = {"status": "unhealthy", "error": str(e)}
        
        health_data = {
            "status": "healthy" if conversation_agent else "unhealthy",
            "agent_status": agent_status,
            "azure_services": azure_health,
            "conversation_functionality": conversation_test,
            "active_sessions": len(conversation_agent.conversation_sessions) if conversation_agent else 0,
            "timestamp": datetime.now().isoformat(),
            "uptime": "available" if conversation_agent else "unavailable"
        }
        
        status_code = 200 if conversation_agent else 503
        
        return func.HttpResponse(
            json.dumps(health_data, indent=2),
            status_code=status_code,
            mimetype="application/json"
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        
        error_response = {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }
        
        return func.HttpResponse(
            json.dumps(error_response),
            status_code=503,
            mimetype="application/json"
        )

@app.route(route="conversation-sessions", methods=["GET"])
async def get_conversation_sessions(req: func.HttpRequest) -> func.HttpResponse:
    """
    Get active conversation sessions and statistics.
    """
    
    try:
        await initialize_components()
        
        if not conversation_agent:
            return func.HttpResponse(
                json.dumps({"error": "Conversation agent not initialized"}),
                status_code=503,
                mimetype="application/json"
            )
        
        # Get session statistics
        sessions_data = {
            "active_sessions": len(conversation_agent.conversation_sessions),
            "total_sessions_created": getattr(conversation_agent, 'total_sessions', 0),
            "session_timeout_minutes": conversation_agent.session_timeout // 60,
            "max_context_length": conversation_agent.max_context_length,
            "timestamp": datetime.now().isoformat()
        }
        
        # Include session summaries if requested
        include_details = req.params.get('include_details', 'false').lower() == 'true'
        if include_details:
            session_summaries = []
            for session_id, session_data in conversation_agent.conversation_sessions.items():
                summary = {
                    "session_id": session_id,
                    "created_at": session_data.get('created_at', 'unknown'),
                    "last_activity": session_data.get('last_activity', 'unknown'),
                    "turn_count": session_data.get('turn_count', 0),
                    "user_context": session_data.get('user_context', {})
                }
                session_summaries.append(summary)
            
            sessions_data["session_summaries"] = session_summaries
        
        return func.HttpResponse(
            json.dumps(sessions_data, indent=2),
            status_code=200,
            mimetype="application/json"
        )
        
    except Exception as e:
        logger.error(f"Error getting conversation sessions: {str(e)}")
        
        error_response = {
            "error": "Failed to get sessions",
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }
        
        return func.HttpResponse(
            json.dumps(error_response),
            status_code=500,
            mimetype="application/json"
        )

# Graceful shutdown handler
@app.function_name("cleanup")
async def cleanup_resources():
    """Clean up resources on function app shutdown"""
    global conversation_agent, message_bus, registry
    
    try:
        if conversation_agent:
            await conversation_agent.stop()
        if registry:
            await registry.stop()
        if message_bus:
            await message_bus.stop()
            
        logger.info("Conversation service resources cleaned up successfully")
        
    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}")