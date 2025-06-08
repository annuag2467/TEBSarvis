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
import aiohttp
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import time

# Add shared utilities
from ..shared.function_utils import (
    validate_environment,
    validate_json_request,
    create_error_response,
    check_rate_limit,
    setup_monitoring,
    sanitize_input,
    get_client_ip,
    AgentInitializer, 
    FunctionResponse, 
    validate_request, 
    add_request_metadata
)
from ..shared.azure_clients import AzureClientManager
from ...agents.reactive.conversation_agent import ConversationAgent
from ...agents.core.agent_registry import get_global_registry
from ...agents.core.agent_communication import MessageBus

import logging.config

# Configure structured logging
logging.config.dictConfig({
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'structured': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'structured'
        }
    },
    'root': {
        'level': os.getenv('LOG_LEVEL', 'INFO'),
        'handlers': ['console']
    }
})

# Initialize global components
app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)
logger = logging.getLogger(__name__)

# Global variables for agent instances
conversation_agent = None
azure_manager = None
message_bus = None
registry = None

# Global components
connection_pool = None
executor = None
cache = {}
cache_ttl = {}
rate_limit_cache = defaultdict(list)
RATE_LIMIT = int(os.getenv('RATE_LIMIT', '100'))  # requests per minute
CACHE_DEFAULT_TTL = 300  # 5 minutes
MAX_WORKERS = int(os.getenv('MAX_WORKERS', '4'))
USER_CONTEXT_TTL = 3600  # 1 hour - for user context caching

async def initialize_components():
    """Initialize agents and Azure components with connection pooling and monitoring"""
    global conversation_agent, azure_manager, message_bus, registry, connection_pool, executor
    
    try:
        # Validate environment variables
        validate_environment()

        # Initialize connection pool if not exists
        if not connection_pool:
            connection_pool = aiohttp.TCPConnector(
                limit=100,  # Max connections
                ttl_dns_cache=300,  # DNS cache TTL
                ssl=False  # Disable SSL for internal communications
            )
        
        # Initialize thread executor if not exists
        if not executor:
            executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

        if not conversation_agent:
            # Initialize Azure manager with connection pool
            azure_manager = AzureClientManager(connector=connection_pool)
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
            
            # Setup monitoring
            setup_monitoring(app_name="ask-assistant")
            
            logger.info("Conversation Agent components initialized successfully")
            
    except Exception as e:
        logger.error(f"Error initializing components: {str(e)}", exc_info=True)
        # Cleanup on failure
        if connection_pool:
            await connection_pool.close()
        if executor:
            executor.shutdown(wait=False)
        raise

@app.route(route="ask-assistant", methods=["POST"])
async def ask_assistant(req: func.HttpRequest) -> func.HttpResponse:
    """
    Natural language Q&A interface for IT support agents with rate limiting and caching.
    
    Expected request body:
    {
        "question": "Has this error occurred before?",
        "session_id": "user_123_session",
        "conversation_history": [...],
        "user_context": {...},
        "options": {...}
    }
    """
    request_id = req.headers.get('x-request-id', f"req_{int(time.time()*1000)}")
    client_ip = get_client_ip(req)
    start_time = time.time()
    
    try:
        # Setup request logging context
        logger.info(f"Processing conversation request {request_id} from {client_ip}")
        
        # Check rate limit
        if not check_rate_limit(client_ip, RATE_LIMIT):
            return create_error_response("Rate limit exceeded", 429)
            
        # Initialize components with retries
        retry_count = 0
        while retry_count < 3:
            try:
                await initialize_components()
                break
            except Exception as e:
                retry_count += 1
                if retry_count == 3:
                    raise
                await asyncio.sleep(1)
        
        # Validate and sanitize request
        data, error_response = validate_json_request(
            req,
            required_fields=['question'],
            optional_fields=['session_id', 'conversation_history', 'user_context', 'options']
        )
        if error_response:
            return error_response
            
        # Sanitize input data
        question = sanitize_input(data['question'])
        session_id = sanitize_input(data.get('session_id', f"session_{int(time.time()*1000)}"))
        conversation_history = sanitize_input(data.get('conversation_history', []))
        user_context = sanitize_input(data.get('user_context', {}))
        options = sanitize_input(data.get('options', {}))

        # Cache key based on question and context
        cache_key = f"{session_id}:{hash(question)}:{hash(str(conversation_history))}"
        
        # Check cache for recent identical questions in same session
        if cache_key in cache and time.time() - cache_ttl[cache_key] < CACHE_DEFAULT_TTL:
            logger.info(f"Cache hit for request {request_id}")
            cached_response = cache[cache_key]
            cached_response['conversation_metadata']['cache_hit'] = True
            return func.HttpResponse(
                json.dumps(cached_response, indent=2),
                status_code=200,
                mimetype="application/json"
            )

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
        
        # Process with conversation agent
        result = await conversation_agent.process_task(task_data)
        processing_time = time.time() - start_time
        
        # Format response
        response = {
            "response": result.get('response', ''),
            "intent": result.get('intent', {}),
            "sources": result.get('sources', []),
            "follow_up_suggestions": result.get('follow_up_suggestions', []),
            "session_id": session_id,
            "conversation_metadata": {
                "function_name": "ask-assistant",
                "processing_time_seconds": processing_time,
                "timestamp": datetime.now().isoformat(),
                "request_id": request_id,
                "api_version": "1.0.0",
                "cache_hit": False
            }
        }
        
        # Add processing metadata
        if 'processing_metadata' in result:
            response['conversation_metadata'].update(result['processing_metadata'])
        
        # Cache successful responses
        if result.get('confidence', 0.0) > 0.5:
            cache[cache_key] = response
            cache_ttl[cache_key] = time.time()
        
        logger.info(f"Conversation request {request_id} processed successfully in {processing_time:.2f}s")
        
        return func.HttpResponse(
            json.dumps(response, indent=2),
            status_code=200,
            mimetype="application/json"
        )
        
    except Exception as e:
        error_time = time.time() - start_time
        logger.error(f"Error in request {request_id} after {error_time:.2f}s: {str(e)}", exc_info=True)
        return create_error_response(
            f"Conversation processing failed: {str(e)}",
            500,
            {"request_id": request_id, "processing_time": error_time}
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
        health_tasks = []
        
        # Check agent initialization
        async def check_agent():
            try:
                await initialize_components()
                agent_status = conversation_agent.get_status() if conversation_agent else {"status": "not_initialized"}
                return {"agent": agent_status}
            except Exception as e:
                return {"agent": {"status": "unhealthy", "error": str(e)}}
        
        # Check Azure services
        async def check_azure():
            try:
                azure_status = await azure_manager.get_health_status() if azure_manager else {"status": "not_initialized"}
                return {"azure_services": azure_status}
            except Exception as e:
                return {"azure_services": {"status": "unhealthy", "error": str(e)}}
        
        # Check message bus
        async def check_message_bus():
            try:
                bus_status = await message_bus.get_status() if message_bus else {"status": "not_initialized"}
                return {"message_bus": bus_status}
            except Exception as e:
                return {"message_bus": {"status": "unhealthy", "error": str(e)}}
        
        # Check agent registry
        async def check_registry():
            try:
                registry_status = await registry.get_status() if registry else {"status": "not_initialized"}
                return {"registry": registry_status}
            except Exception as e:
                return {"registry": {"status": "unhealthy", "error": str(e)}}

        # Run all health checks in parallel
        health_tasks = [
            check_agent(),
            check_azure(),
            check_message_bus(),
            check_registry()
        ]
        health_results = await asyncio.gather(*health_tasks, return_exceptions=True)
        
        # Combine results
        health_data = {}
        for result in health_results:
            if isinstance(result, Exception):
                health_data.update({"error": str(result)})
            else:
                health_data.update(result)
        
        # Add system metrics
        health_data.update({
            "system": {
                "cache_size": len(cache),
                "active_sessions": len(set(k.split(':')[0] for k in cache.keys())),
                "connection_pool": {
                    "active": bool(connection_pool),
                    "connections": connection_pool.size if connection_pool else 0
                },
                "thread_executor": {
                    "active": bool(executor),
                    "max_workers": MAX_WORKERS
                }
            },
            "timestamp": datetime.now().isoformat()
        })
        
        # Determine overall health
        is_healthy = all(
            component.get("status", "unhealthy") == "healthy" 
            for component in health_data.values() 
            if isinstance(component, dict) and "status" in component
        )
        
        return func.HttpResponse(
            json.dumps({
                "status": "healthy" if is_healthy else "unhealthy",
                "components": health_data
            }, indent=2),
            status_code=200 if is_healthy else 503,
            mimetype="application/json"
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}", exc_info=True)
        return create_error_response(
            f"Health check failed: {str(e)}",
            500
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