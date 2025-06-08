"""
Azure Function App for Search Agent API
Endpoint: /search-similar-incidents
Provides semantic search, vector search, and hybrid search for incidents.
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
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import time
from collections import defaultdict
import html
import re

# FIXED IMPORTS
from ...agents.reactive.search_agent import SearchAgent
from ...agents.core.agent_registry import get_global_registry
from ...agents.core.agent_communication import MessageBus
from ..shared.azure_clients import AzureClientManager

import logging.config

# Global connection pool and thread executor
connection_pool = None
executor = None

# Simple in-memory cache with TTL
cache = {}
cache_ttl = {}

# Simple rate limiter
rate_limit_cache = defaultdict(list)

def get_cached_result(key: str, ttl_seconds: int = 300):
    """Get result from cache if not expired"""
    if key in cache:
        if time.time() - cache_ttl.get(key, 0) < ttl_seconds:
            return cache[key]
        else:
            # Cache expired
            del cache[key]
            del cache_ttl[key]
    return None

def set_cached_result(key: str, value, ttl_seconds: int = 300):
    """Store result in cache with TTL"""
    cache[key] = value
    cache_ttl[key] = time.time()

def create_error_response(error_msg: str, status_code: int = 500, details: dict = None):
    """Create standardized error response"""
    error_response = {
        "error": True,
        "message": error_msg,
        "timestamp": datetime.now().isoformat(),
        "status_code": status_code
    }
    
    if details:
        error_response["details"] = details
    
    # Log error for monitoring
    logger.error(f"API Error {status_code}: {error_msg}", extra=details or {})
    
    return func.HttpResponse(
        json.dumps(error_response),
        status_code=status_code,
        mimetype="application/json"
    )

def validate_json_request(req: func.HttpRequest, required_fields: list = None) -> tuple:
    """Validate JSON request and return data or error response"""
    try:
        if not req.get_body():
            return None, create_error_response("Request body is required", 400)
        
        data = req.get_json()
        if not data:
            return None, create_error_response("Invalid JSON in request body", 400)
        
        if required_fields:
            missing = [field for field in required_fields if field not in data]
            if missing:
                return None, create_error_response(
                    f"Missing required fields: {missing}", 400
                )
        
        return data, None
        
    except ValueError as e:
        return None, create_error_response(f"JSON parse error: {str(e)}", 400)

def check_rate_limit(client_ip: str, requests_per_minute: int = 60) -> bool:
    """Simple rate limiting by IP"""
    now = time.time()
    minute_ago = now - 60
    
    # Clean old requests
    rate_limit_cache[client_ip] = [
        req_time for req_time in rate_limit_cache[client_ip] 
        if req_time > minute_ago
    ]
    
    # Check rate limit
    if len(rate_limit_cache[client_ip]) >= requests_per_minute:
        return False
    
    # Add current request
    rate_limit_cache[client_ip].append(now)
    return True

def sanitize_input(text: str, max_length: int = 1000) -> str:
    """Sanitize text input"""
    if not isinstance(text, str):
        return str(text)[:max_length]
    
    # Remove HTML tags and escape HTML
    text = re.sub(r'<[^>]+>', '', text)
    text = html.escape(text)
    
    # Limit length
    return text[:max_length]

def sanitize_request_data(data: dict) -> dict:
    """Recursively sanitize request data"""
    if isinstance(data, dict):
        return {k: sanitize_request_data(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [sanitize_request_data(item) for item in data]
    elif isinstance(data, str):
        return sanitize_input(data)
    else:
        return data

def validate_environment():
    """Validate required environment variables"""
    required_vars = [
        'AZURE_OPENAI_ENDPOINT',
        'AZURE_OPENAI_KEY',
        'COSMOS_DB_URL',
        'COSMOS_DB_KEY',
        'SEARCH_SERVICE_ENDPOINT',
        'SEARCH_API_KEY'
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {missing_vars}")
        raise EnvironmentError(f"Missing environment variables: {missing_vars}")
    
    logger.info("Environment validation passed")

def record_custom_metric(metric_name: str, value: float, properties: dict = None):
    """Record custom metric for Application Insights"""
    try:
        from opencensus.ext.azure import metrics_exporter
        from opencensus.stats import aggregation as aggregation_module
        from opencensus.stats import measure as measure_module
        from opencensus.stats import stats as stats_module
        from opencensus.stats import view as view_module
        from opencensus.tags import tag_map as tag_map_module
        
        # This would integrate with Application Insights
        # For now, just log the metric
        logger.info(f"Metric: {metric_name} = {value}", extra=properties or {})
        
    except ImportError:
        # Fallback to simple logging
        logger.info(f"Metric: {metric_name} = {value}", extra=properties or {})

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
search_agent = None
azure_manager = None
message_bus = None
registry = None

async def initialize_components():
    """Initialize agents and Azure components"""
    global search_agent, azure_manager, message_bus, registry, connection_pool, executor
    
    try:
        # Validate environment variables first
        validate_environment()

        if not search_agent:
            # Initialize connection pool for better performance
            if not connection_pool:
                connection_pool = aiohttp.TCPConnector(
                    limit=100,
                    limit_per_host=30,
                    keepalive_timeout=30
                )
            
            # Thread pool for CPU-bound operations
            if not executor:
                executor = ThreadPoolExecutor(max_workers=4)
            
            # Initialize Azure manager
            azure_manager = AzureClientManager()
            await azure_manager.initialize()
            
            # Initialize message bus and registry
            message_bus = MessageBus()
            await message_bus.start()
            
            registry = get_global_registry()
            await registry.start()
            
            # Initialize search agent
            search_agent = SearchAgent()
            await search_agent.start()
            
            # Register agent
            await registry.register_agent(search_agent)
            
            logger.info("Search Agent components initialized successfully")
            
    except Exception as e:
        logger.error(f"Error initializing components: {str(e)}")
        raise

@app.route(route="search-similar-incidents", methods=["POST"])
async def search_similar_incidents(req: func.HttpRequest) -> func.HttpResponse:
    """
    Search for similar incidents using semantic search.
    
    Expected JSON payload:
    {
        "query": "User cannot login to LMS",
        "search_type": "semantic",  // "semantic", "vector", "hybrid"
        "max_results": 5,
        "filters": {
            "category": "Learning Management System (LMS)",
            "severity": "High",
            "date_range": {
                "start": "2024-01-01",
                "end": "2024-12-31"
            }
        },
        "options": {
            "include_explanations": true,
            "similarity_threshold": 0.3,
            "boost_recent": true
        }
    }
    
    Returns:
    {
        "query": "User cannot login to LMS",
        "search_type": "semantic",
        "results": [...],
        "total_count": 10,
        "search_metadata": {...}
    }
    """
    try:
        await initialize_components()
        
        # Rate limiting
        client_ip = req.headers.get('x-forwarded-for', 'unknown')
        if not check_rate_limit(client_ip):
            return create_error_response("Rate limit exceeded", 429)
        
        # Start timing for metrics
        start_time = time.time()
        
        # Validate request
        data, error_response = validate_json_request(req, required_fields=['query'])
        if error_response:
            return error_response
            
        # Sanitize input data
        data = sanitize_request_data(data)
            
        # Check cache
        cache_key = f"search:{data['search_type']}:{data['query']}"
        cached_result = get_cached_result(cache_key)
        if cached_result:
            record_custom_metric("cache_hit", 1, {"endpoint": "search-similar-incidents"})
            return func.HttpResponse(
                json.dumps(cached_result),
                status_code=200,
                mimetype="application/json"
            )
        
        # Extract parameters
        query = req_body['query']
        search_type = req_body.get('search_type', 'semantic')
        max_results = req_body.get('max_results', 10)
        filters = req_body.get('filters', {})
        options = req_body.get('options', {})
        
        # Validate search type
        valid_search_types = ['semantic', 'vector', 'hybrid', 'text']
        if search_type not in valid_search_types:
            return func.HttpResponse(
                json.dumps({"error": f"Invalid search_type. Must be one of: {valid_search_types}"}),
                status_code=400,
                mimetype="application/json"
            )
        
        # Prepare task for search agent
        task_data = {
            'type': f'{search_type}_search',
            'query': query,
            'max_results': max_results,
            'filters': filters,
            'include_explanations': options.get('include_explanations', True),
            'similarity_threshold': options.get('similarity_threshold', 0.3),
            'boost_recent': options.get('boost_recent', False)
        }
        
        # Process the search request
        start_time = datetime.now()
        result = await search_agent.process_task(task_data)
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Format response
        response = {
            "query": query,
            "search_type": search_type,
            "results": result.get('results', []),
            "total_count": result.get('total_count', 0),
            "search_metadata": {
                "function_name": "search-similar-incidents",
                "processing_time_seconds": processing_time,
                "timestamp": datetime.now().isoformat(),
                "api_version": "1.0.0",
                "similarity_threshold": task_data.get('similarity_threshold'),
                "filters_applied": len(filters) > 0
            }
        }
        
        # Add processing metadata from agent
        if 'processing_metadata' in result:
            response['search_metadata'].update(result['processing_metadata'])
        
        # Add request tracking
        response['request_id'] = req.headers.get('x-request-id', 'unknown')
        
        logger.info(f"Search request processed successfully in {processing_time:.2f}s")
        
        return func.HttpResponse(
            json.dumps(response, indent=2),
            status_code=200,
            mimetype="application/json"
        )
        
    except Exception as e:
        logger.error(f"Error in search-similar-incidents: {str(e)}")
        return func.HttpResponse(
            json.dumps({
                "error": "Search request failed",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }),
            status_code=500,
            mimetype="application/json"
        )

@app.route(route="vector-search", methods=["POST"])
async def vector_search(req: func.HttpRequest) -> func.HttpResponse:
    """
    Dedicated endpoint for vector similarity search.
    
    Expected request body:
    {
        "query": "Login issues with authentication system",
        "max_results": 10,
        "similarity_threshold": 0.4,
        "filters": {...}
    }
    """
    
    logger.info("Vector search request received")
    
    try:
        await initialize_components()
        
        # Parse request
        request_data = req.get_json()
        if not request_data or 'query' not in request_data:
            return func.HttpResponse(
                json.dumps({"error": "query is required"}),
                status_code=400,
                mimetype="application/json"
            )
        
        # Prepare vector search task
        task_data = {
            'type': 'vector_search',
            'query': request_data['query'],
            'max_results': request_data.get('max_results', 10),
            'similarity_threshold': request_data.get('similarity_threshold', 0.4),
            'filters': request_data.get('filters', {})
        }
        
        # Process vector search
        result = await search_agent.process_task(task_data)
        
        return func.HttpResponse(
            json.dumps(result, indent=2),
            status_code=200,
            mimetype="application/json"
        )
        
    except Exception as e:
        logger.error(f"Error in vector search: {str(e)}")
        
        error_response = {
            "error": "Vector search failed",
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }
        
        return func.HttpResponse(
            json.dumps(error_response),
            status_code=500,
            mimetype="application/json"
        )

@app.route(route="hybrid-search", methods=["POST"])
async def hybrid_search(req: func.HttpRequest) -> func.HttpResponse:
    """
    Dedicated endpoint for hybrid search (combines text and vector search).
    
    Expected request body:
    {
        "query": "Email server connectivity problems",
        "text_weight": 0.4,
        "vector_weight": 0.6,
        "max_results": 15,
        "filters": {...}
    }
    """
    
    logger.info("Hybrid search request received")
    
    try:
        await initialize_components()
        
        # Parse request
        request_data = req.get_json()
        if not request_data or 'query' not in request_data:
            return func.HttpResponse(
                json.dumps({"error": "query is required"}),
                status_code=400,
                mimetype="application/json"
            )
        
        # Prepare hybrid search task
        task_data = {
            'type': 'hybrid_search',
            'query': request_data['query'],
            'text_weight': request_data.get('text_weight', 0.4),
            'vector_weight': request_data.get('vector_weight', 0.6),
            'max_results': request_data.get('max_results', 15),
            'filters': request_data.get('filters', {})
        }
        
        # Process hybrid search
        result = await search_agent.process_task(task_data)
        
        return func.HttpResponse(
            json.dumps(result, indent=2),
            status_code=200,
            mimetype="application/json"
        )
        
    except Exception as e:
        logger.error(f"Error in hybrid search: {str(e)}")
        
        error_response = {
            "error": "Hybrid search failed",
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }
        
        return func.HttpResponse(
            json.dumps(error_response),
            status_code=500,
            mimetype="application/json"
        )

@app.route(route="search-suggestions", methods=["POST"])
async def search_suggestions(req: func.HttpRequest) -> func.HttpResponse:
    """
    Get search suggestions based on query.
    
    Expected request body:
    {
        "partial_query": "login prob",
        "max_suggestions": 5,
        "suggestion_type": "completion"  // "completion", "similar", "related"
    }
    """
    
    logger.info("Search suggestions request received")
    
    try:
        await initialize_components()
        
        # Parse request
        request_data = req.get_json()
        if not request_data or 'partial_query' not in request_data:
            return func.HttpResponse(
                json.dumps({"error": "partial_query is required"}),
                status_code=400,
                mimetype="application/json"
            )
        
        # Prepare suggestions task
        task_data = {
            'type': 'search_suggestions',
            'partial_query': request_data['partial_query'],
            'max_suggestions': request_data.get('max_suggestions', 5),
            'suggestion_type': request_data.get('suggestion_type', 'completion')
        }
        
        # Process suggestions
        result = await search_agent.process_task(task_data)
        
        return func.HttpResponse(
            json.dumps(result, indent=2),
            status_code=200,
            mimetype="application/json"
        )
        
    except Exception as e:
        logger.error(f"Error getting search suggestions: {str(e)}")
        
        error_response = {
            "error": "Search suggestions failed",
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }
        
        return func.HttpResponse(
            json.dumps(error_response),
            status_code=500,
            mimetype="application/json"
        )

async def check_azure_services():
    """Check Azure services health"""
    if not azure_manager:
        return {"status": "not_initialized"}
    return await azure_manager.get_health_status()

async def check_agent_registry():
    """Check agent registry health"""
    if not registry:
        return {"status": "not_initialized"}
    return registry.get_health_status()

async def check_message_bus():
    """Check message bus health"""
    if not message_bus:
        return {"status": "not_initialized"}
    return message_bus.get_health_status()

async def check_all_agents():
    """Check all agents health"""
    if not search_agent:
        return {"status": "not_initialized"}
    
    try:
        test_task = {
            'type': 'semantic_search',
            'query': 'test query',
            'max_results': 1,
            'filters': {}
        }
        await search_agent.process_task(test_task)
        return {"status": "healthy", "test_result": "success"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

@app.route(route="search-health", methods=["GET"])
async def search_health(req: func.HttpRequest) -> func.HttpResponse:
    """
    Health check endpoint for the search service.
    """
    
    try:
        await initialize_components()
        
        # Run health checks in parallel
        start_time = time.time()
        health_tasks = [
            check_azure_services(),
            check_agent_registry(),
            check_message_bus(),
            check_all_agents()
        ]
        
        results = await asyncio.gather(*health_tasks, return_exceptions=True)
        azure_health, registry_health, bus_health, agents_health = results
        
        # Handle exceptions
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Health check {i} failed: {result}")
        
        health_data = {
            "status": "healthy" if all(not isinstance(r, Exception) and r.get("status") == "healthy" for r in results) else "unhealthy",
            "agent_status": agents_health if not isinstance(agents_health, Exception) else {"error": str(agents_health)},
            "azure_services": azure_health if not isinstance(azure_health, Exception) else {"error": str(azure_health)},
            "registry": registry_health if not isinstance(registry_health, Exception) else {"error": str(registry_health)},
            "message_bus": bus_health if not isinstance(bus_health, Exception) else {"error": str(bus_health)},
            "cache_size": len(search_agent.search_cache) if search_agent else 0,
            "timestamp": datetime.now().isoformat(),
            "uptime": "available" if search_agent else "unavailable",
            "processing_time": time.time() - start_time
        }
        
        status_code = 200 if search_agent else 503
        
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

@app.route(route="search-capabilities", methods=["GET"])
async def search_capabilities(req: func.HttpRequest) -> func.HttpResponse:
    """
    Get search agent capabilities and configuration.
    """
    
    try:
        await initialize_components()
        
        capabilities_data = {
            "search_types": [
                {
                    "type": "semantic_search",
                    "description": "Semantic similarity search using AI embeddings",
                    "supports_filters": True,
                    "supports_explanations": True
                },
                {
                    "type": "vector_search", 
                    "description": "Vector similarity search using embeddings",
                    "supports_filters": True,
                    "supports_similarity_threshold": True
                },
                {
                    "type": "hybrid_search",
                    "description": "Combines text and vector search for best results",
                    "supports_weight_adjustment": True,
                    "supports_filters": True
                },
                {
                    "type": "text_search",
                    "description": "Traditional text-based search",
                    "supports_filters": True,
                    "supports_boolean_queries": True
                }
            ],
            "filter_options": [
                "category",
                "severity", 
                "priority",
                "status",
                "date_range",
                "assigned_to",
                "resolution_time"
            ],
            "configuration": {
                "default_max_results": 10,
                "similarity_threshold": 0.3,
                "cache_enabled": True,
                "cache_ttl_seconds": 300
            },
            "agent_capabilities": search_agent.get_capabilities() if search_agent else [],
            "timestamp": datetime.now().isoformat()
        }
        
        return func.HttpResponse(
            json.dumps(capabilities_data, indent=2),
            status_code=200,
            mimetype="application/json"
        )
        
    except Exception as e:
        logger.error(f"Error getting capabilities: {str(e)}")
        
        error_response = {
            "error": "Failed to get capabilities",
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }
        
        return func.HttpResponse(
            json.dumps(error_response),
            status_code=500,
            mimetype="application/json"
        )

@app.route(route="search-statistics", methods=["GET"])
async def search_statistics(req: func.HttpRequest) -> func.HttpResponse:
    """
    Get search statistics and performance metrics.
    """
    
    try:
        await initialize_components()
        
        if not search_agent:
            return func.HttpResponse(
                json.dumps({"error": "Search agent not initialized"}),
                status_code=503,
                mimetype="application/json"
            )
        
        # Get agent metrics
        agent_status = search_agent.get_status()
        
        statistics_data = {
            "agent_metrics": {
                "total_searches": agent_status.get('metrics', {}).get('total_processed', 0),
                "successful_searches": agent_status.get('metrics', {}).get('successful', 0),
                "failed_searches": agent_status.get('metrics', {}).get('failed', 0),
                "success_rate": agent_status.get('metrics', {}).get('success_rate', 0),
                "average_search_time": agent_status.get('metrics', {}).get('avg_processing_time', 0),
                "last_search": agent_status.get('metrics', {}).get('last_activity')
            },
            "cache_statistics": {
                "cache_size": len(search_agent.search_cache),
                "cache_hit_rate": getattr(search_agent, 'cache_hit_rate', 0),
                "cache_ttl_seconds": search_agent.cache_ttl
            },
            "search_patterns": {
                "most_common_filters": getattr(search_agent, 'common_filters', {}),
                "average_results_returned": getattr(search_agent, 'avg_results', 0)
            },
            "timestamp": datetime.now().isoformat()
        }
        
        return func.HttpResponse(
            json.dumps(statistics_data, indent=2),
            status_code=200,
            mimetype="application/json"
        )
        
    except Exception as e:
        logger.error(f"Error getting statistics: {str(e)}")
        
        error_response = {
            "error": "Failed to get statistics",
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
    global search_agent, message_bus, registry
    
    try:
        if search_agent:
            await search_agent.stop()
        if registry:
            await registry.stop()
        if message_bus:
            await message_bus.stop()
            
        logger.info("Search service resources cleaned up successfully")
        
    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}")