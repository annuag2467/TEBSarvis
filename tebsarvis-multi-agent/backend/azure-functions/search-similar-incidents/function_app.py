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

# Add the backend path to sys.path to import our agents
backend_path = os.path.join(os.path.dirname(__file__), '..', '..', 'agents')
sys.path.append(backend_path)

# FIXED IMPORTS
from ...agents.reactive.search_agent import SearchAgent
from ...agents.core.agent_registry import get_global_registry
from ...agents.core.agent_communication import MessageBus
from ..shared.azure_clients import AzureClientManager

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
    global search_agent, azure_manager, message_bus, registry
    
    try:
        if not search_agent:
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
        
        # Parse request
        req_body = req.get_json()
        if not req_body or 'query' not in req_body:
            return func.HttpResponse(
                json.dumps({"error": "Missing query in request"}),
                status_code=400,
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

@app.route(route="search-health", methods=["GET"])
async def search_health(req: func.HttpRequest) -> func.HttpResponse:
    """
    Health check endpoint for the search service.
    """
    
    try:
        await initialize_components()
        
        # Check agent status
        agent_status = search_agent.get_status() if search_agent else {"status": "not_initialized"}
        
        # Check Azure services
        azure_health = await azure_manager.get_health_status() if azure_manager else {"status": "not_initialized"}
        
        # Test search functionality
        search_test = {"status": "not_tested"}
        if search_agent:
            try:
                test_task = {
                    'type': 'semantic_search',
                    'query': 'test query',
                    'max_results': 1,
                    'filters': {}
                }
                test_result = await search_agent.process_task(test_task)
                search_test = {"status": "healthy", "test_result": "success"}
            except Exception as e:
                search_test = {"status": "unhealthy", "error": str(e)}
        
        health_data = {
            "status": "healthy" if search_agent else "unhealthy",
            "agent_status": agent_status,
            "azure_services": azure_health,
            "search_functionality": search_test,
            "cache_size": len(search_agent.search_cache) if search_agent else 0,
            "timestamp": datetime.now().isoformat(),
            "uptime": "available" if search_agent else "unavailable"
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