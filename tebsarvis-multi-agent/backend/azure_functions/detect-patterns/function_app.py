"""
Azure Function App for Pattern Detection Agent API
Endpoint: /detect-patterns
Provides ML clustering, trend analysis, and anomaly detection for incidents.
"""

import azure.functions as func
import logging
import json
import asyncio
import sys
import os
from datetime import datetime, timedelta
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
    get_client_ip
)
from ..shared.azure_clients import AzureClientManager
from ..shared.caching import get_cached_result, set_cached_result

# FIXED IMPORTS
from ...agents.proactive.pattern_detection_agent import PatternDetectionAgent
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

# Global components
connection_pool = None
executor = None
cache = {}
cache_ttl = {}
rate_limit_cache = defaultdict(list)
RATE_LIMIT = int(os.getenv('RATE_LIMIT', '100'))  # requests per minute
CACHE_DEFAULT_TTL = 300  # 5 minutes for pattern detections
MAX_WORKERS = int(os.getenv('MAX_WORKERS', '4'))

# Analysis type configurations
ANALYSIS_CONFIGS = {
    'clustering': {
        'cache_ttl': 600,  # 10 minutes
        'default_params': {
            'method': 'mixed',
            'min_cluster_size': 3
        }
    },
    'trends': {
        'cache_ttl': 900,  # 15 minutes
        'default_params': {
            'trend_types': ['volume', 'category', 'severity'],
            'granularity': 'daily'
        }
    },
    'anomalies': {
        'cache_ttl': 300,  # 5 minutes
        'default_params': {
            'sensitivity': 'medium',
            'types': ['volume', 'pattern', 'timing']
        }
    },
    'correlations': {
        'cache_ttl': 1800,  # 30 minutes
        'default_params': {
            'method': 'pearson',
            'attributes': ['category', 'severity', 'priority']
        }
    },
    'comprehensive': {
        'cache_ttl': 1800,  # 30 minutes
        'default_params': {
            'components': ['clustering', 'trends', 'anomalies', 'correlations']
        }
    }
}

# Initialize global components
app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)
logger = logging.getLogger(__name__)

# Global variables for agent instances
pattern_agent = None
azure_manager = None
message_bus = None
registry = None

async def initialize_components():
    """Initialize agents and Azure components with connection pooling and monitoring"""
    global pattern_agent, azure_manager, message_bus, registry, connection_pool, executor
    
    try:
        # Validate environment variables
        validate_environment()

        # Initialize connection pool if not exists
        if not connection_pool:
            connection_pool = aiohttp.TCPConnector(
                limit=100,  # Max connections
                limit_per_host=30,  # Max connections per host
                ttl_dns_cache=300,  # DNS cache TTL
                ssl=False  # Disable SSL for internal communications
            )
        
        # Initialize thread executor if not exists
        if not executor:
            executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

        if not pattern_agent:
            # Initialize Azure manager with connection pool
            azure_manager = AzureClientManager(connector=connection_pool)
            await azure_manager.initialize()
            
            # Initialize message bus and registry
            message_bus = MessageBus()
            await message_bus.start()
            
            registry = get_global_registry()
            await registry.start()
            
            # Initialize pattern detection agent
            pattern_agent = PatternDetectionAgent()
            await pattern_agent.start()
            
            # Register agent
            await registry.register_agent(pattern_agent)
            
            # Setup monitoring
            setup_monitoring(app_name="detect-patterns")
            
            logger.info("Pattern Detection Agent components initialized successfully")
            
    except Exception as e:
        logger.error(f"Error initializing components: {str(e)}", exc_info=True)
        # Cleanup on failure
        if connection_pool:
            await connection_pool.close()
        if executor:
            executor.shutdown(wait=False)
        raise

@app.route(route="detect-patterns", methods=["POST"])
async def detect_patterns(req: func.HttpRequest) -> func.HttpResponse:
    """
    Comprehensive pattern detection including clustering, trends, and anomalies.
    """
    logger.info("Pattern detection request received")
    
    try:
        await initialize_components()
        
        # Rate limiting
        client_ip = req.headers.get('x-forwarded-for', 'unknown')
        if not check_rate_limit(client_ip):
            return create_error_response("Rate limit exceeded", 429)
            
        # Validate request
        data, error_response = validate_json_request(req)
        if error_response:
            return error_response
            
        # Sanitize input data
        data = sanitize_input(data)
            
        # Check cache
        cache_key = f"patterns:{json.dumps(data, sort_keys=True)}"
        cached_result = get_cached_result(cache_key)
        if cached_result:
            logger.info("Cache hit for pattern detection request")
            return func.HttpResponse(
                json.dumps(cached_result, indent=2),
                status_code=200,
                mimetype="application/json"
            )
            
        # Extract parameters
        analysis_type = data.get('analysis_type', 'comprehensive')
        time_range = data.get('time_range', {'days': 30})
        parameters = data.get('parameters', {})
        options = data.get('options', {})
        
        # Validate analysis type
        valid_types = ['clustering', 'trends', 'anomalies', 'comprehensive', 'correlations']
        if analysis_type not in valid_types:
            return create_error_response(
                f"Invalid analysis_type. Must be one of: {valid_types}", 
                400
            )
        
        # Prepare task based on analysis type
        task_data = {
            'type': f"{analysis_type}_analysis",
            'time_range': time_range
        }
        
        # Add type-specific parameters
        task_data.update(ANALYSIS_CONFIGS[analysis_type]['default_params'])
        task_data.update(parameters)  # Override defaults with user parameters
        
        # Add options
        task_data.update({
            'include_insights': options.get('include_insights', True),
            'generate_forecasts': options.get('generate_forecasts', False),
            'detailed_analysis': options.get('detailed_analysis', False)
        })
        
        # Process the pattern detection request
        start_time = datetime.now()
        result = await pattern_agent.process_task(task_data)
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Add processing metadata
        result['analysis_metadata'] = result.get('analysis_metadata', {})
        result['analysis_metadata'].update({
            'function_name': 'detect-patterns',
            'processing_time_seconds': processing_time,
            'timestamp': datetime.now().isoformat(),
            'api_version': '1.0.0',
            'analysis_type': analysis_type
        })
        
        # Add request tracking
        result['request_id'] = req.headers.get('x-request-id', 'unknown')
        
        # Cache successful result
        set_cached_result(cache_key, result, ANALYSIS_CONFIGS[analysis_type]['cache_ttl'])
        
        logger.info(f"Pattern detection request processed successfully in {processing_time:.2f}s")
        
        return func.HttpResponse(
            json.dumps(result, indent=2),
            status_code=200,
            mimetype="application/json"
        )
        
    except Exception as e:
        logger.error(f"Error in pattern detection: {str(e)}", exc_info=True)
        return create_error_response(
            "Pattern detection failed",
            500,
            {"error_message": str(e)}
        )

@app.route(route="pattern-health", methods=["GET"])
async def pattern_health(req: func.HttpRequest) -> func.HttpResponse:
    """
    Health check endpoint for the pattern detection service.
    """
    try:
        await initialize_components()
        
        # Check agent status
        agent_status = pattern_agent.get_status() if pattern_agent else {"status": "not_initialized"}
        
        # Check Azure services
        azure_health = await azure_manager.get_health_status() if azure_manager else {"status": "not_initialized"}
        
        # Test pattern detection functionality
        pattern_test = {"status": "not_tested"}
        if pattern_agent:
            try:
                test_task = {
                    'type': 'incident_clustering',
                    'time_range': {'days': 1},
                    'clustering_method': 'categorical',
                    'min_cluster_size': 1
                }
                test_result = await pattern_agent.process_task(test_task)
                pattern_test = {"status": "healthy", "test_result": "success"}
            except Exception as e:
                pattern_test = {"status": "unhealthy", "error": str(e)}
        
        health_data = {
            "status": "healthy" if pattern_agent else "unhealthy",
            "agent_status": agent_status,
            "azure_services": azure_health,
            "pattern_functionality": pattern_test,
            "cache_size": len(pattern_agent.analysis_cache) if pattern_agent else 0,
            "timestamp": datetime.now().isoformat(),
            "uptime": "available" if pattern_agent else "unavailable"
        }
        
        status_code = 200 if pattern_agent else 503
        
        return func.HttpResponse(
            json.dumps(health_data, indent=2),
            status_code=status_code,
            mimetype="application/json"
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}", exc_info=True)
        return create_error_response("Health check failed", 503, {"error": str(e)})

# Graceful shutdown handler
@app.function_name("cleanup")
async def cleanup_resources():
    """Clean up resources on function app shutdown"""
    global pattern_agent, message_bus, registry, connection_pool, executor
    
    try:
        if pattern_agent:
            await pattern_agent.stop()
        if registry:
            await registry.stop()
        if message_bus:
            await message_bus.stop()
        if connection_pool:
            await connection_pool.close()
        if executor:
            executor.shutdown(wait=True)
            
        logger.info("Pattern detection service resources cleaned up successfully")
        
    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}", exc_info=True)
        raise
