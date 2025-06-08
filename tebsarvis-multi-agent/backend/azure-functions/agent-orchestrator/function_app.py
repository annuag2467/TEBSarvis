"""
Azure Function App for Agent Orchestrator API
Endpoint: /coordinate-*
Provides unified orchestration across all agents using OrchestrationManager.
"""

import azure.functions as func
import logging
import json
import asyncio
import sys
import os
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import contextlib
import socket
from functools import wraps

# Add the backend path to sys.path to import our agents
# backend_path = os.path.join(os.path.dirname(__file__), '..', '..', 'agents')
# sys.path.append(backend_path)

# FIXED IMPORTS - Using OrchestrationManager for unified orchestration
from ...agents.orchestrator.orchestration_manager import OrchestrationManager, OrchestrationHealthMonitor
from ...agents.core.agent_registry import get_global_registry
from ...agents.core.agent_communication import MessageBus
from ..shared.azure_clients import AzureClientManager
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
app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)
logger = logging.getLogger(__name__)

# Performance components
connection_pool = None
executor = None
cache = {}
cache_ttl = {}
rate_limit_cache = defaultdict(list)
metrics = defaultdict(float)

# Configuration
DEFAULT_CACHE_TTL = 300  # 5 minutes
MAX_REQUESTS_PER_MINUTE = 60
RATE_LIMIT_WINDOW = 60  # 1 minute
CONNECTION_POOL_SIZE = 100
THREAD_POOL_SIZE = 10

REQUIRED_ENV_VARS = [
    'AzureWebJobsStorage',
    'APPINSIGHTS_INSTRUMENTATIONKEY'
]

def validate_environment():
    """Validate all required environment variables are set"""
    missing_vars = [var for var in REQUIRED_ENV_VARS if not os.getenv(var)]
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

def cache_response(ttl: int = DEFAULT_CACHE_TTL):
    """Cache function response with TTL"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            cache_key = f"{func.__name__}:{str(args)}:{str(kwargs)}"
            
            # Check cache
            if cache_key in cache and datetime.now() < cache_ttl.get(cache_key, datetime.min):
                metrics['cache_hits'] += 1
                return cache[cache_key]
            
            # Execute function
            metrics['cache_misses'] += 1
            result = await func(*args, **kwargs)
            
            # Cache result
            cache[cache_key] = result
            cache_ttl[cache_key] = datetime.now() + timedelta(seconds=ttl)
            
            return result
        return wrapper
    return decorator

def rate_limit(func):
    """Rate limit by client IP"""
    @wraps(func)
    async def wrapper(req: func.HttpRequest, *args, **kwargs):
        client_ip = req.headers.get('X-Forwarded-For', 'unknown')
        
        # Clean old requests
        now = datetime.now()
        rate_limit_cache[client_ip] = [t for t in rate_limit_cache[client_ip] 
                                     if now - t < timedelta(seconds=RATE_LIMIT_WINDOW)]
        
        # Check rate limit
        if len(rate_limit_cache[client_ip]) >= MAX_REQUESTS_PER_MINUTE:
            return func.HttpResponse(
                json.dumps({
                    "error": "Rate limit exceeded",
                    "retry_after": RATE_LIMIT_WINDOW
                }),
                status_code=429,
                mimetype="application/json"
            )
        
        # Add request timestamp
        rate_limit_cache[client_ip].append(now)
        
        return await func(req, *args, **kwargs)
    return wrapper

def validate_json_payload(required_fields: list):
    """Validate JSON payload has required fields"""
    def decorator(func):
        @wraps(func)
        async def wrapper(req: func.HttpRequest, *args, **kwargs):
            try:
                payload = req.get_json()
                missing_fields = [field for field in required_fields if field not in payload]
                if missing_fields:
                    return func.HttpResponse(
                        json.dumps({
                            "error": f"Missing required fields: {', '.join(missing_fields)}",
                            "required_fields": required_fields
                        }),
                        status_code=400,
                        mimetype="application/json"
                    )
            except ValueError:
                return func.HttpResponse(
                    json.dumps({"error": "Invalid JSON payload"}),
                    status_code=400,
                    mimetype="application/json"
                )
            return await func(req, *args, **kwargs)
        return wrapper
    return decorator

def track_performance(func):
    """Track function performance metrics"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = datetime.now()
        try:
            result = await func(*args, **kwargs)
            processing_time = (datetime.now() - start_time).total_seconds()
            metrics[f"{func.__name__}_success"] += 1
            metrics[f"{func.__name__}_time"] += processing_time
            return result
        except Exception as e:
            metrics[f"{func.__name__}_error"] += 1
            raise
    return wrapper

# Orchestration components
orchestration_manager = None
azure_manager = None
message_bus = None
registry = None

async def initialize_orchestration():
    """Initialize orchestration manager and all components with performance optimizations"""
    global orchestration_manager, azure_manager, message_bus, registry, connection_pool, executor
    
    try:
        # Validate environment
        validate_environment()
        
        if not orchestration_manager:
            # Initialize connection pool
            if not connection_pool:
                connection_pool = aiohttp.TCPConnector(
                    limit=CONNECTION_POOL_SIZE,
                    ttl_dns_cache=300,
                    keepalive_timeout=60
                )
                
            # Initialize thread pool
            if not executor:
                executor = ThreadPoolExecutor(max_workers=THREAD_POOL_SIZE)
            
            # Initialize Azure manager with connection pooling
            azure_manager = AzureClientManager(
                connection_pool=connection_pool,
                executor=executor
            )
            await azure_manager.initialize()
            
            # Initialize message bus and registry with performance settings
            message_bus = MessageBus(connection_pool=connection_pool)
            registry = get_global_registry()
            
            # Initialize unified orchestration manager
            orchestration_manager = OrchestrationManager(
                registry=registry,
                message_bus=message_bus,
                connection_pool=connection_pool,
                executor=executor
            )
            await orchestration_manager.start()
            
            logger.info("Orchestration Manager initialized successfully with performance optimizations")
            
    except Exception as e:
        logger.error(f"Error initializing orchestration: {str(e)}")
        raise

@app.route(route="coordinate-incident-resolution", methods=["POST"])
@rate_limit
@validate_json_payload(["incident_data"])
@track_performance
@cache_response(ttl=60)  # Cache similar incidents for 1 minute
async def coordinate_incident_resolution(req: func.HttpRequest) -> func.HttpResponse:
    """
    Coordinate a complete incident resolution workflow across multiple agents.
    Uses intelligent orchestration to choose the best coordination strategy.
    
    Expected JSON payload:
    {
        "incident_data": {
            "id": "INC001",
            "summary": "User cannot access LMS",
            "description": "Multiple users reporting login issues",
            "category": "Learning Management System (LMS)",
            "severity": "High",
            "priority": "High"
        },
        "orchestration_options": {
            "requires_consensus": false,
            "complexity": "medium",
            "timeout_minutes": 30
        }
    }
    """
    try:
        await initialize_orchestration()
        
        # Parse request with validation
        req_body = req.get_json()
        incident_data = req_body['incident_data']
        orchestration_options = req_body.get('orchestration_options', {})
        
        # Generate cache key based on incident details
        cache_key = f"incident:{incident_data.get('id')}:{incident_data.get('category')}"
        cached_result = cache.get(cache_key)
        if cached_result and datetime.now() < cache_ttl.get(cache_key, datetime.min):
            metrics['cache_hits'] += 1
            return func.HttpResponse(
                json.dumps(cached_result, indent=2),
                status_code=200,
                mimetype="application/json"
            )
        
        # Prepare intelligent workflow request
        workflow_request = {
            'type': 'incident_resolution',
            'complexity': orchestration_options.get('complexity', 'medium'),
            'requires_consensus': orchestration_options.get('requires_consensus', False),
            'agents_required': ['context', 'search', 'resolution', 'conversation'],
            'incident_data': incident_data,
            'shared_data': {
                'incident_summary': incident_data.get('summary', ''),
                'incident_category': incident_data.get('category', ''),
                'priority_level': incident_data.get('priority', 'normal')
            },
            'timeout_minutes': orchestration_options.get('timeout_minutes', 30)
        }
        
        # Execute with intelligent orchestration and performance tracking
        start_time = datetime.now()
        result = await orchestration_manager.execute_intelligent_workflow(workflow_request)
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Add metrics and API metadata
        result['api_metadata'] = {
            'function_name': 'coordinate-incident-resolution',
            'processing_time_seconds': processing_time,
            'timestamp': datetime.now().isoformat(),
            'api_version': '1.0.0',
            'orchestration_method': result.get('execution_method', 'intelligent'),
            'cache_status': 'miss'
        }
        metrics['incident_resolution_time'] += processing_time
        metrics['incident_resolution_count'] += 1
        
        # Cache the result
        cache[cache_key] = result
        cache_ttl[cache_key] = datetime.now() + timedelta(seconds=60)
        
        # Add request tracking
        result['request_id'] = req.headers.get('x-request-id', 'unknown')
        
        logger.info(f"Incident resolution coordinated successfully in {processing_time:.2f}s")
        
        return func.HttpResponse(
            json.dumps(result, indent=2),
            status_code=200,
            mimetype="application/json"
        )
        
    except Exception as e:
        metrics['incident_resolution_errors'] += 1
        logger.error(f"Error in coordinate-incident-resolution: {str(e)}")
        return func.HttpResponse(
            json.dumps({
                "error": "Incident resolution coordination failed",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }),
            status_code=500,
            mimetype="application/json"
        )

@app.route(route="coordinate-pattern-analysis", methods=["POST"])
@rate_limit
@validate_json_payload(["analysis_request"])
@track_performance
@cache_response(ttl=300)  # Cache pattern analysis for 5 minutes
async def coordinate_pattern_analysis(req: func.HttpRequest) -> func.HttpResponse:
    """
    Coordinate comprehensive pattern analysis across multiple agents.
    
    Expected JSON payload:
    {
        "analysis_request": {
            "time_range": {"days": 30},
            "analysis_types": ["clustering", "trends", "anomalies"],
            "alert_conditions": {}
        },
        "orchestration_options": {
            "complexity": "high",
            "timeout_minutes": 45
        }
    }
    """
    try:
        await initialize_orchestration()
        
        # Parse request with validation
        req_body = req.get_json()
        analysis_request = req_body['analysis_request']
        orchestration_options = req_body.get('orchestration_options', {})
        
        # Generate cache key based on analysis parameters
        cache_key = f"pattern_analysis:{str(analysis_request)}"
        cached_result = cache.get(cache_key)
        if cached_result and datetime.now() < cache_ttl.get(cache_key, datetime.min):
            metrics['cache_hits'] += 1
            return func.HttpResponse(
                json.dumps(cached_result, indent=2),
                status_code=200,
                mimetype="application/json"
            )
        
        # Prepare intelligent workflow request with performance settings
        workflow_request = {
            'type': 'pattern_analysis',
            'complexity': orchestration_options.get('complexity', 'high'),
            'requires_consensus': False,
            'agents_required': ['pattern_detection', 'alerting'],
            'analysis_request': analysis_request,
            'timeout_minutes': orchestration_options.get('timeout_minutes', 45),
            'performance_settings': {
                'use_connection_pool': True,
                'batch_size': 1000,
                'parallel_processing': True
            }
        }
        
        # Execute with performance tracking
        start_time = datetime.now()
        result = await orchestration_manager.execute_intelligent_workflow(workflow_request)
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Add metrics and API metadata
        result['api_metadata'] = {
            'function_name': 'coordinate-pattern-analysis',
            'processing_time_seconds': processing_time,
            'timestamp': datetime.now().isoformat(),
            'api_version': '1.0.0',
            'orchestration_method': result.get('execution_method', 'intelligent'),
            'cache_status': 'miss'
        }
        metrics['pattern_analysis_time'] += processing_time
        metrics['pattern_analysis_count'] += 1
        
        # Cache the result
        cache[cache_key] = result
        cache_ttl[cache_key] = datetime.now() + timedelta(seconds=300)
        
        logger.info(f"Pattern analysis coordinated successfully in {processing_time:.2f}s")
        
        return func.HttpResponse(
            json.dumps(result, indent=2),
            status_code=200,
            mimetype="application/json"
        )
        
    except Exception as e:
        metrics['pattern_analysis_errors'] += 1
        logger.error(f"Error in coordinate-pattern-analysis: {str(e)}")
        return func.HttpResponse(
            json.dumps({
                "error": "Pattern analysis coordination failed",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }),
            status_code=500,
            mimetype="application/json"
        )

@app.route(route="coordinate-proactive-monitoring", methods=["POST"])
@rate_limit
@validate_json_payload(["monitoring_config"])
@track_performance
async def coordinate_proactive_monitoring(req: func.HttpRequest) -> func.HttpResponse:
    """
    Coordinate proactive monitoring workflow with real-time alerting.
    
    Expected JSON payload:
    {
        "monitoring_config": {
            "monitoring_rules": ["volume_spike", "pattern_anomaly"],
            "notification_channels": ["email", "teams"],
            "monitoring_window": {"minutes": 15}
        },
        "orchestration_options": {
            "complexity": "medium",
            "timeout_minutes": 10
        }
    }
    """
    try:
        await initialize_orchestration()
        
        # Parse request with validation
        req_body = req.get_json()
        monitoring_config = req_body['monitoring_config']
        orchestration_options = req_body.get('orchestration_options', {})
        
        # Prepare intelligent workflow request with performance settings
        workflow_request = {
            'type': 'proactive_monitoring',
            'complexity': orchestration_options.get('complexity', 'medium'),
            'requires_consensus': False,
            'agents_required': ['pattern_detection', 'alerting'],
            'monitoring_config': monitoring_config,
            'timeout_minutes': orchestration_options.get('timeout_minutes', 10),
            'performance_settings': {
                'use_connection_pool': True,
                'batch_size': 100,
                'parallel_processing': True,
                'real_time_mode': True
            }
        }
        
        # Execute with performance tracking
        start_time = datetime.now()
        result = await orchestration_manager.execute_intelligent_workflow(workflow_request)
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Add metrics and API metadata
        result['api_metadata'] = {
            'function_name': 'coordinate-proactive-monitoring',
            'processing_time_seconds': processing_time,
            'timestamp': datetime.now().isoformat(),
            'api_version': '1.0.0',
            'orchestration_method': result.get('execution_method', 'intelligent')
        }
        metrics['proactive_monitoring_time'] += processing_time
        metrics['proactive_monitoring_count'] += 1
        
        logger.info(f"Proactive monitoring coordinated successfully in {processing_time:.2f}s")
        
        return func.HttpResponse(
            json.dumps(result, indent=2),
            status_code=200,
            mimetype="application/json"
        )
        
    except Exception as e:
        metrics['proactive_monitoring_errors'] += 1
        logger.error(f"Error in coordinate-proactive-monitoring: {str(e)}")
        return func.HttpResponse(
            json.dumps({
                "error": "Proactive monitoring coordination failed",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }),
            status_code=500,
            mimetype="application/json"
        )

@app.route(route="orchestration-status", methods=["GET"])
@rate_limit
@track_performance
@cache_response(ttl=30)  # Cache status for 30 seconds
async def orchestration_status(req: func.HttpRequest) -> func.HttpResponse:
    """
    Get comprehensive status of the orchestration system.
    """
    try:
        await initialize_orchestration()
        
        if not orchestration_manager:
            return func.HttpResponse(
                json.dumps({"error": "Orchestration manager not initialized"}),
                status_code=503,
                mimetype="application/json"
            )
        
        # Get orchestration status with performance metrics
        status = orchestration_manager.get_orchestration_status()
        status['performance_metrics'] = {
            'cache_stats': {
                'hits': metrics['cache_hits'],
                'misses': metrics['cache_misses']
            },
            'response_times': {
                'incident_resolution': metrics['incident_resolution_time'] / max(metrics['incident_resolution_count'], 1),
                'pattern_analysis': metrics['pattern_analysis_time'] / max(metrics['pattern_analysis_count'], 1),
                'proactive_monitoring': metrics['proactive_monitoring_time'] / max(metrics['proactive_monitoring_count'], 1)
            },
            'error_rates': {
                'incident_resolution': metrics['incident_resolution_errors'],
                'pattern_analysis': metrics['pattern_analysis_errors'],
                'proactive_monitoring': metrics['proactive_monitoring_errors']
            },
            'resource_usage': {
                'connection_pool': {
                    'active_connections': len(connection_pool._acquired) if connection_pool else 0,
                    'idle_connections': len(connection_pool._free) if connection_pool else 0
                },
                'thread_pool': {
                    'active_threads': executor._work_queue.qsize() if executor else 0,
                    'max_threads': executor._max_workers if executor else 0
                } if executor else {}
            }
        }
        
        return func.HttpResponse(
            json.dumps(status, indent=2),
            status_code=200,
            mimetype="application/json"
        )
        
    except Exception as e:
        logger.error(f"Error getting orchestration status: {str(e)}")
        return func.HttpResponse(
            json.dumps({
                "error": "Failed to get orchestration status",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }),
            status_code=500,
            mimetype="application/json"
        )

@app.route(route="orchestration-health", methods=["GET"])
@rate_limit
@track_performance
@cache_response(ttl=30)  # Cache health status for 30 seconds
async def orchestration_health(req: func.HttpRequest) -> func.HttpResponse:
    """
    Comprehensive health check for the orchestration system.
    """
    try:
        await initialize_orchestration()
        
        if not orchestration_manager:
            return func.HttpResponse(
                json.dumps({
                    "status": "unhealthy",
                    "error": "Orchestration manager not initialized",
                    "timestamp": datetime.now().isoformat()
                }),
                status_code=503,
                mimetype="application/json"
            )
        
        # Perform comprehensive health check with connection pooling
        health_monitor = OrchestrationHealthMonitor(
            orchestration_manager,
            connection_pool=connection_pool,
            executor=executor
        )
        health_status = await health_monitor.perform_health_check()
        
        # Add performance metrics
        health_status['performance'] = {
            'cache_efficiency': metrics['cache_hits'] / max(metrics['cache_hits'] + metrics['cache_misses'], 1),
            'average_response_times': {
                'incident_resolution': metrics['incident_resolution_time'] / max(metrics['incident_resolution_count'], 1),
                'pattern_analysis': metrics['pattern_analysis_time'] / max(metrics['pattern_analysis_count'], 1)
            },
            'error_rates': {
                'incident_resolution': metrics['incident_resolution_errors'] / max(metrics['incident_resolution_count'], 1),
                'pattern_analysis': metrics['pattern_analysis_errors'] / max(metrics['pattern_analysis_count'], 1)
            }
        }
        
        # Add Azure services health
        if azure_manager:
            azure_health = await azure_manager.get_health_status()
            health_status['azure_services'] = azure_health
        
        # Add resource utilization
        health_status['resource_utilization'] = {
            'connection_pool': {
                'used_connections': len(connection_pool._acquired) if connection_pool else 0,
                'available_connections': CONNECTION_POOL_SIZE - len(connection_pool._acquired) if connection_pool else 0,
                'utilization_percent': (len(connection_pool._acquired) / CONNECTION_POOL_SIZE * 100) if connection_pool else 0
            },
            'thread_pool': {
                'active_threads': executor._work_queue.qsize() if executor else 0,
                'max_threads': THREAD_POOL_SIZE,
                'utilization_percent': (executor._work_queue.qsize() / THREAD_POOL_SIZE * 100) if executor else 0
            } if executor else {}
        }
        
        status_code = 200 if health_status['overall_health'] == 'healthy' else 503
        
        return func.HttpResponse(
            json.dumps(health_status, indent=2),
            status_code=status_code,
            mimetype="application/json"
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return func.HttpResponse(
            json.dumps({
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }),
            status_code=503,
            mimetype="application/json"
        )

@app.route(route="optimize-performance", methods=["POST"])
@rate_limit
@track_performance
async def optimize_performance(req: func.HttpRequest) -> func.HttpResponse:
    """
    Trigger system performance optimization.
    """
    try:
        await initialize_orchestration()
        
        if not orchestration_manager:
            return func.HttpResponse(
                json.dumps({"error": "Orchestration manager not initialized"}),
                status_code=503,
                mimetype="application/json"
            )
        
        # Clear expired cache entries
        now = datetime.now()
        expired_keys = [k for k, v in cache_ttl.items() if now > v]
        for k in expired_keys:
            del cache[k]
            del cache_ttl[k]
        
        # Reset connection pool if needed
        if connection_pool and len(connection_pool._acquired) / CONNECTION_POOL_SIZE > 0.8:
            await connection_pool.close()
            connection_pool = aiohttp.TCPConnector(
                limit=CONNECTION_POOL_SIZE,
                ttl_dns_cache=300,
                keepalive_timeout=60
            )
        
        # Trigger performance optimization
        optimization_result = await orchestration_manager.optimize_system_performance()
        optimization_result['optimizations_performed'] = {
            'cache_entries_cleared': len(expired_keys),
            'connection_pool_reset': bool(connection_pool),
            'metrics_collected': bool(metrics)
        }
        
        return func.HttpResponse(
            json.dumps(optimization_result, indent=2),
            status_code=200,
            mimetype="application/json"
        )
        
    except Exception as e:
        logger.error(f"Error optimizing performance: {str(e)}")
        return func.HttpResponse(
            json.dumps({
                "error": "Performance optimization failed",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }),
            status_code=500,
            mimetype="application/json"
        )

@app.route(route="workflow-status", methods=["GET"])
@rate_limit
@track_performance
@cache_response(ttl=15)  # Cache workflow status for 15 seconds
async def workflow_status(req: func.HttpRequest) -> func.HttpResponse:
    """
    Get status of a specific workflow execution with performance metrics.
    Query parameter: execution_id
    """
    try:
        await initialize_orchestration()
        
        execution_id = req.params.get('execution_id')
        if not execution_id:
            return func.HttpResponse(
                json.dumps({"error": "execution_id parameter is required"}),
                status_code=400,
                mimetype="application/json"
            )
        
        # Get workflow status with performance tracking
        start_time = datetime.now()
        
        # Check coordinator status
        coordinator_status = await orchestration_manager.coordinator.get_workflow_status(execution_id)
        if coordinator_status:
            processing_time = (datetime.now() - start_time).total_seconds()
            coordinator_status['performance'] = {
                'lookup_time': processing_time,
                'source': 'coordinator'
            }
            return func.HttpResponse(
                json.dumps(coordinator_status, indent=2),
                status_code=200,
                mimetype="application/json"
            )
        
        # Check workflow engine
        workflow_status = await orchestration_manager.workflow_engine.get_execution_status(execution_id)
        if workflow_status:
            processing_time = (datetime.now() - start_time).total_seconds()
            workflow_status['performance'] = {
                'lookup_time': processing_time,
                'source': 'workflow_engine'
            }
            return func.HttpResponse(
                json.dumps(workflow_status, indent=2),
                status_code=200,
                mimetype="application/json"
            )
        
        # Check collaboration manager
        collaboration_status = await orchestration_manager.collaboration_manager.get_collaboration_status(execution_id)
        if collaboration_status:
            processing_time = (datetime.now() - start_time).total_seconds()
            collaboration_status['performance'] = {
                'lookup_time': processing_time,
                'source': 'collaboration_manager'
            }
            return func.HttpResponse(
                json.dumps(collaboration_status, indent=2),
                status_code=200,
                mimetype="application/json"
            )
        
        return func.HttpResponse(
            json.dumps({"error": f"Workflow with ID {execution_id} not found"}),
            status_code=404,
            mimetype="application/json"
        )
        
    except Exception as e:
        logger.error(f"Error getting workflow status: {str(e)}")
        return func.HttpResponse(
            json.dumps({
                "error": "Failed to get workflow status",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }),
            status_code=500,
            mimetype="application/json"
        )

@app.function_name(name="cleanup")
async def cleanup_resources():
    """Clean up resources on function app shutdown"""
    global orchestration_manager, message_bus, registry, connection_pool, executor
    
    try:
        logger.info("Starting orchestration service cleanup...")
        
        # Clean up orchestration components
        if orchestration_manager:
            await orchestration_manager.stop()
            logger.info("Orchestration manager stopped")
            
        if registry:
            await registry.stop()
            logger.info("Registry stopped")
            
        if message_bus:
            await message_bus.stop()
            logger.info("Message bus stopped")
        
        # Clean up performance components
        if connection_pool:
            await connection_pool.close()
            logger.info("Connection pool closed")
            
        if executor:
            executor.shutdown(wait=True)
            logger.info("Thread pool executor shut down")
        
        # Clear caches
        cache.clear()
        cache_ttl.clear()
        rate_limit_cache.clear()
        metrics.clear()
        
        logger.info("Orchestration service resources cleaned up successfully")
        
    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}")
        raise

@app.function_name(name="startup")
async def startup():
    """Initialize resources on function app startup"""
    try:
        logger.info("Starting orchestration service initialization...")
        await initialize_orchestration()
        logger.info("Orchestration service initialized successfully")
        
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        raise