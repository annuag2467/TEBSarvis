"""
Azure Function App for Alerting Agent API
Endpoint: /proactive-alerts
Provides real-time monitoring, alerting, and notification capabilities.
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
    get_client_ip
)
from ..shared.azure_clients import AzureClientManager

# Add the backend path to sys.path to import our agents
# backend_path = os.path.join(os.path.dirname(__file__), '..', '..', 'agents')
# sys.path.append(backend_path)

from ...agents.proactive.alerting_agent import AlertingAgent
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
alerting_agent = None
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

async def initialize_components():
    """Initialize agents and Azure components with connection pooling and monitoring"""
    global alerting_agent, azure_manager, message_bus, registry, connection_pool, executor
    
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

        if not alerting_agent:
            # Initialize Azure manager with connection pool
            azure_manager = AzureClientManager(connector=connection_pool)
            await azure_manager.initialize()
            
            # Initialize message bus and registry
            message_bus = MessageBus()
            await message_bus.start()
            
            registry = get_global_registry()
            await registry.start()
            
            # Initialize alerting agent
            alerting_agent = AlertingAgent()
            await alerting_agent.start()
            
            # Register agent
            await registry.register_agent(alerting_agent)
            
            # Setup monitoring
            setup_monitoring(app_name="proactive-alerts")
            
            logger.info("Alerting Agent components initialized successfully")
            
    except Exception as e:
        logger.error(f"Error initializing components: {str(e)}", exc_info=True)
        # Cleanup on failure
        if connection_pool:
            await connection_pool.close()
        if executor:
            executor.shutdown(wait=False)
        raise

@app.route(route="proactive-alerts", methods=["POST"])
async def proactive_alerts(req: func.HttpRequest) -> func.HttpResponse:
    """
    Generate proactive alerts based on monitoring rules and patterns.
    
    Expected request body:
    {
        "monitoring_type": "real_time_monitoring",
        "monitoring_window": {"minutes": 15},
        "rule_ids": ["volume_spike", "pattern_anomaly"],
        "notification_channels": ["email", "teams"]
    }
    """
    request_id = req.headers.get('x-request-id', f"req_{int(time.time()*1000)}")
    client_ip = get_client_ip(req)
    start_time = time.time()
    
    try:
        # Setup request logging context
        logger.info(f"Processing alert request {request_id} from {client_ip}")
        
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
            required_fields=['monitoring_type'],
            optional_fields=['monitoring_window', 'rule_ids', 'notification_channels']
        )
        if error_response:
            return error_response
            
        # Sanitize input data
        monitoring_type = sanitize_input(data['monitoring_type'])
        monitoring_window = sanitize_input(data.get('monitoring_window', {'minutes': 15}))
        rule_ids = [sanitize_input(rule_id) for rule_id in data.get('rule_ids', [])]
        notification_channels = [sanitize_input(channel) for channel in data.get('notification_channels', ['email'])]

        # Validate monitoring type
        valid_types = ['real_time_monitoring', 'threshold_monitoring', 'predictive_alerting', 'alert_management']
        if monitoring_type not in valid_types:
            return create_error_response(
                f"Invalid monitoring_type. Must be one of: {valid_types}",
                400
            )

        # Cache check for recent alert queries
        cache_key = f"{monitoring_type}:{json.dumps(monitoring_window)}:{':'.join(rule_ids)}"
        cached_result = None
        if monitoring_type != 'real_time_monitoring':  # Don't cache real-time monitoring
            cached_result = cache.get(cache_key)
            if cached_result and time.time() - cache_ttl.get(cache_key, 0) < CACHE_DEFAULT_TTL:
                logger.info(f"Cache hit for request {request_id}")
                return func.HttpResponse(
                    json.dumps(cached_result, indent=2),
                    status_code=200,
                    mimetype="application/json"
                )
        
        # Prepare task for alerting agent
        task_data = {
            'type': monitoring_type,
            'monitoring_window': monitoring_window,
            'rule_ids': rule_ids,
            'notification_channels': notification_channels
        }
        
        # Add specific parameters based on monitoring type
        if monitoring_type == 'threshold_monitoring':
            task_data.update({
                'metrics_data': data.get('metrics_data', {}),
                'threshold_config': data.get('threshold_config', {})
            })
        elif monitoring_type == 'predictive_alerting':
            task_data.update({
                'trend_data': data.get('trend_data', {}),
                'prediction_horizon': data.get('prediction_horizon', {'hours': 24}),
                'confidence_threshold': data.get('confidence_threshold', 0.7)
            })
        elif monitoring_type == 'alert_management':
            task_data.update({
                'action': data.get('action', 'list'),
                'alert_ids': data.get('alert_ids', []),
                'user_id': data.get('user_id', 'system')
            })
        
        # Process with alerting agent
        result = await alerting_agent.process_task(task_data)
        processing_time = time.time() - start_time
        
        # Format response
        response = {
            "monitoring_type": monitoring_type,
            "generated_alerts": result.get('alerts', []),
            "evaluated_rules": result.get('evaluated_rules', 0),
            "active_alerts_count": result.get('active_alerts_count', 0),
            "monitoring_metadata": {
                "function_name": "proactive-alerts",
                "processing_time_seconds": processing_time,
                "timestamp": datetime.now().isoformat(),
                "request_id": request_id,
                "api_version": "1.0.0",
                "cache_hit": False
            }
        }
        
        # Add processing metadata
        if 'processing_metadata' in result:
            response['monitoring_metadata'].update(result['processing_metadata'])
        
        # Cache successful results if not real-time monitoring
        if monitoring_type != 'real_time_monitoring':
            cache[cache_key] = response
            cache_ttl[cache_key] = time.time()
        
        logger.info(f"Alert request {request_id} processed successfully in {processing_time:.2f}s")
        
        return func.HttpResponse(
            json.dumps(response, indent=2),
            status_code=200,
            mimetype="application/json"
        )
        
    except Exception as e:
        error_time = time.time() - start_time
        logger.error(f"Error in request {request_id} after {error_time:.2f}s: {str(e)}", exc_info=True)
        return create_error_response(
            f"Alert processing failed: {str(e)}",
            500,
            {"request_id": request_id, "processing_time": error_time}
        )

@app.route(route="alert-management", methods=["POST"])
async def manage_alerts(req: func.HttpRequest) -> func.HttpResponse:
    """
    Manage alert lifecycle (acknowledge, resolve, suppress).
    
    Expected request body:
    {
        "action": "acknowledge",  // "acknowledge", "resolve", "suppress", "list"
        "alert_ids": ["alert_123", "alert_456"],
        "user_id": "admin_user",
        "suppress_duration": {"hours": 2}  // for suppress action
    }
    """
    
    logger.info("Alert management request received")
    
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
        action = request_data.get('action', 'list')
        valid_actions = ['acknowledge', 'resolve', 'suppress', 'list']
        
        if action not in valid_actions:
            return func.HttpResponse(
                json.dumps({"error": f"Invalid action. Must be one of: {valid_actions}"}),
                status_code=400,
                mimetype="application/json"
            )
        
        # Prepare alert management task
        task_data = {
            'type': 'alert_management',
            'action': action,
            'alert_ids': request_data.get('alert_ids', []),
            'user_id': request_data.get('user_id', 'system')
        }
        
        # Add suppress duration if applicable
        if action == 'suppress':
            task_data['suppress_duration'] = request_data.get('suppress_duration', {'hours': 1})
        
        # Process alert management
        result = await alerting_agent.process_task(task_data)
        
        return func.HttpResponse(
            json.dumps(result, indent=2),
            status_code=200,
            mimetype="application/json"
        )
        
    except Exception as e:
        logger.error(f"Error in alert management: {str(e)}")
        
        error_response = {
            "error": "Alert management failed",
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }
        
        return func.HttpResponse(
            json.dumps(error_response),
            status_code=500,
            mimetype="application/json"
        )

@app.route(route="notification-dispatch", methods=["POST"])
async def dispatch_notifications(req: func.HttpRequest) -> func.HttpResponse:
    """
    Dispatch notifications through multiple channels.
    
    Expected request body:
    {
        "alerts": [...],
        "notification_channels": ["email", "teams", "slack"],
        "override_config": {
            "urgent": true,
            "escalation": false
        }
    }
    """
    
    logger.info("Notification dispatch request received")
    
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
        if 'alerts' not in request_data:
            return func.HttpResponse(
                json.dumps({"error": "alerts field is required"}),
                status_code=400,
                mimetype="application/json"
            )
        
        # Prepare notification dispatch task
        task_data = {
            'type': 'notification_dispatch',
            'alerts': request_data['alerts'],
            'notification_channels': request_data.get('notification_channels', ['email']),
            'override_config': request_data.get('override_config', {})
        }
        
        # Process notification dispatch
        result = await alerting_agent.process_task(task_data)
        
        return func.HttpResponse(
            json.dumps(result, indent=2),
            status_code=200,
            mimetype="application/json"
        )
        
    except Exception as e:
        logger.error(f"Error dispatching notifications: {str(e)}")
        
        error_response = {
            "error": "Notification dispatch failed",
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }
        
        return func.HttpResponse(
            json.dumps(error_response),
            status_code=500,
            mimetype="application/json"
        )

@app.route(route="alert-statistics", methods=["GET"])
async def alert_statistics(req: func.HttpRequest) -> func.HttpResponse:
    """
    Get alerting statistics and metrics.
    """
    
    try:
        await initialize_components()
        
        if not alerting_agent:
            return func.HttpResponse(
                json.dumps({"error": "Alerting agent not initialized"}),
                status_code=503,
                mimetype="application/json"
            )
        
        # Get alert statistics
        stats = alerting_agent.get_alert_statistics()
        
        return func.HttpResponse(
            json.dumps(stats, indent=2),
            status_code=200,
            mimetype="application/json"
        )
        
    except Exception as e:
        logger.error(f"Error getting alert statistics: {str(e)}")
        
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

@app.route(route="alert-health", methods=["GET"])
async def alert_health(req: func.HttpRequest) -> func.HttpResponse:
    """
    Health check endpoint for the alerting service.
    """
    
    try:
        await initialize_components()
        
        # Check agent status
        agent_status = alerting_agent.get_status() if alerting_agent else {"status": "not_initialized"}
        
        # Check Azure services
        azure_health = await azure_manager.get_health_status() if azure_manager else {"status": "not_initialized"}
        
        # Test alerting functionality
        alert_test = {"status": "not_tested"}
        if alerting_agent:
            try:
                test_task = {
                    'type': 'real_time_monitoring',
                    'monitoring_window': {'minutes': 1},
                    'rule_ids': []
                }
                test_result = await alerting_agent.process_task(test_task)
                alert_test = {"status": "healthy", "test_result": "success"}
            except Exception as e:
                alert_test = {"status": "unhealthy", "error": str(e)}
        
        health_data = {
            "status": "healthy" if alerting_agent else "unhealthy",
            "agent_status": agent_status,
            "azure_services": azure_health,
            "alerting_functionality": alert_test,
            "active_alerts": len(alerting_agent.active_alerts) if alerting_agent else 0,
            "alert_rules": len(alerting_agent.alert_rules) if alerting_agent else 0,
            "timestamp": datetime.now().isoformat(),
            "uptime": "available" if alerting_agent else "unavailable"
        }
        
        status_code = 200 if alerting_agent else 503
        
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

@app.route(route="alerting-health", methods=["GET"])
async def alerting_health(req: func.HttpRequest) -> func.HttpResponse:
    """
    Health check endpoint for the alerting service.
    """
    try:
        health_tasks = []
        
        # Check agent initialization
        async def check_agent():
            try:
                await initialize_components()
                agent_status = alerting_agent.get_status() if alerting_agent else {"status": "not_initialized"}
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
        
        # Check monitoring rules
        async def check_monitoring_rules():
            try:
                if alerting_agent:
                    rules = await alerting_agent.get_active_rules()
                    return {
                        "monitoring_rules": {
                            "status": "healthy",
                            "active_rules": len(rules),
                            "last_updated": alerting_agent.rules_last_updated
                        }
                    }
                return {"monitoring_rules": {"status": "not_initialized"}}
            except Exception as e:
                return {"monitoring_rules": {"status": "unhealthy", "error": str(e)}}

        # Run all health checks in parallel
        health_tasks = [
            check_agent(),
            check_azure(),
            check_message_bus(),
            check_monitoring_rules()
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
                "active_alerts": len(alerting_agent.active_alerts) if alerting_agent else 0,
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

# Graceful shutdown handler
@app.function_name("cleanup")
async def cleanup_resources():
    """Clean up resources on function app shutdown"""
    global alerting_agent, message_bus, registry
    
    try:
        if alerting_agent:
            await alerting_agent.stop()
        if registry:
            await registry.stop()
        if message_bus:
            await message_bus.stop()
            
        logger.info("Alerting service resources cleaned up successfully")
        
    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}")