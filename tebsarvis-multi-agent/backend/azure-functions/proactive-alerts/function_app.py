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
from typing import Dict, Any, Optional

# Add the backend path to sys.path to import our agents
backend_path = os.path.join(os.path.dirname(__file__), '..', '..', 'agents')
sys.path.append(backend_path)

from proactive.alerting_agent import AlertingAgent
from core.agent_registry import get_global_registry
from core.agent_communication import MessageBus
from shared.azure_clients import AzureClientManager

# Initialize global components
app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)
logger = logging.getLogger(__name__)

# Global variables for agent instances
alerting_agent = None
azure_manager = None
message_bus = None
registry = None

async def initialize_components():
    """Initialize agents and Azure components"""
    global alerting_agent, azure_manager, message_bus, registry
    
    try:
        if not alerting_agent:
            # Initialize Azure manager
            azure_manager = AzureClientManager()
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
            
            logger.info("Alerting Agent components initialized successfully")
            
    except Exception as e:
        logger.error(f"Error initializing components: {str(e)}")
        raise

@app.route(route="proactive-alerts", methods=["POST"])
async def proactive_alerts(req: func.HttpRequest) -> func.HttpResponse:
    """
    Generate proactive alerts based on monitoring rules and patterns.
    
    Expected request body:
    {
        "monitoring_type": "real_time_monitoring",  // "real_time_monitoring", "threshold_monitoring", "predictive_alerting"
        "monitoring_window": {
            "minutes": 15
        },
        "rule_ids": ["volume_spike", "pattern_anomaly"],
        "notification_channels": ["email", "teams"]
    }
    
    Returns:
    {
        "generated_alerts": [...],
        "evaluated_rules": 5,
        "active_alerts_count": 3,
        "monitoring_metadata": {...}
    }
    """
    
    logger.info("Proactive alerts request received")
    
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
        
        # Get monitoring type
        monitoring_type = request_data.get('monitoring_type', 'real_time_monitoring')
        monitoring_window = request_data.get('monitoring_window', {'minutes': 15})
        rule_ids = request_data.get('rule_ids', [])
        notification_channels = request_data.get('notification_channels', ['email'])
        
        # Validate monitoring type
        valid_types = ['real_time_monitoring', 'threshold_monitoring', 'predictive_alerting', 'alert_management']
        if monitoring_type not in valid_types:
            return func.HttpResponse(
                json.dumps({"error": f"Invalid monitoring_type. Must be one of: {valid_types}"}),
                status_code=400,
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
                'metrics_data': request_data.get('metrics_data', {}),
                'threshold_config': request_data.get('threshold_config', {})
            })
        elif monitoring_type == 'predictive_alerting':
            task_data.update({
                'trend_data': request_data.get('trend_data', {}),
                'prediction_horizon': request_data.get('prediction_horizon', {'hours': 24}),
                'confidence_threshold': request_data.get('confidence_threshold', 0.7)
            })
        elif monitoring_type == 'alert_management':
            task_data.update({
                'action': request_data.get('action', 'list'),
                'alert_ids': request_data.get('alert_ids', []),
                'user_id': request_data.get('user_id', 'system')
            })
        
        # Process the alerting request
        start_time = datetime.now()
        result = await alerting_agent.process_task(task_data)
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Add processing metadata
        result['monitoring_metadata'] = result.get('monitoring_metadata', {})
        result['monitoring_metadata'].update({
            'function_name': 'proactive-alerts',
            'processing_time_seconds': processing_time,
            'timestamp': datetime.now().isoformat(),
            'api_version': '1.0.0'
        })
        
        # Add request tracking
        result['request_id'] = req.headers.get('x-request-id', 'unknown')
        
        logger.info(f"Alerting request processed successfully in {processing_time:.2f}s")
        
        return func.HttpResponse(
            json.dumps(result, indent=2),
            status_code=200,
            mimetype="application/json"
        )
        
    except Exception as e:
        logger.error(f"Error processing alerting request: {str(e)}")
        
        error_response = {
            "error": "Internal server error",
            "message": str(e),
            "timestamp": datetime.now().isoformat(),
            "function_name": "proactive-alerts"
        }
        
        return func.HttpResponse(
            json.dumps(error_response),
            status_code=500,
            mimetype="application/json"
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