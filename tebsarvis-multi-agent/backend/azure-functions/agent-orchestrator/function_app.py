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
from datetime import datetime
from typing import Dict, Any, Optional

# Add the backend path to sys.path to import our agents
backend_path = os.path.join(os.path.dirname(__file__), '..', '..', 'agents')
sys.path.append(backend_path)

# FIXED IMPORTS - Using OrchestrationManager for unified orchestration
from ...agents.orchestrator.orchestration_manager import OrchestrationManager, OrchestrationHealthMonitor
from ...agents.core.agent_registry import get_global_registry
from ...agents.core.agent_communication import MessageBus
from ..shared.azure_clients import AzureClientManager

# Initialize global components
app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)
logger = logging.getLogger(__name__)

# Global variables for orchestration components
orchestration_manager = None
azure_manager = None
message_bus = None
registry = None

async def initialize_orchestration():
    """Initialize orchestration manager and all components"""
    global orchestration_manager, azure_manager, message_bus, registry
    
    try:
        if not orchestration_manager:
            # Initialize Azure manager
            azure_manager = AzureClientManager()
            await azure_manager.initialize()
            
            # Initialize message bus and registry
            message_bus = MessageBus()
            registry = get_global_registry()
            
            # Initialize unified orchestration manager
            orchestration_manager = OrchestrationManager(registry, message_bus)
            await orchestration_manager.start()
            
            logger.info("Orchestration Manager initialized successfully")
            
    except Exception as e:
        logger.error(f"Error initializing orchestration: {str(e)}")
        raise

@app.route(route="coordinate-incident-resolution", methods=["POST"])
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
        
        # Parse request
        req_body = req.get_json()
        if not req_body or 'incident_data' not in req_body:
            return func.HttpResponse(
                json.dumps({"error": "Missing incident_data in request"}),
                status_code=400,
                mimetype="application/json"
            )
        
        incident_data = req_body['incident_data']
        orchestration_options = req_body.get('orchestration_options', {})
        
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
        
        # Execute with intelligent orchestration
        start_time = datetime.now()
        result = await orchestration_manager.execute_intelligent_workflow(workflow_request)
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Add API metadata
        result['api_metadata'] = {
            'function_name': 'coordinate-incident-resolution',
            'processing_time_seconds': processing_time,
            'timestamp': datetime.now().isoformat(),
            'api_version': '1.0.0',
            'orchestration_method': result.get('execution_method', 'intelligent')
        }
        
        # Add request tracking
        result['request_id'] = req.headers.get('x-request-id', 'unknown')
        
        logger.info(f"Incident resolution coordinated successfully in {processing_time:.2f}s")
        
        return func.HttpResponse(
            json.dumps(result, indent=2),
            status_code=200,
            mimetype="application/json"
        )
        
    except Exception as e:
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
        
        # Parse request
        req_body = req.get_json()
        analysis_request = req_body.get('analysis_request', {"time_range": {"days": 30}})
        orchestration_options = req_body.get('orchestration_options', {})
        
        # Prepare intelligent workflow request
        workflow_request = {
            'type': 'pattern_analysis',
            'complexity': orchestration_options.get('complexity', 'high'),
            'requires_consensus': False,  # Pattern analysis doesn't typically need consensus
            'agents_required': ['pattern_detection', 'alerting'],
            'analysis_request': analysis_request,
            'timeout_minutes': orchestration_options.get('timeout_minutes', 45)
        }
        
        # Execute with intelligent orchestration
        start_time = datetime.now()
        result = await orchestration_manager.execute_intelligent_workflow(workflow_request)
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Add API metadata
        result['api_metadata'] = {
            'function_name': 'coordinate-pattern-analysis',
            'processing_time_seconds': processing_time,
            'timestamp': datetime.now().isoformat(),
            'api_version': '1.0.0',
            'orchestration_method': result.get('execution_method', 'intelligent')
        }
        
        logger.info(f"Pattern analysis coordinated successfully in {processing_time:.2f}s")
        
        return func.HttpResponse(
            json.dumps(result, indent=2),
            status_code=200,
            mimetype="application/json"
        )
        
    except Exception as e:
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
        
        # Parse request
        req_body = req.get_json()
        monitoring_config = req_body.get('monitoring_config', {
            "monitoring_rules": ["volume_spike"],
            "notification_channels": ["email"]
        })
        orchestration_options = req_body.get('orchestration_options', {})
        
        # Prepare intelligent workflow request
        workflow_request = {
            'type': 'proactive_monitoring',
            'complexity': orchestration_options.get('complexity', 'medium'),
            'requires_consensus': False,
            'agents_required': ['pattern_detection', 'alerting'],
            'monitoring_config': monitoring_config,
            'timeout_minutes': orchestration_options.get('timeout_minutes', 10)
        }
        
        # Execute with intelligent orchestration
        start_time = datetime.now()
        result = await orchestration_manager.execute_intelligent_workflow(workflow_request)
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Add API metadata
        result['api_metadata'] = {
            'function_name': 'coordinate-proactive-monitoring',
            'processing_time_seconds': processing_time,
            'timestamp': datetime.now().isoformat(),
            'api_version': '1.0.0',
            'orchestration_method': result.get('execution_method', 'intelligent')
        }
        
        logger.info(f"Proactive monitoring coordinated successfully in {processing_time:.2f}s")
        
        return func.HttpResponse(
            json.dumps(result, indent=2),
            status_code=200,
            mimetype="application/json"
        )
        
    except Exception as e:
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
        
        # Get orchestration status
        status = orchestration_manager.get_orchestration_status()
        
        return func.HttpResponse(
            json.dumps(status, indent=2),
            status_code=200,
            mimetype="application/json"
        )
        
    except Exception as e:
        logger.error(f"Error getting orchestration status: {str(e)}")
        
        error_response = {
            "error": "Failed to get orchestration status",
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }
        
        return func.HttpResponse(
            json.dumps(error_response),
            status_code=500,
            mimetype="application/json"
        )

@app.route(route="orchestration-health", methods=["GET"])
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
        
        # Perform comprehensive health check
        health_monitor = OrchestrationHealthMonitor(orchestration_manager)
        health_status = await health_monitor.perform_health_check()
        
        # Add Azure services health
        if azure_manager:
            azure_health = await azure_manager.get_health_status()
            health_status['azure_services'] = azure_health
        
        status_code = 200 if health_status['overall_health'] == 'healthy' else 503
        
        return func.HttpResponse(
            json.dumps(health_status, indent=2),
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

@app.route(route="optimize-performance", methods=["POST"])
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
        
        # Trigger performance optimization
        optimization_result = await orchestration_manager.optimize_system_performance()
        
        return func.HttpResponse(
            json.dumps(optimization_result, indent=2),
            status_code=200,
            mimetype="application/json"
        )
        
    except Exception as e:
        logger.error(f"Error optimizing performance: {str(e)}")
        
        error_response = {
            "error": "Performance optimization failed",
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }
        
        return func.HttpResponse(
            json.dumps(error_response),
            status_code=500,
            mimetype="application/json"
        )

@app.route(route="workflow-status", methods=["GET"])
async def workflow_status(req: func.HttpRequest) -> func.HttpResponse:
    """
    Get status of a specific workflow execution.
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
        
        # Get workflow status from coordinator or workflow engine
        coordinator_status = await orchestration_manager.coordinator.get_workflow_status(execution_id)
        if coordinator_status:
            return func.HttpResponse(
                json.dumps(coordinator_status, indent=2),
                status_code=200,
                mimetype="application/json"
            )
        
        # Check workflow engine
        workflow_status = await orchestration_manager.workflow_engine.get_execution_status(execution_id)
        if workflow_status:
            return func.HttpResponse(
                json.dumps(workflow_status, indent=2),
                status_code=200,
                mimetype="application/json"
            )
        
        # Check collaboration manager
        collaboration_status = await orchestration_manager.collaboration_manager.get_collaboration_status(execution_id)
        if collaboration_status:
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
        
        error_response = {
            "error": "Failed to get workflow status",
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
    global orchestration_manager, message_bus, registry
    
    try:
        if orchestration_manager:
            await orchestration_manager.stop()
        if registry:
            await registry.stop()
        if message_bus:
            await message_bus.stop()
            
        logger.info("Orchestration service resources cleaned up successfully")
        
    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}")