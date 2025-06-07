"""
Azure Function App for Resolution Agent API  
Endpoint: /recommend-resolution
Provides AI-powered resolution recommendations with ranking and validation.
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
from ...agents.reactive.resolution_agent import ResolutionAgent
from ...agents.core.agent_registry import get_global_registry
from ...agents.core.agent_communication import MessageBus
from ..shared.azure_clients import AzureClientManager

# Initialize global components
app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)
logger = logging.getLogger(__name__)

# Global variables for agent instances
resolution_agent = None
azure_manager = None
message_bus = None
registry = None

async def initialize_components():
    """Initialize agents and Azure components"""
    global resolution_agent, azure_manager, message_bus, registry
    
    try:
        if not resolution_agent:
            # Initialize Azure manager
            azure_manager = AzureClientManager()
            await azure_manager.initialize()
            
            # Initialize message bus and registry
            message_bus = MessageBus()
            await message_bus.start()
            
            registry = get_global_registry()
            await registry.start()
            
            # Initialize resolution agent
            resolution_agent = ResolutionAgent()
            await resolution_agent.start()
            
            # Register agent
            await registry.register_agent(resolution_agent)
            
            logger.info("Resolution Agent components initialized successfully")
            
    except Exception as e:
        logger.error(f"Error initializing components: {str(e)}")
        raise

@app.route(route="recommend-resolution", methods=["POST"])
async def recommend_resolution(req: func.HttpRequest) -> func.HttpResponse:
    """
    Generate resolution recommendations for an incident.
    
    Expected JSON payload:
    {
        "incident_data": {
            "id": "INC001",
            "summary": "User cannot access LMS",
            "description": "Multiple users reporting login issues",
            "category": "Learning Management System (LMS)",
            "severity": "High",
            "priority": "High",
            "environment": "production",
            "affected_users": 150
        },
        "resolution_options": {
            "max_solutions": 5,
            "include_similar_incidents": true,
            "detailed_steps": true,
            "confidence_threshold": 0.5
        }
    }
    
    Returns:
    {
        "incident_id": "INC001",
        "recommendations": [...],
        "overall_confidence": 0.85,
        "similar_incidents_used": 3,
        "resolution_metadata": {...}
    }
    """
    try:
        await initialize_components()
        
        # Parse request
        req_body = req.get_json()
        if not req_body or 'incident_data' not in req_body:
            return func.HttpResponse(
                json.dumps({"error": "Missing incident_data in request"}),
                status_code=400,
                mimetype="application/json"
            )
        
        incident_data = req_body['incident_data']
        resolution_options = req_body.get('resolution_options', {})
        
        # Validate required fields in incident_data
        required_fields = ['summary']
        missing_fields = [field for field in required_fields if field not in incident_data]
        if missing_fields:
            return func.HttpResponse(
                json.dumps({"error": f"Missing required fields in incident_data: {missing_fields}"}),
                status_code=400,
                mimetype="application/json"
            )
        
        # Prepare task for resolution agent
        task_data = {
            'type': 'incident_resolution',
            'incident_data': incident_data,
            'max_solutions': resolution_options.get('max_solutions', 5),
            'include_similar_incidents': resolution_options.get('include_similar_incidents', True),
            'detailed_steps': resolution_options.get('detailed_steps', True),
            'confidence_threshold': resolution_options.get('confidence_threshold', 0.5)
        }
        
        # Process with resolution agent
        start_time = datetime.now()
        result = await resolution_agent.process_task(task_data)
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Format response
        response = {
            "incident_id": incident_data.get('id', 'unknown'),
            "recommendations": result.get('solutions', []),
            "overall_confidence": result.get('overall_confidence', 0.0),
            "similar_incidents_used": result.get('similar_incidents_used', 0),
            "resolution_metadata": {
                "function_name": "recommend-resolution",
                "processing_time_seconds": processing_time,
                "timestamp": datetime.now().isoformat(),
                "api_version": "1.0.0",
                "rag_enabled": result.get('rag_enabled', False),
                "templates_used": result.get('templates_used', [])
            }
        }
        
        # Add processing metadata from agent
        if 'processing_metadata' in result:
            response['resolution_metadata'].update(result['processing_metadata'])
        
        # Add request tracking
        response['request_id'] = req.headers.get('x-request-id', 'unknown')
        
        logger.info(f"Resolution request processed successfully in {processing_time:.2f}s")
        
        return func.HttpResponse(
            json.dumps(response, indent=2),
            status_code=200,
            mimetype="application/json"
        )
        
    except Exception as e:
        logger.error(f"Error in recommend-resolution: {str(e)}")
        return func.HttpResponse(
            json.dumps({
                "error": "Resolution recommendation failed",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }),
            status_code=500,
            mimetype="application/json"
        )

@app.route(route="solution-ranking", methods=["POST"])
async def rank_solutions(req: func.HttpRequest) -> func.HttpResponse:
    """
    Rank and score solution alternatives.
    
    Expected request body:
    {
        "solution_candidates": [...],
        "incident_context": {...},
        "ranking_criteria": {
            "effectiveness_weight": 0.4,
            "complexity_weight": 0.3,
            "time_weight": 0.3
        }
    }
    """
    
    logger.info("Solution ranking request received")
    
    try:
        await initialize_components()
        
        # Parse request
        request_data = req.get_json()
        if not request_data or 'solution_candidates' not in request_data:
            return func.HttpResponse(
                json.dumps({"error": "solution_candidates is required"}),
                status_code=400,
                mimetype="application/json"
            )
        
        # Prepare solution ranking task
        task_data = {
            'type': 'solution_ranking',
            'solution_candidates': request_data['solution_candidates'],
            'incident_context': request_data.get('incident_context', {}),
            'ranking_criteria': request_data.get('ranking_criteria', {})
        }
        
        # Process solution ranking
        result = await resolution_agent.process_task(task_data)
        
        return func.HttpResponse(
            json.dumps(result, indent=2),
            status_code=200,
            mimetype="application/json"
        )
        
    except Exception as e:
        logger.error(f"Error in solution ranking: {str(e)}")
        
        error_response = {
            "error": "Solution ranking failed",
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }
        
        return func.HttpResponse(
            json.dumps(error_response),
            status_code=500,
            mimetype="application/json"
        )

@app.route(route="validate-resolution", methods=["POST"])
async def validate_resolution(req: func.HttpRequest) -> func.HttpResponse:
    """
    Validate and improve solution quality.
    
    Expected request body:
    {
        "proposed_solution": {
            "title": "Restart authentication service",
            "steps": [...],
            "estimated_time": "15 minutes"
        },
        "incident_data": {...},
        "validation_options": {
            "check_prerequisites": true,
            "suggest_improvements": true,
            "risk_assessment": true
        }
    }
    """
    
    logger.info("Resolution validation request received")
    
    try:
        await initialize_components()
        
        # Parse request
        request_data = req.get_json()
        if not request_data or 'proposed_solution' not in request_data:
            return func.HttpResponse(
                json.dumps({"error": "proposed_solution is required"}),
                status_code=400,
                mimetype="application/json"
            )
        
        # Prepare validation task
        task_data = {
            'type': 'resolution_validation',
            'proposed_solution': request_data['proposed_solution'],
            'incident_data': request_data.get('incident_data', {}),
            'validation_options': request_data.get('validation_options', {})
        }
        
        # Process validation
        result = await resolution_agent.process_task(task_data)
        
        return func.HttpResponse(
            json.dumps(result, indent=2),
            status_code=200,
            mimetype="application/json"
        )
        
    except Exception as e:
        logger.error(f"Error in resolution validation: {str(e)}")
        
        error_response = {
            "error": "Resolution validation failed",
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }
        
        return func.HttpResponse(
            json.dumps(error_response),
            status_code=500,
            mimetype="application/json"
        )

@app.route(route="resolution-templates", methods=["GET"])
async def get_resolution_templates(req: func.HttpRequest) -> func.HttpResponse:
    """
    Get available resolution templates by category.
    Query parameters: category (optional)
    """
    
    try:
        await initialize_components()
        
        # Get category filter
        category = req.params.get('category')
        
        # Prepare template request
        task_data = {
            'type': 'get_templates',
            'category': category
        }
        
        # Get templates
        result = await resolution_agent.process_task(task_data)
        
        return func.HttpResponse(
            json.dumps(result, indent=2),
            status_code=200,
            mimetype="application/json"
        )
        
    except Exception as e:
        logger.error(f"Error getting resolution templates: {str(e)}")
        
        error_response = {
            "error": "Failed to get templates",
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }
        
        return func.HttpResponse(
            json.dumps(error_response),
            status_code=500,
            mimetype="application/json"
        )

@app.route(route="resolution-health", methods=["GET"])
async def resolution_health(req: func.HttpRequest) -> func.HttpResponse:
    """
    Health check endpoint for the resolution service.
    """
    
    try:
        await initialize_components()
        
        # Check agent status
        agent_status = resolution_agent.get_status() if resolution_agent else {"status": "not_initialized"}
        
        # Check Azure services
        azure_health = await azure_manager.get_health_status() if azure_manager else {"status": "not_initialized"}
        
        # Test resolution functionality
        resolution_test = {"status": "not_tested"}
        if resolution_agent:
            try:
                test_task = {
                    'type': 'incident_resolution',
                    'incident_data': {
                        'summary': 'Test incident for health check',
                        'category': 'test'
                    },
                    'max_solutions': 1
                }
                test_result = await resolution_agent.process_task(test_task)
                resolution_test = {"status": "healthy", "test_result": "success"}
            except Exception as e:
                resolution_test = {"status": "unhealthy", "error": str(e)}
        
        health_data = {
            "status": "healthy" if resolution_agent else "unhealthy",
            "agent_status": agent_status,
            "azure_services": azure_health,
            "resolution_functionality": resolution_test,
            "solution_cache_size": len(resolution_agent.solution_cache) if resolution_agent else 0,
            "timestamp": datetime.now().isoformat(),
            "uptime": "available" if resolution_agent else "unavailable"
        }
        
        status_code = 200 if resolution_agent else 503
        
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

@app.route(route="resolution-capabilities", methods=["GET"])
async def resolution_capabilities(req: func.HttpRequest) -> func.HttpResponse:
    """
    Get resolution agent capabilities and configuration.
    """
    
    try:
        await initialize_components()
        
        capabilities_data = {
            "resolution_types": [
                {
                    "type": "incident_resolution",
                    "description": "Generate ranked solution recommendations for incidents",
                    "supports_similar_incidents": True,
                    "supports_templates": True
                },
                {
                    "type": "solution_ranking",
                    "description": "Rank and score solution alternatives",
                    "ranking_criteria": ["effectiveness", "complexity", "time", "risk"]
                },
                {
                    "type": "resolution_validation",
                    "description": "Validate and improve solution quality",
                    "validation_checks": ["prerequisites", "risks", "completeness"]
                }
            ],
            "template_categories": [
                "authentication",
                "network",
                "database",
                "application",
                "infrastructure",
                "security"
            ],
            "confidence_metrics": [
                "solution_match_score",
                "historical_success_rate",
                "complexity_assessment",
                "resource_availability"
            ],
            "configuration": {
                "max_similar_incidents": 5,
                "confidence_threshold": 0.6,
                "rag_enabled": True,
                "templates_enabled": True
            },
            "agent_capabilities": resolution_agent.get_capabilities() if resolution_agent else [],
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

@app.route(route="resolution-statistics", methods=["GET"])
async def resolution_statistics(req: func.HttpRequest) -> func.HttpResponse:
    """
    Get resolution statistics and performance metrics.
    """
    
    try:
        await initialize_components()
        
        if not resolution_agent:
            return func.HttpResponse(
                json.dumps({"error": "Resolution agent not initialized"}),
                status_code=503,
                mimetype="application/json"
            )
        
        # Get agent metrics
        agent_status = resolution_agent.get_status()
        
        statistics_data = {
            "agent_metrics": {
                "total_resolutions": agent_status.get('metrics', {}).get('total_processed', 0),
                "successful_resolutions": agent_status.get('metrics', {}).get('successful', 0),
                "failed_resolutions": agent_status.get('metrics', {}).get('failed', 0),
                "success_rate": agent_status.get('metrics', {}).get('success_rate', 0),
                "average_resolution_time": agent_status.get('metrics', {}).get('avg_processing_time', 0),
                "last_resolution": agent_status.get('metrics', {}).get('last_activity')
            },
            "solution_metrics": {
                "cache_size": len(resolution_agent.solution_cache),
                "average_confidence": getattr(resolution_agent, 'avg_confidence', 0),
                "template_usage": getattr(resolution_agent, 'template_usage', {})
            },
            "performance_metrics": {
                "rag_usage_rate": getattr(resolution_agent, 'rag_usage_rate', 0),
                "similar_incidents_found_rate": getattr(resolution_agent, 'similar_found_rate', 0)
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
    global resolution_agent, message_bus, registry
    
    try:
        if resolution_agent:
            await resolution_agent.stop()
        if registry:
            await registry.stop()
        if message_bus:
            await message_bus.stop()
            
        logger.info("Resolution service resources cleaned up successfully")
        
    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}")