"""
recommend-resolution/function_app.py - Resolution recommendation endpoint
"""
import azure.functions as func
import asyncio
import json
import logging
from datetime import datetime

# Import your agents (adjust import paths as needed)
from backend.agents.reactive.resolution_agent import ResolutionAgent
from backend.agents.core.agent_registry import initialize_global_registry
from backend.agents.core.agent_communication import MessageBus

app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)

# Global variables for agent instances
resolution_agent = None
registry = None
message_bus = None

async def initialize_agents():
    """Initialize agents on startup"""
    global resolution_agent, registry, message_bus
    
    if not resolution_agent:
        registry = await initialize_global_registry()
        message_bus = MessageBus()
        await message_bus.start()
        
        resolution_agent = ResolutionAgent()
        await registry.register_agent(resolution_agent)
        await resolution_agent.start()

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
            "priority": "High"
        }
    }
    """
    try:
        await initialize_agents()
        
        # Parse request
        req_body = req.get_json()
        if not req_body or 'incident_data' not in req_body:
            return func.HttpResponse(
                json.dumps({"error": "Missing incident_data in request"}),
                status_code=400,
                mimetype="application/json"
            )
        
        incident_data = req_body['incident_data']
        
        # Process with resolution agent
        task_data = {
            'type': 'incident_resolution',
            'incident_data': incident_data
        }
        
        result = await resolution_agent.process_task(task_data)
        
        # Format response
        response = {
            "incident_id": incident_data.get('id'),
            "recommendations": result.get('solutions', []),
            "confidence": result.get('overall_confidence', 0.0),
            "similar_incidents_used": result.get('similar_incidents_used', 0),
            "timestamp": datetime.now().isoformat(),
            "agent_metadata": result.get('processing_metadata', {})
        }
        
        return func.HttpResponse(
            json.dumps(response),
            status_code=200,
            mimetype="application/json"
        )
        
    except Exception as e:
        logging.error(f"Error in recommend-resolution: {str(e)}")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            status_code=500,
            mimetype="application/json"
        )
