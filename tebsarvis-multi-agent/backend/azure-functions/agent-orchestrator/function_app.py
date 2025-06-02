from backend.agents.orchestrator.agent_coordinator import AgentCoordinator
import logging

coordinator = None

async def initialize_coordinator():
    """Initialize agent coordinator"""
    global coordinator, registry, message_bus
    
    if not coordinator:
        if not registry:
            await initialize_agents()
        
        coordinator = AgentCoordinator(registry, message_bus)
        await coordinator.start()

@app.route(route="coordinate-incident-resolution", methods=["POST"])
async def coordinate_incident_resolution(req: func.HttpRequest) -> func.HttpResponse:
    """
    Coordinate a complete incident resolution workflow across multiple agents.
    
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
        await initialize_coordinator()
        
        # Parse request
        req_body = req.get_json()
        if not req_body or 'incident_data' not in req_body:
            return func.HttpResponse(
                json.dumps({"error": "Missing incident_data in request"}),
                status_code=400,
                mimetype="application/json"
            )
        
        incident_data = req_body['incident_data']
        
        # Coordinate resolution workflow
        result = await coordinator.coordinate_incident_resolution(incident_data)
        
        return func.HttpResponse(
            json.dumps(result),
            status_code=200,
            mimetype="application/json"
        )
        
    except Exception as e:
        logging.error(f"Error in coordinate-incident-resolution: {str(e)}")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            status_code=500,
            mimetype="application/json"
        )

@app.route(route="coordinate-pattern-analysis", methods=["POST"])
async def coordinate_pattern_analysis(req: func.HttpRequest) -> func.HttpResponse:
    """
    Coordinate pattern analysis across multiple agents.
    """
    try:
        await initialize_coordinator()
        
        # Parse request
        req_body = req.get_json()
        analysis_request = req_body if req_body else {"time_range": {"days": 30}}
        
        # Coordinate pattern analysis
        result = await coordinator.coordinate_pattern_analysis(analysis_request)
        
        return func.HttpResponse(
            json.dumps(result),
            status_code=200,
            mimetype="application/json"
        )
        
    except Exception as e:
        logging.error(f"Error in coordinate-pattern-analysis: {str(e)}")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            status_code=500,
            mimetype="application/json"
        )
