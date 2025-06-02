
@app.route(route="health-check", methods=["GET"])
async def health_check(req: func.HttpRequest) -> func.HttpResponse:
    """
    Health check endpoint for the multi-agent system.
    """
    try:
        # Initialize basic components
        await initialize_agents()
        
        # Get health status from various components
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "components": {
                "agent_registry": "healthy" if registry else "not_initialized",
                "message_bus": "healthy" if message_bus else "not_initialized",
                "resolution_agent": "healthy" if resolution_agent else "not_initialized"
            },
            "version": "1.0.0"
        }
        
        # Check Azure services
        try:
            from backend.agents.shared.azure_clients import AzureClientManager
            azure_manager = AzureClientManager()
            azure_health = await azure_manager.get_health_status()
            health_status["azure_services"] = azure_health
        except Exception as e:
            health_status["azure_services"] = {"error": str(e)}
        
        return func.HttpResponse(
            json.dumps(health_status),
            status_code=200,
            mimetype="application/json"
        )
        
    except Exception as e:
        logging.error(f"Error in health-check: {str(e)}")
        return func.HttpResponse(
            json.dumps({
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }),
            status_code=500,
            mimetype="application/json"
        )