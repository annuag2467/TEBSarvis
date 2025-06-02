from backend.agents.reactive.search_agent import SearchAgent

search_agent = None

async def initialize_search_agent():
    """Initialize search agent"""
    global search_agent, registry, message_bus
    
    if not search_agent:
        if not registry:
            await initialize_agents()
        
        search_agent = SearchAgent()
        await registry.register_agent(search_agent)
        await search_agent.start()

@app.route(route="search-similar-incidents", methods=["POST"])
async def search_similar_incidents(req: func.HttpRequest) -> func.HttpResponse:
    """
    Search for similar incidents using semantic search.
    
    Expected JSON payload:
    {
        "query": "User cannot login to LMS",
        "max_results": 5,
        "filters": {
            "category": "Learning Management System (LMS)",
            "severity": "High"
        }
    }
    """
    try:
        await initialize_search_agent()
        
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
        max_results = req_body.get('max_results', 10)
        filters = req_body.get('filters', {})
        
        # Process with search agent
        task_data = {
            'type': 'semantic_search',
            'query': query,
            'max_results': max_results,
            'filters': filters
        }
        
        result = await search_agent.process_task(task_data)
        
        # Format response
        response = {
            "query": query,
            "results": result.get('results', []),
            "total_count": result.get('total_count', 0),
            "search_type": result.get('search_type', 'semantic'),
            "timestamp": datetime.now().isoformat(),
            "processing_metadata": result.get('processing_metadata', {})
        }
        
        return func.HttpResponse(
            json.dumps(response),
            status_code=200,
            mimetype="application/json"
        )
        
    except Exception as e:
        logging.error(f"Error in search-similar-incidents: {str(e)}")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            status_code=500,
            mimetype="application/json"
        )