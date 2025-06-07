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

# Add the backend path to sys.path to import our agents
backend_path = os.path.join(os.path.dirname(__file__), '..', '..', 'agents')
sys.path.append(backend_path)

from ...agents.proactive.pattern_detection_agent import PatternDetectionAgent
from ...agents.core.agent_registry import get_global_registry
from ...agents.core.agent_communication import MessageBus
from ..shared.azure_clients import AzureClientManager

# Initialize global components
app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)
logger = logging.getLogger(__name__)

# Global variables for agent instances
pattern_agent = None
azure_manager = None
message_bus = None
registry = None

async def initialize_components():
    """Initialize agents and Azure components"""
    global pattern_agent, azure_manager, message_bus, registry
    
    try:
        if not pattern_agent:
            # Initialize Azure manager
            azure_manager = AzureClientManager()
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
            
            logger.info("Pattern Detection Agent components initialized successfully")
            
    except Exception as e:
        logger.error(f"Error initializing components: {str(e)}")
        raise

@app.route(route="detect-patterns", methods=["POST"])
async def detect_patterns(req: func.HttpRequest) -> func.HttpResponse:
    """
    Comprehensive pattern detection including clustering, trends, and anomalies.
    
    Expected request body:
    {
        "analysis_type": "comprehensive",  // "clustering", "trends", "anomalies", "comprehensive"
        "time_range": {
            "days": 30
        },
        "parameters": {
            "clustering_method": "mixed",
            "min_cluster_size": 3,
            "trend_types": ["volume", "category", "severity"],
            "anomaly_sensitivity": "medium"
        },
        "options": {
            "include_insights": true,
            "generate_forecasts": true,
            "detailed_analysis": false
        }
    }
    
    Returns:
    {
        "analysis_type": "comprehensive",
        "clusters": [...],
        "trends": {...},
        "anomalies": {...},
        "insights": {...},
        "analysis_metadata": {...}
    }
    """
    
    logger.info("Pattern detection request received")
    
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
        
        analysis_type = request_data.get('analysis_type', 'comprehensive')
        time_range = request_data.get('time_range', {'days': 30})
        parameters = request_data.get('parameters', {})
        options = request_data.get('options', {})
        
        # Validate analysis type
        valid_types = ['clustering', 'trends', 'anomalies', 'comprehensive', 'correlations']
        if analysis_type not in valid_types:
            return func.HttpResponse(
                json.dumps({"error": f"Invalid analysis_type. Must be one of: {valid_types}"}),
                status_code=400,
                mimetype="application/json"
            )
        
        # Prepare task based on analysis type
        if analysis_type == 'clustering':
            task_data = {
                'type': 'incident_clustering',
                'time_range': time_range,
                'clustering_method': parameters.get('clustering_method', 'mixed'),
                'min_cluster_size': parameters.get('min_cluster_size', 3)
            }
        elif analysis_type == 'trends':
            task_data = {
                'type': 'trend_analysis',
                'analysis_period': time_range,
                'trend_types': parameters.get('trend_types', ['volume', 'category', 'severity']),
                'granularity': parameters.get('granularity', 'daily')
            }
        elif analysis_type == 'anomalies':
            task_data = {
                'type': 'anomaly_detection',
                'detection_window': {'days': 7},
                'baseline_period': time_range,
                'sensitivity': parameters.get('anomaly_sensitivity', 'medium'),
                'anomaly_types': parameters.get('anomaly_types', ['volume', 'pattern', 'timing'])
            }
        elif analysis_type == 'correlations':
            task_data = {
                'type': 'correlation_analysis',
                'time_range': time_range,
                'attributes': parameters.get('attributes', ['category', 'severity', 'priority']),
                'method': parameters.get('correlation_method', 'pearson')
            }
        else:  # comprehensive
            task_data = {
                'type': 'comprehensive_analysis',
                'time_range': time_range,
                'components': parameters.get('components', ['clustering', 'trends', 'anomalies', 'correlations'])
            }
        
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
        
        logger.info(f"Pattern detection request processed successfully in {processing_time:.2f}s")
        
        return func.HttpResponse(
            json.dumps(result, indent=2),
            status_code=200,
            mimetype="application/json"
        )
        
    except Exception as e:
        logger.error(f"Error generating pattern insights: {str(e)}")
        
        error_response = {
            "error": "Pattern insights generation failed",
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }
        
        return func.HttpResponse(
            json.dumps(error_response),
            status_code=500,
            mimetype="application/json"
        )

@app.route(route="correlation-analysis", methods=["POST"])
async def analyze_correlations(req: func.HttpRequest) -> func.HttpResponse:
    """
    Analyze correlations between different incident attributes.
    
    Expected request body:
    {
        "attributes": ["category", "severity", "priority", "resolution_time"],
        "time_range": {"days": 60},
        "correlation_method": "pearson"  // "pearson", "spearman", "kendall"
    }
    """
    
    logger.info("Correlation analysis request received")
    
    try:
        await initialize_components()
        
        # Parse request
        request_data = req.get_json() or {}
        
        # Prepare correlation analysis task
        task_data = {
            'type': 'correlation_analysis',
            'attributes': request_data.get('attributes', ['category', 'severity', 'priority']),
            'time_range': request_data.get('time_range', {'days': 60}),
            'method': request_data.get('correlation_method', 'pearson')
        }
        
        # Process correlation analysis
        result = await pattern_agent.process_task(task_data)
        
        return func.HttpResponse(
            json.dumps(result, indent=2),
            status_code=200,
            mimetype="application/json"
        )
        
    except Exception as e:
        logger.error(f"Error in correlation analysis: {str(e)}")
        
        error_response = {
            "error": "Correlation analysis failed",
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }
        
        return func.HttpResponse(
            json.dumps(error_response),
            status_code=500,
            mimetype="application/json"
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
            "analysis_cache_size": len(pattern_agent.analysis_cache) if pattern_agent else 0,
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

@app.route(route="pattern-capabilities", methods=["GET"])
async def pattern_capabilities(req: func.HttpRequest) -> func.HttpResponse:
    """
    Get pattern detection agent capabilities and configuration.
    """
    
    try:
        await initialize_components()
        
        capabilities_data = {
            "analysis_types": [
                {
                    "type": "incident_clustering",
                    "description": "Cluster incidents using ML algorithms",
                    "methods": ["semantic", "categorical", "mixed"]
                },
                {
                    "type": "trend_analysis",
                    "description": "Analyze trends in incident patterns over time",
                    "trend_types": ["volume", "category", "severity", "resolution_time"]
                },
                {
                    "type": "anomaly_detection",
                    "description": "Detect anomalous patterns in incident data",
                    "anomaly_types": ["volume", "pattern", "timing", "category"]
                },
                {
                    "type": "correlation_analysis",
                    "description": "Find correlations between incident attributes",
                    "methods": ["pearson", "spearman", "kendall"]
                },
                {
                    "type": "comprehensive_analysis",
                    "description": "Complete pattern analysis combining multiple techniques"
                }
            ],
            "clustering_methods": [
                "semantic",
                "categorical", 
                "mixed"
            ],
            "granularity_options": [
                "hourly",
                "daily",
                "weekly"
            ],
            "sensitivity_levels": [
                "low",
                "medium",
                "high"
            ],
            "configuration": {
                "min_cluster_size": 3,
                "trend_window_days": 30,
                "cache_duration_seconds": 1800,
                "default_analysis_period": {"days": 30}
            },
            "agent_capabilities": pattern_agent.get_capabilities() if pattern_agent else [],
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

@app.route(route="pattern-statistics", methods=["GET"])
async def pattern_statistics(req: func.HttpRequest) -> func.HttpResponse:
    """
    Get pattern detection statistics and performance metrics.
    """
    
    try:
        await initialize_components()
        
        if not pattern_agent:
            return func.HttpResponse(
                json.dumps({"error": "Pattern agent not initialized"}),
                status_code=503,
                mimetype="application/json"
            )
        
        # Get agent metrics
        agent_status = pattern_agent.get_status()
        
        statistics_data = {
            "agent_metrics": {
                "total_tasks_processed": agent_status.get('metrics', {}).get('total_processed', 0),
                "successful_tasks": agent_status.get('metrics', {}).get('successful', 0),
                "failed_tasks": agent_status.get('metrics', {}).get('failed', 0),
                "success_rate": agent_status.get('metrics', {}).get('success_rate', 0),
                "average_processing_time": agent_status.get('metrics', {}).get('avg_processing_time', 0),
                "last_activity": agent_status.get('metrics', {}).get('last_activity')
            },
            "cache_statistics": {
                "cache_size": len(pattern_agent.analysis_cache),
                "cache_duration_seconds": pattern_agent.cache_duration
            },
            "configuration": {
                "min_cluster_size": pattern_agent.min_cluster_size,
                "trend_window_days": pattern_agent.trend_window_days
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
    global pattern_agent, message_bus, registry
    
    try:
        if pattern_agent:
            await pattern_agent.stop()
        if registry:
            await registry.stop()
        if message_bus:
            await message_bus.stop()
            
        logger.info("Pattern detection service resources cleaned up successfully")
        
    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}"),
        raise
        
    except Exception as e:
        logger.error(f"Error processing pattern detection request: {str(e)}")
        
        error_response = {
            "error": "Internal server error",
            "message": str(e),
            "timestamp": datetime.now().isoformat(),
            "function_name": "detect-patterns"
        }
        
        return func.HttpResponse(
            json.dumps(error_response),
            status_code=500,
            mimetype="application/json"
        )

@app.route(route="cluster-incidents", methods=["POST"])
async def cluster_incidents(req: func.HttpRequest) -> func.HttpResponse:
    """
    Dedicated endpoint for incident clustering analysis.
    
    Expected request body:
    {
        "time_range": {"days": 30},
        "clustering_method": "mixed",  // "semantic", "categorical", "mixed"
        "min_cluster_size": 3,
        "include_insights": true
    }
    """
    
    logger.info("Incident clustering request received")
    
    try:
        await initialize_components()
        
        # Parse request
        request_data = req.get_json() or {}
        
        # Prepare clustering task
        task_data = {
            'type': 'incident_clustering',
            'time_range': request_data.get('time_range', {'days': 30}),
            'clustering_method': request_data.get('clustering_method', 'mixed'),
            'min_cluster_size': request_data.get('min_cluster_size', 3)
        }
        
        # Process clustering
        result = await pattern_agent.process_task(task_data)
        
        return func.HttpResponse(
            json.dumps(result, indent=2),
            status_code=200,
            mimetype="application/json"
        )
        
    except Exception as e:
        logger.error(f"Error in incident clustering: {str(e)}")
        
        error_response = {
            "error": "Clustering analysis failed",
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }
        
        return func.HttpResponse(
            json.dumps(error_response),
            status_code=500,
            mimetype="application/json"
        )

@app.route(route="trend-analysis", methods=["POST"])
async def analyze_trends(req: func.HttpRequest) -> func.HttpResponse:
    """
    Dedicated endpoint for trend analysis.
    
    Expected request body:
    {
        "analysis_period": {"days": 90},
        "trend_types": ["volume", "category", "severity", "resolution_time"],
        "granularity": "daily",  // "hourly", "daily", "weekly"
        "generate_forecasts": true
    }
    """
    
    logger.info("Trend analysis request received")
    
    try:
        await initialize_components()
        
        # Parse request
        request_data = req.get_json() or {}
        
        # Prepare trend analysis task
        task_data = {
            'type': 'trend_analysis',
            'analysis_period': request_data.get('analysis_period', {'days': 90}),
            'trend_types': request_data.get('trend_types', ['volume', 'category', 'severity']),
            'granularity': request_data.get('granularity', 'daily')
        }
        
        # Process trend analysis
        result = await pattern_agent.process_task(task_data)
        
        return func.HttpResponse(
            json.dumps(result, indent=2),
            status_code=200,
            mimetype="application/json"
        )
        
    except Exception as e:
        logger.error(f"Error in trend analysis: {str(e)}")
        
        error_response = {
            "error": "Trend analysis failed",
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }
        
        return func.HttpResponse(
            json.dumps(error_response),
            status_code=500,
            mimetype="application/json"
        )

@app.route(route="anomaly-detection", methods=["POST"])
async def detect_anomalies(req: func.HttpRequest) -> func.HttpResponse:
    """
    Dedicated endpoint for anomaly detection.
    
    Expected request body:
    {
        "detection_window": {"days": 7},
        "baseline_period": {"days": 30},
        "sensitivity": "medium",  // "low", "medium", "high"
        "anomaly_types": ["volume", "pattern", "timing", "category"]
    }
    """
    
    logger.info("Anomaly detection request received")
    
    try:
        await initialize_components()
        
        # Parse request
        request_data = req.get_json() or {}
        
        # Prepare anomaly detection task
        task_data = {
            'type': 'anomaly_detection',
            'detection_window': request_data.get('detection_window', {'days': 7}),
            'baseline_period': request_data.get('baseline_period', {'days': 30}),
            'sensitivity': request_data.get('sensitivity', 'medium'),
            'anomaly_types': request_data.get('anomaly_types', ['volume', 'pattern', 'timing'])
        }
        
        # Process anomaly detection
        result = await pattern_agent.process_task(task_data)
        
        return func.HttpResponse(
            json.dumps(result, indent=2),
            status_code=200,
            mimetype="application/json"
        )
        
    except Exception as e:
        logger.error(f"Error in anomaly detection: {str(e)}")
        
        error_response = {
            "error": "Anomaly detection failed",
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }
        
        return func.HttpResponse(
            json.dumps(error_response),
            status_code=500,
            mimetype="application/json"
        )

@app.route(route="pattern-insights", methods=["POST"])
async def generate_pattern_insights(req: func.HttpRequest) -> func.HttpResponse:
    """
    Generate actionable insights from pattern data.
    
    Expected request body:
    {
        "pattern_data": {
            "clusters": [...],
            "trends": {...},
            "anomalies": {...}
        },
        "insight_types": ["operational", "strategic", "preventive"]
    }
    """
    
    logger.info("Pattern insights request received")
    
    try:
        await initialize_components()
        
        # Parse request
        request_data = req.get_json()
        if not request_data or 'pattern_data' not in request_data:
            return func.HttpResponse(
                json.dumps({"error": "pattern_data is required"}),
                status_code=400,
                mimetype="application/json"
            )
        
        # Prepare insight generation task
        task_data = {
            'type': 'pattern_insights',
            'pattern_data': request_data['pattern_data'],
            'insight_types': request_data.get('insight_types', ['operational', 'strategic', 'preventive'])
        }
        
        # Process insight generation
        result = await pattern_agent.process_task(task_data)
        
        return func.HttpResponse(
            json.dumps(result, indent=2),
            status_code=200,
            mimetype="application/json"
        )
    except Exception as e:
        logger.error(f"Error generating pattern insights: {str(e)}")
        
        error_response = {
            "error": "Pattern insights generation failed",
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }
        
        return func.HttpResponse(
            json.dumps(error_response),
            status_code=500,
            mimetype="application/json"
        )
        