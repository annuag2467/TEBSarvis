"""
Azure Function App for System Health Check
Endpoint: /health-check
Provides comprehensive health monitoring for the multi-agent system.
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

from core.agent_registry import get_global_registry, initialize_global_registry
from core.agent_communication import MessageBus
from shared.azure_clients import AzureClientManager
from reactive.resolution_agent import ResolutionAgent
from reactive.search_agent import SearchAgent
from reactive.conversation_agent import ConversationAgent
from reactive.context_agent import ContextAgent
from proactive.pattern_detection_agent import PatternDetectionAgent
from proactive.alerting_agent import AlertingAgent

# Initialize global components
app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)
logger = logging.getLogger(__name__)

# Global variables for components
azure_manager = None
message_bus = None
registry = None
agents = {}

async def initialize_agents():
    """Initialize all agents and components"""
    global azure_manager, message_bus, registry, agents
    
    try:
        if not azure_manager:
            # Initialize Azure manager
            azure_manager = AzureClientManager()
            await azure_manager.initialize()
            
            # Initialize message bus and registry
            message_bus = MessageBus()
            await message_bus.start()
            
            registry = await initialize_global_registry()
            
            # Initialize agents
            agents['resolution'] = ResolutionAgent()
            agents['search'] = SearchAgent()
            agents['conversation'] = ConversationAgent()
            agents['context'] = ContextAgent()
            agents['pattern_detection'] = PatternDetectionAgent()
            agents['alerting'] = AlertingAgent()
            
            # Start agents
            for agent_name, agent in agents.items():
                await agent.start()
                await registry.register_agent(agent)
                logger.info(f"{agent_name} agent initialized successfully")
            
            logger.info("All system components initialized successfully")
            
    except Exception as e:
        logger.error(f"Error initializing system components: {str(e)}")
        raise

@app.route(route="health-check", methods=["GET"])
async def health_check(req: func.HttpRequest) -> func.HttpResponse:
    """
    Comprehensive health check endpoint for the multi-agent system.
    
    Returns:
    {
        "status": "healthy|degraded|unhealthy",
        "timestamp": "2024-01-01T00:00:00Z",
        "components": {
            "azure_services": {...},
            "agent_registry": {...},
            "message_bus": {...},
            "agents": {...}
        },
        "overall_score": 95,
        "issues": [...],
        "recommendations": [...]
    }
    """
    
    logger.info("Health check request received")
    
    try:
        # Initialize components if needed
        await initialize_agents()
        
        health_results = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "components": {},
            "overall_score": 100,
            "issues": [],
            "recommendations": [],
            "version": "1.0.0"
        }
        
        # Check Azure services
        try:
            azure_health = await check_azure_services()
            health_results["components"]["azure_services"] = azure_health
            
            if azure_health["status"] != "healthy":
                health_results["issues"].append("Azure services degraded")
                health_results["overall_score"] -= 20
                
        except Exception as e:
            health_results["components"]["azure_services"] = {
                "status": "error",
                "error": str(e)
            }
            health_results["issues"].append(f"Azure services error: {str(e)}")
            health_results["overall_score"] -= 30
        
        # Check agent registry
        try:
            registry_health = await check_agent_registry()
            health_results["components"]["agent_registry"] = registry_health
            
            if registry_health["status"] != "healthy":
                health_results["issues"].append("Agent registry issues")
                health_results["overall_score"] -= 15
                
        except Exception as e:
            health_results["components"]["agent_registry"] = {
                "status": "error",
                "error": str(e)
            }
            health_results["issues"].append(f"Agent registry error: {str(e)}")
            health_results["overall_score"] -= 20
        
        # Check message bus
        try:
            message_bus_health = await check_message_bus()
            health_results["components"]["message_bus"] = message_bus_health
            
            if message_bus_health["status"] != "healthy":
                health_results["issues"].append("Message bus issues")
                health_results["overall_score"] -= 15
                
        except Exception as e:
            health_results["components"]["message_bus"] = {
                "status": "error",
                "error": str(e)
            }
            health_results["issues"].append(f"Message bus error: {str(e)}")
            health_results["overall_score"] -= 20
        
        # Check individual agents
        try:
            agents_health = await check_all_agents()
            health_results["components"]["agents"] = agents_health
            
            unhealthy_agents = [
                agent_id for agent_id, status in agents_health.items()
                if status.get("status") != "healthy"
            ]
            
            if unhealthy_agents:
                health_results["issues"].append(f"Unhealthy agents: {', '.join(unhealthy_agents)}")
                health_results["overall_score"] -= len(unhealthy_agents) * 10
                
        except Exception as e:
            health_results["components"]["agents"] = {
                "status": "error",
                "error": str(e)
            }
            health_results["issues"].append(f"Agent health check error: {str(e)}")
            health_results["overall_score"] -= 25
        
        # Determine overall status
        if health_results["overall_score"] >= 90:
            health_results["status"] = "healthy"
        elif health_results["overall_score"] >= 70:
            health_results["status"] = "degraded"
            health_results["recommendations"].append("Some components need attention")
        else:
            health_results["status"] = "unhealthy"
            health_results["recommendations"].append("Immediate attention required")
        
        # Add system-level recommendations
        if health_results["overall_score"] < 100:
            health_results["recommendations"].append("Check system logs for detailed error information")
        
        # Set appropriate HTTP status code
        status_code = 200 if health_results["status"] == "healthy" else 503
        
        return func.HttpResponse(
            json.dumps(health_results, indent=2),
            status_code=status_code,
            mimetype="application/json"
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        
        error_response = {
            "status": "error",
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
            "overall_score": 0,
            "message": "Health check system failure"
        }
        
        return func.HttpResponse(
            json.dumps(error_response),
            status_code=503,
            mimetype="application/json"
        )

async def check_azure_services() -> Dict[str, Any]:
    """Check health of Azure services"""
    try:
        azure_health = await azure_manager.get_health_status()
        
        # Count healthy services
        services = ['openai', 'cosmos', 'search']
        healthy_count = sum(1 for service in services if azure_health.get(service) == 'healthy')
        
        overall_status = "healthy" if healthy_count == len(services) else "degraded"
        
        return {
            "status": overall_status,
            "services": azure_health,
            "healthy_services": healthy_count,
            "total_services": len(services),
            "last_checked": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "last_checked": datetime.now().isoformat()
        }

async def check_agent_registry() -> Dict[str, Any]:
    """Check health of agent registry"""
    try:
        if not registry:
            return {"status": "not_initialized"}
        
        stats = registry.get_registry_statistics()
        
        status = "healthy" if stats.get('active_agents', 0) > 0 else "degraded"
        
        return {
            "status": status,
            "statistics": stats,
            "last_checked": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "last_checked": datetime.now().isoformat()
        }

async def check_message_bus() -> Dict[str, Any]:
    """Check health of message bus"""
    try:
        if not message_bus:
            return {"status": "not_initialized"}
        
        stats = message_bus.get_statistics()
        
        # Simple health check based on message processing
        status = "healthy" if stats.get('uptime_seconds', 0) > 0 else "degraded"
        
        return {
            "status": status,
            "statistics": stats,
            "last_checked": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "last_checked": datetime.now().isoformat()
        }

async def check_all_agents() -> Dict[str, Any]:
    """Check health of all agents"""
    agent_health = {}
    
    for agent_name, agent in agents.items():
        try:
            if agent:
                status = agent.get_status()
                
                # Determine health based on status
                agent_status = status.get('status', 'unknown')
                if agent_status == 'error':
                    health_status = "unhealthy"
                elif agent_status in ['idle', 'processing']:
                    health_status = "healthy"
                else:
                    health_status = "degraded"
                
                agent_health[agent_name] = {
                    "status": health_status,
                    "agent_status": agent_status,
                    "active_tasks": status.get('active_tasks', 0),
                    "success_rate": status.get('metrics', {}).get('success_rate', 0),
                    "last_activity": status.get('metrics', {}).get('last_activity'),
                    "last_checked": datetime.now().isoformat()
                }
            else:
                agent_health[agent_name] = {
                    "status": "not_initialized",
                    "last_checked": datetime.now().isoformat()
                }
        except Exception as e:
                
                agent_health[agent_name] = {
                    "status": "error",
                    "error": str(e),
                    "last_checked": datetime.now().isoformat()
                }
        
        return agent_health

    async def check_individual_agent(agent_name: str, agent) -> Dict[str, Any]:
        """
        Perform detailed health check on individual agent
        """
        try:
            # Get basic status
            status = agent.get_status()
            
            # Perform ping test
            ping_start = datetime.now()
            ping_response = await agent.ping()
            ping_duration = (datetime.now() - ping_start).total_seconds()
            
            # Check recent performance metrics
            metrics = status.get('metrics', {})
            
            # Determine overall agent health
            health_indicators = {
                'responsive': ping_response is not None,
                'low_latency': ping_duration < 1.0,
                'good_success_rate': metrics.get('success_rate', 0) > 0.8,
                'recent_activity': metrics.get('last_activity') is not None
            }
            
            healthy_indicators = sum(health_indicators.values())
            health_percentage = (healthy_indicators / len(health_indicators)) * 100
            
            if health_percentage >= 75:
                overall_status = "healthy"
            elif health_percentage >= 50:
                overall_status = "degraded"
            else:
                overall_status = "unhealthy"
            
            return {
                "status": overall_status,
                "health_percentage": health_percentage,
                "ping_duration_ms": ping_duration * 1000,
                "agent_status": status.get('status', 'unknown'),
                "active_tasks": status.get('active_tasks', 0),
                "metrics": metrics,
                "health_indicators": health_indicators,
                "last_checked": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "last_checked": datetime.now().isoformat()
            }

    @app.route(route="health-check/detailed", methods=["GET"])
    async def detailed_health_check(req: func.HttpRequest) -> func.HttpResponse:
        """
        Detailed health check with performance metrics and diagnostics
        """
        logger.info("Detailed health check request received")
        
        try:
            await initialize_agents()
            
            detailed_results = {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "system_info": {
                    "python_version": sys.version,
                    "platform": sys.platform,
                    "function_app_version": "1.0.0"
                },
                "performance_metrics": {},
                "detailed_components": {},
                "connectivity_tests": {},
                "resource_usage": {},
                "recommendations": []
            }
            
            # Performance timing tests
            start_time = datetime.now()
            
            # Test Azure OpenAI connectivity
            try:
                openai_start = datetime.now()
                openai_test = await azure_manager.openai_client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": "Health check ping"}],
                    max_tokens=5
                )
                openai_duration = (datetime.now() - openai_start).total_seconds()
                
                detailed_results["connectivity_tests"]["openai"] = {
                    "status": "healthy",
                    "response_time_ms": openai_duration * 1000,
                    "last_tested": datetime.now().isoformat()
                }
            except Exception as e:
                detailed_results["connectivity_tests"]["openai"] = {
                    "status": "error",
                    "error": str(e),
                    "last_tested": datetime.now().isoformat()
                }
            
            # Test Cosmos DB connectivity
            try:
                cosmos_start = datetime.now()
                # Simple query to test connectivity
                cosmos_response = await azure_manager.cosmos_client.get_database_client("health").read()
                cosmos_duration = (datetime.now() - cosmos_start).total_seconds()
                
                detailed_results["connectivity_tests"]["cosmos_db"] = {
                    "status": "healthy",
                    "response_time_ms": cosmos_duration * 1000,
                    "last_tested": datetime.now().isoformat()
                }
            except Exception as e:
                detailed_results["connectivity_tests"]["cosmos_db"] = {
                    "status": "error",
                    "error": str(e),
                    "last_tested": datetime.now().isoformat()
                }
            
            # Test Cognitive Search connectivity
            try:
                search_start = datetime.now()
                search_response = await azure_manager.search_client.get_service_statistics()
                search_duration = (datetime.now() - search_start).total_seconds()
                
                detailed_results["connectivity_tests"]["cognitive_search"] = {
                    "status": "healthy",
                    "response_time_ms": search_duration * 1000,
                    "service_stats": search_response,
                    "last_tested": datetime.now().isoformat()
                }
            except Exception as e:
                detailed_results["connectivity_tests"]["cognitive_search"] = {
                    "status": "error",
                    "error": str(e),
                    "last_tested": datetime.now().isoformat()
                }
            
            # Detailed agent checks
            for agent_name, agent in agents.items():
                detailed_results["detailed_components"][agent_name] = await check_individual_agent(agent_name, agent)
            
            # Calculate overall performance metrics
            total_duration = (datetime.now() - start_time).total_seconds()
            detailed_results["performance_metrics"] = {
                "total_health_check_duration_ms": total_duration * 1000,
                "average_response_time_ms": sum([
                    test.get("response_time_ms", 0) 
                    for test in detailed_results["connectivity_tests"].values()
                    if "response_time_ms" in test
                ]) / len(detailed_results["connectivity_tests"]),
                "timestamp": datetime.now().isoformat()
            }
            
            # Resource usage (basic)
            try:
                import psutil
                detailed_results["resource_usage"] = {
                    "cpu_percent": psutil.cpu_percent(),
                    "memory_percent": psutil.virtual_memory().percent,
                    "timestamp": datetime.now().isoformat()
                }
            except ImportError:
                detailed_results["resource_usage"] = {
                    "note": "psutil not available for resource monitoring"
                }
            
            # Generate recommendations based on findings
            connectivity_issues = [
                service for service, status in detailed_results["connectivity_tests"].items()
                if status.get("status") != "healthy"
            ]
            
            if connectivity_issues:
                detailed_results["recommendations"].append(
                    f"Check connectivity to: {', '.join(connectivity_issues)}"
                )
            
            slow_services = [
                service for service, status in detailed_results["connectivity_tests"].items()
                if status.get("response_time_ms", 0) > 1000
            ]
            
            if slow_services:
                detailed_results["recommendations"].append(
                    f"High latency detected for: {', '.join(slow_services)}"
                )
            
            unhealthy_agents = [
                agent for agent, status in detailed_results["detailed_components"].items()
                if status.get("status") != "healthy"
            ]
            
            if unhealthy_agents:
                detailed_results["recommendations"].append(
                    f"Agent health issues: {', '.join(unhealthy_agents)}"
                )
            
            # Determine overall status
            if not connectivity_issues and not unhealthy_agents:
                detailed_results["status"] = "healthy"
            elif len(connectivity_issues) + len(unhealthy_agents) <= 2:
                detailed_results["status"] = "degraded"
            else:
                detailed_results["status"] = "unhealthy"
            
            status_code = 200 if detailed_results["status"] == "healthy" else 503
            
            return func.HttpResponse(
                json.dumps(detailed_results, indent=2),
                status_code=status_code,
                mimetype="application/json"
            )
            
        except Exception as e:
            logger.error(f"Detailed health check failed: {str(e)}")
            
            error_response = {
                "status": "error",
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "message": "Detailed health check system failure"
            }
            
            return func.HttpResponse(
                json.dumps(error_response),
                status_code=503,
                mimetype="application/json"
            )

    @app.route(route="health-check/agent/{agent_name}", methods=["GET"])
    async def agent_specific_health_check(req: func.HttpRequest) -> func.HttpResponse:
        """
        Health check for specific agent
        """
        agent_name = req.route_params.get('agent_name')
        logger.info(f"Agent-specific health check requested for: {agent_name}")
        
        try:
            await initialize_agents()
            
            if agent_name not in agents:
                return func.HttpResponse(
                    json.dumps({
                        "error": f"Agent '{agent_name}' not found",
                        "available_agents": list(agents.keys())
                    }),
                    status_code=404,
                    mimetype="application/json"
                )
            
            agent = agents[agent_name]
            agent_health = await check_individual_agent(agent_name, agent)
            
            # Add agent-specific diagnostics
            agent_health["diagnostics"] = {
                "agent_type": type(agent).__name__,
                "capabilities": getattr(agent, 'capabilities', []),
                "configuration": getattr(agent, 'config', {}),
                "recent_errors": getattr(agent, 'recent_errors', [])
            }
            
            status_code = 200 if agent_health["status"] == "healthy" else 503
            
            return func.HttpResponse(
                json.dumps(agent_health, indent=2),
                status_code=status_code,
                mimetype="application/json"
            )
            
        except Exception as e:
            logger.error(f"Agent health check failed for {agent_name}: {str(e)}")
            
            error_response = {
                "status": "error",
                "agent_name": agent_name,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            
            return func.HttpResponse(
                json.dumps(error_response),
                status_code=503,
                mimetype="application/json"
            )

    @app.route(route="health-check/metrics", methods=["GET"])
    async def health_metrics(req: func.HttpRequest) -> func.HttpResponse:
        """
        Return health metrics in Prometheus format for monitoring integration
        """
        try:
            await initialize_agents()
            
            # Basic health check
            basic_health = await check_azure_services()
            agents_health = await check_all_agents()
            
            metrics = []
            
            # Azure services metrics
            for service, status in basic_health.get("services", {}).items():
                metric_value = 1 if status == "healthy" else 0
                metrics.append(f'tebsarvis_azure_service_health{{service="{service}"}} {metric_value}')
            
            # Agent health metrics
            for agent_name, health in agents_health.items():
                metric_value = 1 if health.get("status") == "healthy" else 0
                metrics.append(f'tebsarvis_agent_health{{agent="{agent_name}"}} {metric_value}')
                
                # Success rate metrics
                success_rate = health.get("success_rate", 0)
                metrics.append(f'tebsarvis_agent_success_rate{{agent="{agent_name}"}} {success_rate}')
                
                # Active tasks
                active_tasks = health.get("active_tasks", 0)
                metrics.append(f'tebsarvis_agent_active_tasks{{agent="{agent_name}"}} {active_tasks}')
            
            # System-wide metrics
            healthy_agents = sum(1 for health in agents_health.values() if health.get("status") == "healthy")
            total_agents = len(agents_health)
            
            metrics.append(f'tebsarvis_healthy_agents_total {healthy_agents}')
            metrics.append(f'tebsarvis_total_agents {total_agents}')
            metrics.append(f'tebsarvis_system_health_score {(healthy_agents/total_agents)*100 if total_agents > 0 else 0}')
            
            metrics_text = '\n'.join(metrics) + '\n'
            
            return func.HttpResponse(
                metrics_text,
                status_code=200,
                mimetype="text/plain"
            )
            
        except Exception as e:
            logger.error(f"Health metrics generation failed: {str(e)}")
            return func.HttpResponse(
                f"# Error generating metrics: {str(e)}\n",
                status_code=503,
                mimetype="text/plain"
            )