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

# FIXED IMPORTS
from ...agents.core.agent_registry import get_global_registry, initialize_global_registry
from ...agents.core.agent_communication import MessageBus
from ...agents.reactive.resolution_agent import ResolutionAgent
from ...agents.reactive.search_agent import SearchAgent
from ...agents.reactive.conversation_agent import ConversationAgent
from ...agents.reactive.context_agent import ContextAgent
from ...agents.proactive.pattern_detection_agent import PatternDetectionAgent
from ...agents.proactive.alerting_agent import AlertingAgent

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

# Global components
connection_pool = None
executor = None
cache = {}
cache_ttl = {}
rate_limit_cache = defaultdict(list)
RATE_LIMIT = int(os.getenv('RATE_LIMIT', '100'))  # requests per minute
CACHE_DEFAULT_TTL = 60  # 1 minute for health checks
MAX_WORKERS = int(os.getenv('MAX_WORKERS', '4'))

# Initialize global components
app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)
logger = logging.getLogger(__name__)

# Global variables for components
azure_manager = None
message_bus = None
registry = None
agents = {}

async def initialize_agents():
    """Initialize all agents and components with connection pooling and monitoring"""
    global azure_manager, message_bus, registry, agents, connection_pool, executor
    
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

        if not azure_manager:
            # Initialize Azure manager with connection pool
            azure_manager = AzureClientManager(connector=connection_pool)
            await azure_manager.initialize()
            
            # Initialize message bus and registry
            message_bus = MessageBus()
            await message_bus.start()
            
            registry = await initialize_global_registry()
            
            # Initialize agents with parallel startup
            agent_classes = {
                'resolution': ResolutionAgent,
                'search': SearchAgent,
                'conversation': ConversationAgent,
                'context': ContextAgent,
                'pattern_detection': PatternDetectionAgent,
                'alerting': AlertingAgent
            }
            
            # Create and start agents in parallel
            async def init_agent(name, agent_class):
                try:
                    agent = agent_class()
                    await agent.start()
                    await registry.register_agent(agent)
                    logger.info(f"{name} agent initialized successfully")
                    return name, agent
                except Exception as e:
                    logger.error(f"Error initializing {name} agent: {str(e)}", exc_info=True)
                    raise
            
            init_tasks = [
                init_agent(name, cls) 
                for name, cls in agent_classes.items()
            ]
            agent_results = await asyncio.gather(*init_tasks, return_exceptions=True)
            
            # Process results and handle any failures
            failed_agents = []
            for result in agent_results:
                if isinstance(result, Exception):
                    failed_agents.append(str(result))
                else:
                    name, agent = result
                    agents[name] = agent
            
            if failed_agents:
                raise Exception(f"Failed to initialize agents: {', '.join(failed_agents)}")
            
            # Setup monitoring
            setup_monitoring(app_name="health-check")
            
            logger.info(f"All system components initialized successfully with {len(agents)} agents")
            
    except Exception as e:
        logger.error(f"Error initializing system components: {str(e)}", exc_info=True)
        # Cleanup on failure
        if connection_pool:
            await connection_pool.close()
        if executor:
            executor.shutdown(wait=False)
        for agent in agents.values():
            try:
                await agent.stop()
            except:
                pass
        raise

@app.route(route="health-check", methods=["GET"])
async def health_check(req: func.HttpRequest) -> func.HttpResponse:
    """
    Comprehensive health check endpoint for the multi-agent system with rate limiting and parallel checks.
    
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
        "performance": {...},
        "issues": [...],
        "recommendations": [...]
    }
    """
    request_id = req.headers.get('x-request-id', f"req_{int(time.time()*1000)}")
    client_ip = get_client_ip(req)
    start_time = time.time()
    
    try:
        # Setup request logging context
        logger.info(f"Processing health check request {request_id} from {client_ip}")
        
        # Check rate limit with higher allowance for health checks
        if not check_rate_limit(client_ip, RATE_LIMIT * 2):  # Double rate limit for health checks
            return create_error_response("Rate limit exceeded", 429)
            
        # Initialize components with retries
        retry_count = 0
        while retry_count < 3:
            try:
                await initialize_agents()
                break
            except Exception as e:
                retry_count += 1
                if retry_count == 3:
                    raise
                await asyncio.sleep(1)

        # Get system resource usage
        try:
            import psutil
            process = psutil.Process(os.getpid())
            resource_usage = {
                "cpu_percent": psutil.cpu_percent(interval=0.1),
                "memory_percent": psutil.virtual_memory().percent,
                "process_memory_mb": process.memory_info().rss / 1024 / 1024,
                "open_files": len(process.open_files()),
                "threads": process.num_threads(),
                "connections": len(process.connections())
            }
        except ImportError:
            resource_usage = {"note": "psutil not available"}
        except Exception as e:
            resource_usage = {"error": str(e)}

        # Run all health checks in parallel
        health_tasks = {
            "azure": check_azure_services(),
            "registry": check_agent_registry(),
            "message_bus": check_message_bus(),
            "agents": check_all_agents(),
            "workspace": check_workspace_health(),
            "network": check_network_health()
        }
        
        results = await asyncio.gather(*health_tasks.values(), return_exceptions=True)
        health_results = dict(zip(health_tasks.keys(), results))
        
        # Process results and calculate health score
        overall_score = 100
        issues = []
        recommendations = []
        components = {}
        
        for component, result in health_results.items():
            if isinstance(result, Exception):
                components[component] = {
                    "status": "error",
                    "error": str(result),
                    "timestamp": datetime.now().isoformat()
                }
                issues.append(f"{component} check failed: {str(result)}")
                overall_score -= 20
                recommendations.append(f"Investigate {component} failure and restart if necessary")
            else:
                components[component] = result
                if result.get("status") != "healthy":
                    issues.append(f"{component} is {result.get('status', 'unhealthy')}")
                    overall_score -= 10
                    if "recommendations" in result:
                        recommendations.extend(result["recommendations"])

        # Add performance metrics
        performance = {
            "processing_time": time.time() - start_time,
            "resource_usage": resource_usage,
            "connection_pool": {
                "active_connections": connection_pool.size if connection_pool else 0,
                "limit": connection_pool.limit if connection_pool else 0
            },
            "thread_pool": {
                "active_workers": len(agents),
                "max_workers": MAX_WORKERS
            }
        }

        # Determine overall status
        status = "healthy"
        if overall_score < 60:
            status = "unhealthy"
        elif overall_score < 80:
            status = "degraded"

        response = {
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "request_id": request_id,
            "components": components,
            "performance": performance,
            "overall_score": max(0, overall_score),
            "issues": issues,
            "recommendations": recommendations,
            "version": "1.0.0"
        }

        # Cache successful health check results briefly
        if status == "healthy":
            cache_key = "health_check"
            cache[cache_key] = response
            cache_ttl[cache_key] = time.time()

        logger.info(f"Health check {request_id} completed in {time.time() - start_time:.2f}s with status: {status}")

        return func.HttpResponse(
            json.dumps(response, indent=2),
            status_code=200 if status != "unhealthy" else 503,
            mimetype="application/json"
        )

    except Exception as e:
        error_time = time.time() - start_time
        logger.error(f"Health check {request_id} failed after {error_time:.2f}s: {str(e)}", exc_info=True)
        return create_error_response(
            f"Health check failed: {str(e)}",
            500,
            {
                "request_id": request_id,
                "processing_time": error_time,
                "components_initialized": bool(agents)
            }
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
        
        # Add resource monitoring
        try:
            import psutil
            resource_usage = {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent
            }
        except ImportError:
            resource_usage = {"note": "psutil not available"}
        except Exception as e:
            resource_usage = {"error": str(e)}
        
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
            
            agent_health["resource_usage"] = resource_usage  # Add this line before return
            
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
            # Add resource monitoring
            try:
                import psutil
                resource_usage = {
                    "cpu_percent": psutil.cpu_percent(),
                    "memory_percent": psutil.virtual_memory().percent
                }
                # Add resource metrics
                metrics.append(f'tebsarvis_cpu_percent {resource_usage["cpu_percent"]}')
                metrics.append(f'tebsarvis_memory_percent {resource_usage["memory_percent"]}')
            except ImportError:
                metrics.append('# psutil not available for resource monitoring')
            except Exception as e:
                metrics.append(f'# Error getting resource metrics: {str(e)}')
        
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

async def check_workspace_health():
    """Check workspace components and file system health"""
    try:
        workspace_health = {
            "status": "healthy",
            "components": {},
            "recommendations": []
        }
        
        # Check agent files
        agent_files = {
            "resolution_agent": "agents/reactive/resolution_agent.py",
            "search_agent": "agents/reactive/search_agent.py",
            "conversation_agent": "agents/reactive/conversation_agent.py",
            "context_agent": "agents/reactive/context_agent.py",
            "pattern_detection_agent": "agents/proactive/pattern_detection_agent.py",
            "alerting_agent": "agents/proactive/alerting_agent.py"
        }
        
        missing_files = []
        for name, path in agent_files.items():
            full_path = os.path.join(os.path.dirname(__file__), "..", "..", path)
            if not os.path.exists(full_path):
                missing_files.append(path)
        
        if missing_files:
            workspace_health["status"] = "degraded"
            workspace_health["components"]["missing_files"] = missing_files
            workspace_health["recommendations"].append(
                f"Restore missing agent files: {', '.join(missing_files)}"
            )
        
        # Check write permissions
        try:
            test_file = os.path.join(os.path.dirname(__file__), "test_write.tmp")
            with open(test_file, "w") as f:
                f.write("test")
            os.remove(test_file)
            workspace_health["components"]["write_permissions"] = "ok"
        except Exception as e:
            workspace_health["status"] = "degraded"
            workspace_health["components"]["write_permissions"] = str(e)
            workspace_health["recommendations"].append(
                "Fix workspace write permissions"
            )
        
        return workspace_health
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "recommendations": ["Check workspace file system permissions"]
        }

async def check_network_health():
    """Check network connectivity and performance"""
    try:
        network_health = {
            "status": "healthy",
            "components": {},
            "recommendations": []
        }
        
        # Test connection pool
        if connection_pool:
            network_health["components"]["connection_pool"] = {
                "active_connections": connection_pool.size,
                "limit": connection_pool.limit
            }
            if connection_pool.size > connection_pool.limit * 0.8:
                network_health["status"] = "degraded"
                network_health["recommendations"].append(
                    "High number of active connections, consider increasing pool limit"
                )
        
        # Test basic connectivity
        basic_urls = [
            "https://management.azure.com",
            "https://graph.microsoft.com"
        ]
        
        async def test_url(url):
            try:
                async with aiohttp.ClientSession(connector=connection_pool) as session:
                    start = time.time()
                    async with session.get(url) as response:
                        latency = time.time() - start
                        return url, {
                            "status": response.status,
                            "latency": latency
                        }
            except Exception as e:
                return url, {"error": str(e)}
        
        # Run connectivity tests in parallel
        test_results = await asyncio.gather(*[test_url(url) for url in basic_urls])
        network_health["components"]["connectivity"] = dict(test_results)
        
        # Analyze results
        for url, result in dict(test_results).items():
            if "error" in result:
                network_health["status"] = "degraded"
                network_health["recommendations"].append(
                    f"Check connectivity to {url}"
                )
            elif result.get("latency", 0) > 1.0:  # High latency threshold
                if network_health["status"] == "healthy":
                    network_health["status"] = "degraded"
                network_health["recommendations"].append(
                    f"High latency to {url}: {result['latency']:.2f}s"
                )
        
        return network_health
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "recommendations": ["Check network configuration and DNS resolution"]
        }