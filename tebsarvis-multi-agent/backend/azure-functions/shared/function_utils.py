"""
Shared utilities for Azure Function Apps
Provides common patterns for agent initialization, error handling, and response formatting.
"""

import logging
import json
from datetime import datetime
from typing import Dict, Any, Optional
import azure.functions as func

class FunctionResponse:
    """Standardized response handler for Azure Functions"""
    
    @staticmethod
    def success(data: Dict[str, Any], status_code: int = 200) -> func.HttpResponse:
        """Create successful response"""
        return func.HttpResponse(
            json.dumps(data, indent=2),
            status_code=status_code,
            mimetype="application/json"
        )
    
    @staticmethod
    def error(message: str, error_type: str = "Internal server error", 
              status_code: int = 500, details: Optional[Dict] = None) -> func.HttpResponse:
        """Create error response"""
        error_response = {
            "error": error_type,
            "message": message,
            "timestamp": datetime.now().isoformat()
        }
        
        if details:
            error_response["details"] = details
            
        return func.HttpResponse(
            json.dumps(error_response),
            status_code=status_code,
            mimetype="application/json"
        )
    
    @staticmethod
    def validation_error(missing_fields: list) -> func.HttpResponse:
        """Create validation error response"""
        return FunctionResponse.error(
            f"Missing required fields: {missing_fields}",
            "Validation error",
            400,
            {"missing_fields": missing_fields}
        )

class AgentInitializer:
    """Standardized agent initialization for Azure Functions"""
    
    def __init__(self, agent_class, agent_name: str):
        self.agent_class = agent_class
        self.agent_name = agent_name
        self.agent = None
        self.azure_manager = None
        self.message_bus = None
        self.registry = None
        self.logger = logging.getLogger(agent_name)
    
    async def initialize(self):
        """Initialize all components"""
        try:
            if not self.agent:
                # Import here to avoid circular imports
                from core.agent_registry import get_global_registry
                from core.agent_communication import MessageBus
                from shared.azure_clients import AzureClientManager
                
                # Initialize Azure manager
                self.azure_manager = AzureClientManager()
                await self.azure_manager.initialize()
                
                # Initialize message bus and registry
                self.message_bus = MessageBus()
                await self.message_bus.start()
                
                self.registry = get_global_registry()
                await self.registry.start()
                
                # Initialize agent
                self.agent = self.agent_class()
                await self.agent.start()
                
                # Register agent
                await self.registry.register_agent(self.agent)
                
                self.logger.info(f"{self.agent_name} components initialized successfully")
                
        except Exception as e:
            self.logger.error(f"Error initializing {self.agent_name}: {str(e)}")
            raise
    
    async def cleanup(self):
        """Clean up resources"""
        try:
            if self.agent:
                await self.agent.stop()
            if self.registry:
                await self.registry.stop()
            if self.message_bus:
                await self.message_bus.stop()
                
            self.logger.info(f"{self.agent_name} resources cleaned up successfully")
            
        except Exception as e:
            self.logger.error(f"Error during {self.agent_name} cleanup: {str(e)}")

def validate_request(req: func.HttpRequest, required_fields: list) -> Optional[Dict[str, Any]]:
    """Validate request and return parsed JSON or None if invalid"""
    try:
        if not req.get_body():
            return None
        
        request_data = req.get_json()
        if not request_data:
            return None
        
        # Check required fields
        missing_fields = [field for field in required_fields if field not in request_data]
        if missing_fields:
            return None
        
        return request_data
        
    except ValueError:
        return None

def add_request_metadata(result: Dict[str, Any], function_name: str, 
                        processing_time: float, req: func.HttpRequest) -> Dict[str, Any]:
    """Add standard metadata to response"""
    metadata_key = f"{function_name.replace('-', '_')}_metadata"
    
    result[metadata_key] = result.get(metadata_key, {})
    result[metadata_key].update({
        'function_name': function_name,
        'processing_time_seconds': processing_time,
        'timestamp': datetime.now().isoformat(),
        'api_version': '1.0.0'
    })
    
    # Add request tracking
    result['request_id'] = req.headers.get('x-request-id', 'unknown')
    
    return result