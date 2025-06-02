"""
Azure Configuration Management for TEBSarvis Multi-Agent System
Centralized configuration for all Azure services and resources.
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass
import logging

@dataclass
class AzureOpenAIConfig:
    """Configuration for Azure OpenAI services"""
    endpoint: str
    api_key: str
    api_version: str
    chat_model: str
    embedding_model: str
    deployment_name: str
    max_tokens: int = 4000
    temperature: float = 0.7
    timeout: int = 30

@dataclass
class CosmosDBConfig:
    """Configuration for Azure Cosmos DB"""
    url: str
    key: str
    database_name: str
    container_name: str
    partition_key: str = "/category"
    throughput: int = 400
    consistency_level: str = "Session"
    timeout: int = 30

@dataclass
class CognitiveSearchConfig:
    """Configuration for Azure Cognitive Search"""
    endpoint: str
    api_key: str
    index_name: str
    semantic_config_name: str = "semantic-config"
    vector_config_name: str = "vector-config"
    timeout: int = 30

@dataclass
class FunctionAppConfig:
    """Configuration for Azure Functions"""
    worker_runtime: str
    extension_version: str
    worker_process_count: int
    auth_level: str
    cors_origins: list
    timeout: int = 300

class AzureConfigManager:
    """
    Centralized configuration manager for Azure services.
    Handles environment variable loading and validation.
    """
    
    def __init__(self):
        self.logger = logging.getLogger("azure.config")
        self._validate_environment()
        
        # Initialize service configurations
        self.openai = self._load_openai_config()
        self.cosmos = self._load_cosmos_config()
        self.search = self._load_search_config()
        self.functions = self._load_functions_config()
        
        # Agent system configuration
        self.agent_config = self._load_agent_config()
        
        # Performance and monitoring
        self.performance = self._load_performance_config()
        
    def _validate_environment(self):
        """Validate that required environment variables are set"""
        required_vars = [
            'AZURE_OPENAI_ENDPOINT',
            'AZURE_OPENAI_KEY',
            'COSMOS_DB_URL',
            'COSMOS_DB_KEY',
            'SEARCH_SERVICE_ENDPOINT',
            'SEARCH_API_KEY'
        ]
        
        missing_vars = []
        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)
        
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
        
        self.logger.info("Environment validation completed successfully")
    
    def _load_openai_config(self) -> AzureOpenAIConfig:
        """Load Azure OpenAI configuration"""
        return AzureOpenAIConfig(
            endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
            api_key=os.getenv('AZURE_OPENAI_KEY'),
            api_version=os.getenv('AZURE_OPENAI_API_VERSION', '2024-02-01'),
            chat_model=os.getenv('AZURE_OPENAI_CHAT_MODEL', 'gpt-4'),
            embedding_model=os.getenv('AZURE_OPENAI_EMBEDDING_MODEL', 'text-embedding-ada-002'),
            deployment_name=os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME', 'gpt-4-deployment'),
            max_tokens=int(os.getenv('AZURE_OPENAI_MAX_TOKENS', '4000')),
            temperature=float(os.getenv('AZURE_OPENAI_TEMPERATURE', '0.7')),
            timeout=int(os.getenv('AZURE_OPENAI_TIMEOUT', '30'))
        )
    
    def _load_cosmos_config(self) -> CosmosDBConfig:
        """Load Cosmos DB configuration"""
        return CosmosDBConfig(
            url=os.getenv('COSMOS_DB_URL'),
            key=os.getenv('COSMOS_DB_KEY'),
            database_name=os.getenv('COSMOS_DB_DATABASE', 'tebsarvis'),
            container_name=os.getenv('COSMOS_DB_CONTAINER', 'incidents'),
            partition_key=os.getenv('COSMOS_DB_PARTITION_KEY', '/category'),
            throughput=int(os.getenv('COSMOS_DB_THROUGHPUT', '400')),
            consistency_level=os.getenv('COSMOS_DB_CONSISTENCY_LEVEL', 'Session'),
            timeout=int(os.getenv('COSMOS_DB_REQUEST_TIMEOUT', '30'))
        )
    
    def _load_search_config(self) -> CognitiveSearchConfig:
        """Load Cognitive Search configuration"""
        return CognitiveSearchConfig(
            endpoint=os.getenv('SEARCH_SERVICE_ENDPOINT'),
            api_key=os.getenv('SEARCH_API_KEY'),
            index_name=os.getenv('SEARCH_INDEX_NAME', 'incidents-index'),
            semantic_config_name=os.getenv('SEARCH_SEMANTIC_CONFIG', 'semantic-config'),
            vector_config_name=os.getenv('SEARCH_VECTOR_CONFIG', 'vector-config'),
            timeout=int(os.getenv('SEARCH_TIMEOUT', '30'))
        )
    
    def _load_functions_config(self) -> FunctionAppConfig:
        """Load Azure Functions configuration"""
        cors_origins = os.getenv('CORS_ORIGINS', 'http://localhost:3000').split(',')
        
        return FunctionAppConfig(
            worker_runtime=os.getenv('AZURE_FUNCTIONS_WORKER_RUNTIME', 'python'),
            extension_version=os.getenv('FUNCTIONS_EXTENSION_VERSION', '~4'),
            worker_process_count=int(os.getenv('FUNCTIONS_WORKER_PROCESS_COUNT', '1')),
            auth_level=os.getenv('FUNCTION_AUTH_LEVEL', 'Function'),
            cors_origins=cors_origins,
            timeout=int(os.getenv('FUNCTION_TIMEOUT', '300'))
        )
    
    def _load_agent_config(self) -> Dict[str, Any]:
        """Load agent system configuration"""
        return {
            'registry_timeout': int(os.getenv('AGENT_REGISTRY_TIMEOUT', '300')),
            'message_bus_timeout': int(os.getenv('MESSAGE_BUS_TIMEOUT', '30')),
            'max_concurrent_agents': int(os.getenv('MAX_CONCURRENT_AGENTS', '20')),
            'heartbeat_interval': int(os.getenv('AGENT_HEARTBEAT_INTERVAL', '30')),
            'max_retries': int(os.getenv('MAX_AGENT_RETRIES', '3')),
            'task_timeout': int(os.getenv('AGENT_TASK_TIMEOUT', '300')),
            'collaboration_timeout': int(os.getenv('COLLABORATION_SESSION_TIMEOUT', '1800')),
            'workflow_timeout': int(os.getenv('WORKFLOW_EXECUTION_TIMEOUT', '900'))
        }
    
    def _load_performance_config(self) -> Dict[str, Any]:
        """Load performance and monitoring configuration"""
        return {
            'max_embedding_batch_size': int(os.getenv('MAX_EMBEDDING_BATCH_SIZE', '100')),
            'search_cache_ttl': int(os.getenv('SEARCH_RESULT_CACHE_TTL', '300')),
            'log_level': os.getenv('LOG_LEVEL', 'INFO'),
            'app_insights_connection': os.getenv('APPLICATION_INSIGHTS_CONNECTION_STRING'),
            'webhook_secret': os.getenv('WEBHOOK_SECRET_KEY'),
            'notification_email_endpoint': os.getenv('NOTIFICATION_EMAIL_ENDPOINT'),
            'teams_webhook_url': os.getenv('TEAMS_WEBHOOK_URL')
        }
    
    def get_service_config(self, service_name: str) -> Any:
        """Get configuration for a specific service"""
        service_configs = {
            'openai': self.openai,
            'cosmos': self.cosmos,
            'search': self.search,
            'functions': self.functions
        }
        
        return service_configs.get(service_name.lower())
    
    def get_connection_string(self, service_name: str) -> str:
        """Get connection string for a service"""
        if service_name.lower() == 'cosmos':
            return f"AccountEndpoint={self.cosmos.url};AccountKey={self.cosmos.key};"
        elif service_name.lower() == 'search':
            return f"endpoint={self.search.endpoint};key={self.search.api_key}"
        elif service_name.lower() == 'openai':
            return f"endpoint={self.openai.endpoint};key={self.openai.api_key}"
        else:
            raise ValueError(f"Unknown service: {service_name}")
    
    def validate_configuration(self) -> Dict[str, bool]:
        """Validate all service configurations"""
        validation_results = {
            'openai': self._validate_openai(),
            'cosmos': self._validate_cosmos(),
            'search': self._validate_search(),
            'functions': self._validate_functions()
        }
        
        return validation_results
    
    def _validate_openai(self) -> bool:
        """Validate OpenAI configuration"""
        try:
            required_fields = ['endpoint', 'api_key', 'chat_model', 'embedding_model']
            for field in required_fields:
                if not getattr(self.openai, field):
                    return False
            return True
        except Exception:
            return False
    
    def _validate_cosmos(self) -> bool:
        """Validate Cosmos DB configuration"""
        try:
            required_fields = ['url', 'key', 'database_name', 'container_name']
            for field in required_fields:
                if not getattr(self.cosmos, field):
                    return False
            return True
        except Exception:
            return False
    
    def _validate_search(self) -> bool:
        """Validate Search configuration"""
        try:
            required_fields = ['endpoint', 'api_key', 'index_name']
            for field in required_fields:
                if not getattr(self.search, field):
                    return False
            return True
        except Exception:
            return False
    
    def _validate_functions(self) -> bool:
        """Validate Functions configuration"""
        try:
            required_fields = ['worker_runtime', 'extension_version']
            for field in required_fields:
                if not getattr(self.functions, field):
                    return False
            return True
        except Exception:
            return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'openai': {
                'endpoint': self.openai.endpoint[:50] + "..." if len(self.openai.endpoint) > 50 else self.openai.endpoint,
                'model': self.openai.chat_model,
                'embedding_model': self.openai.embedding_model
            },
            'cosmos': {
                'database': self.cosmos.database_name,
                'container': self.cosmos.container_name,
                'throughput': self.cosmos.throughput
            },
            'search': {
                'index': self.search.index_name,
                'endpoint': self.search.endpoint[:50] + "..." if len(self.search.endpoint) > 50 else self.search.endpoint
            },
            'agent_config': self.agent_config,
            'performance': self.performance
        }

# Global configuration instance
_config_instance: Optional[AzureConfigManager] = None

def get_azure_config() -> AzureConfigManager:
    """Get the global Azure configuration instance"""
    global _config_instance
    if _config_instance is None:
        _config_instance = AzureConfigManager()
    return _config_instance

def reload_config() -> AzureConfigManager:
    """Reload configuration from environment variables"""
    global _config_instance
    _config_instance = AzureConfigManager()
    return _config_instance