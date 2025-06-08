"""
Verify setup and configuration for TEBSarvis Multi-Agent System
"""
import os
import sys
import json
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Tuple

# Add backend to path
backend_path = Path(__file__).parent.parent / 'backend'
sys.path.append(str(backend_path))

from agents.core.agent_system import get_agent_system
from agents.core.agent_registry import get_global_registry
from agents.core.message_bus_manager import get_message_bus
from config.azure_config import AzureConfigManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("setup_verification")

def verify_environment() -> List[str]:
    """Verify required environment variables"""
    required_vars = [
        'AZURE_OPENAI_ENDPOINT',
        'AZURE_OPENAI_KEY',
        'COSMOS_DB_URL',
        'COSMOS_DB_KEY',
        'SEARCH_SERVICE_ENDPOINT',
        'SEARCH_API_KEY',
        'AzureWebJobsStorage',
        'APPINSIGHTS_INSTRUMENTATIONKEY'
    ]
    
    missing = []
    for var in required_vars:
        if not os.getenv(var):
            missing.append(var)
    return missing

def verify_azure_functions() -> List[str]:
    """Verify Azure Functions configuration"""
    function_path = backend_path / 'azure-functions'
    issues = []
    
    # Check each function app
    for app_dir in function_path.iterdir():
        if not app_dir.is_dir() or app_dir.name == 'shared':
            continue
            
        # Check required files
        required_files = ['function_app.py', 'host.json', 'local.settings.json']
        for file in required_files:
            if not (app_dir / file).exists():
                issues.append(f"Missing {file} in {app_dir.name}")
                
        # Verify local.settings.json
        settings_file = app_dir / 'local.settings.json'
        if settings_file.exists():
            try:
                with open(settings_file) as f:
                    settings = json.load(f)
                if not settings.get('Values', {}).get('FUNCTIONS_WORKER_RUNTIME'):
                    issues.append(f"Invalid local.settings.json in {app_dir.name}")
            except json.JSONDecodeError:
                issues.append(f"Invalid JSON in local.settings.json for {app_dir.name}")
                
    return issues

async def verify_agent_system() -> Tuple[bool, str]:
    """Verify agent system initialization"""
    try:
        # Initialize core components
        system = await get_agent_system()
        registry = get_global_registry()
        message_bus = await get_message_bus()
        
        # Start components
        await system.start()
        
        # Verify registry
        if not registry.get_registry_statistics():
            return False, "Registry not properly initialized"
            
        # Verify message bus
        if not message_bus.is_running():
            return False, "Message bus not running"
            
        return True, "Agent system verified successfully"
        
    except Exception as e:
        return False, f"Agent system verification failed: {str(e)}"
    
async def verify_azure_config() -> Tuple[bool, str]:
    """Verify Azure configuration"""
    try:
        config = AzureConfigManager()
        return True, "Azure configuration verified"
    except Exception as e:
        return False, f"Azure configuration failed: {str(e)}"

async def main():
    """Run all verifications"""
    logger.info("Starting TEBSarvis setup verification...")
    
    # Check environment variables
    missing_vars = verify_environment()
    if missing_vars:
        logger.error(f"Missing environment variables: {', '.join(missing_vars)}")
        return False
        
    # Check Azure Functions
    function_issues = verify_azure_functions()
    if function_issues:
        logger.error("Azure Functions issues found:")
        for issue in function_issues:
            logger.error(f"  - {issue}")
        return False
        
    # Verify Azure config
    azure_ok, azure_msg = await verify_azure_config()
    if not azure_ok:
        logger.error(azure_msg)
        return False
    logger.info(azure_msg)
    
    # Verify agent system
    system_ok, system_msg = await verify_agent_system()
    if not system_ok:
        logger.error(system_msg)
        return False
    logger.info(system_msg)
    
    logger.info("All verifications passed! System is ready to run.")
    return True

if __name__ == "__main__":
    if not asyncio.run(main()):
        sys.exit(1)
