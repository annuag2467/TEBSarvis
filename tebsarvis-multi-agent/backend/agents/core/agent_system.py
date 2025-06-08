"""
Integrated Agent System Manager
Coordinates MessageBus and AgentRegistry for unified agent management.
"""

import asyncio
import logging
from typing import Optional, Dict, Any, List
from .message_types import Priority
from .agent_communication import MessageBus, MessageRouter
# from .agent_communication import AgentCommunicator
from .agent_registry import AgentRegistry, get_global_registry
from .base_agent import BaseAgent

class AgentSystem:
    """
    Unified agent system that coordinates MessageBus and AgentRegistry.
    Provides single point of control for agent lifecycle management.
    """
    
    def __init__(self):
        self.message_bus = MessageBus()
        self.registry = AgentRegistry()
        self.router = MessageRouter(self.message_bus)
        self.logger = logging.getLogger("agent_system")
        self.running = False
    
    async def start(self):
        """Start the complete agent system"""
        self.running = True
        
        # Start components
        await self.message_bus.start()
        await self.registry.start()
        
        self.logger.info("Agent System started successfully")
    
    async def stop(self):
        """Stop the complete agent system"""
        self.running = False
        
        # Stop components
        await self.registry.stop()
        await self.message_bus.stop()
        
        self.logger.info("Agent System stopped")
    
    async def register_agent(self, agent: BaseAgent, endpoint: Optional[str] = None,
                           metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Register an agent with both the registry and message bus.
        
        Args:
            agent: Agent instance to register
            endpoint: Optional endpoint for remote agents
            metadata: Additional metadata
            
        Returns:
            True if registration successful
        """
        try:
            # Register with registry first
            registry_success = await self.registry.register_agent(agent, endpoint, metadata)
            if not registry_success:
                return False
            
            # Register with message bus
            capability_names = [cap.name for cap in agent.get_capabilities()]
            self.message_bus.register_agent(agent.agent_id, agent, capability_names)
            
            # Set message bus reference in agent
            agent.message_bus = self.message_bus
            from .agent_communication import AgentCommunicator
            agent.communicator = AgentCommunicator(agent.agent_id, self.message_bus)
            
            self.logger.info(f"Agent {agent.agent_id} registered successfully with system")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register agent {agent.agent_id}: {str(e)}")
            return False
    
    async def deregister_agent(self, agent_id: str) -> bool:
        """Deregister agent from both systems"""
        try:
            # Deregister from both systems
            registry_success = await self.registry.deregister_agent(agent_id)
            self.message_bus.unregister_agent(agent_id)
            
            return registry_success
            
        except Exception as e:
            self.logger.error(f"Failed to deregister agent {agent_id}: {str(e)}")
            return False
    
    def get_message_bus(self) -> MessageBus:
        """Get the message bus instance"""
        return self.message_bus
    
    def get_registry(self) -> AgentRegistry:
        """Get the registry instance"""
        return self.registry
    
    def get_router(self) -> MessageRouter:
        """Get the message router instance"""
        return self.router

# Global system instance
_global_agent_system: Optional[AgentSystem] = None

async def get_agent_system() -> AgentSystem:
    """Get or create the global agent system"""
    global _global_agent_system
    if _global_agent_system is None:
        _global_agent_system = AgentSystem()
        await _global_agent_system.start()
    return _global_agent_system

async def shutdown_agent_system():
    """Shutdown the global agent system"""
    global _global_agent_system
    if _global_agent_system:
        await _global_agent_system.stop()
        _global_agent_system = None
