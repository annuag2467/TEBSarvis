"""
Agent Registry for TEBSarvis Multi-Agent System
Handles agent discovery, registration, and capability tracking.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import json
import weakref

from .base_agent import BaseAgent, AgentCapability, AgentStatus

class RegistrationStatus(Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    UNHEALTHY = "unhealthy"
    DEREGISTERED = "deregistered"

@dataclass
class AgentRegistration:
    """Agent registration information"""
    agent_id: str
    agent_type: str
    capabilities: List[AgentCapability]
    status: RegistrationStatus
    registered_at: datetime
    last_heartbeat: datetime
    health_status: Dict[str, Any]
    load_metrics: Dict[str, float]
    metadata: Dict[str, Any]
    endpoint: Optional[str] = None
    version: str = "1.0.0"

class AgentRegistry:
    """
    Central registry for agent discovery and management.
    Tracks agent capabilities, health, and availability.
    """
    
    def __init__(self):
        self.agents: Dict[str, AgentRegistration] = {}
        self.agent_references: Dict[str, weakref.ref] = {}  # Weak references to agent objects
        self.capability_index: Dict[str, Set[str]] = {}  # capability -> set of agent_ids
        self.type_index: Dict[str, Set[str]] = {}  # agent_type -> set of agent_ids
        self.logger = logging.getLogger("agent_registry")
        
        # Configuration
        self.heartbeat_interval = 30  # seconds
        self.heartbeat_timeout = 90  # seconds
        self.health_check_interval = 60  # seconds
        
        # Statistics
        self.stats = {
            'total_registrations': 0,
            'active_agents': 0,
            'total_capabilities': 0,
            'registry_uptime': datetime.now()
        }
        
        # Start background tasks
        self.running = False
        self.background_tasks = []
    
    async def start(self):
        """Start the agent registry"""
        self.running = True
        
        # Start background monitoring tasks
        self.background_tasks = [
            asyncio.create_task(self._heartbeat_monitor()),
            asyncio.create_task(self._health_checker()),
            asyncio.create_task(self._cleanup_stale_agents())
        ]
        
        self.logger.info("Agent Registry started")
    
    async def stop(self):
        """Stop the agent registry"""
        self.running = False
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self.background_tasks:
            await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        self.logger.info("Agent Registry stopped")
    
    async def register_agent(self, agent: BaseAgent, endpoint: Optional[str] = None,
                           metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Register an agent with the registry.
        
        Args:
            agent: Agent instance to register
            endpoint: Optional endpoint URL for remote agents
            metadata: Additional metadata about the agent
            
        Returns:
            True if registration successful, False otherwise
        """
        try:
            agent_id = agent.agent_id
            agent_type = agent.agent_type
            capabilities = agent.get_capabilities()
            
            # Create registration record
            registration = AgentRegistration(
                agent_id=agent_id,
                agent_type=agent_type,
                capabilities=capabilities,
                status=RegistrationStatus.ACTIVE,
                registered_at=datetime.now(),
                last_heartbeat=datetime.now(),
                health_status={'status': 'healthy'},
                load_metrics={'cpu': 0.0, 'memory': 0.0, 'active_tasks': 0},
                metadata=metadata or {},
                endpoint=endpoint
            )
            
            # Store registration
            self.agents[agent_id] = registration
            
            # Store weak reference to agent object
            self.agent_references[agent_id] = weakref.ref(agent)
            
            # Update indexes
            self._update_capability_index(agent_id, capabilities)
            self._update_type_index(agent_id, agent_type)
            
            # Update statistics
            self.stats['total_registrations'] += 1
            self._update_active_count()
            
            self.logger.info(f"Agent {agent_id} ({agent_type}) registered successfully")
            
            # Notify other systems about new agent
            await self._notify_agent_registered(registration)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register agent {agent.agent_id}: {str(e)}")
            return False
    
    async def deregister_agent(self, agent_id: str) -> bool:
        """
        Deregister an agent from the registry.
        
        Args:
            agent_id: ID of agent to deregister
            
        Returns:
            True if deregistration successful, False otherwise
        """
        try:
            if agent_id not in self.agents:
                self.logger.warning(f"Attempted to deregister unknown agent: {agent_id}")
                return False
            
            registration = self.agents[agent_id]
            
            # Update status
            registration.status = RegistrationStatus.DEREGISTERED
            
            # Remove from indexes
            self._remove_from_capability_index(agent_id, registration.capabilities)
            self._remove_from_type_index(agent_id, registration.agent_type)
            
            # Remove references
            if agent_id in self.agent_references:
                del self.agent_references[agent_id]
            
            # Remove from registry
            del self.agents[agent_id]
            
            # Update statistics
            self._update_active_count()
            
            self.logger.info(f"Agent {agent_id} deregistered successfully")
            
            # Notify other systems about agent removal
            await self._notify_agent_deregistered(registration)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to deregister agent {agent_id}: {str(e)}")
            return False
    
    async def update_agent_status(self, agent_id: str, status: RegistrationStatus,
                                health_status: Optional[Dict[str, Any]] = None,
                                load_metrics: Optional[Dict[str, float]] = None) -> bool:
        """
        Update agent status and metrics.
        
        Args:
            agent_id: ID of agent to update
            status: New registration status
            health_status: Updated health information
            load_metrics: Updated load metrics
            
        Returns:
            True if update successful, False otherwise
        """
        try:
            if agent_id not in self.agents:
                self.logger.warning(f"Attempted to update unknown agent: {agent_id}")
                return False
            
            registration = self.agents[agent_id]
            
            # Update status
            old_status = registration.status
            registration.status = status
            registration.last_heartbeat = datetime.now()
            
            # Update health status
            if health_status:
                registration.health_status.update(health_status)
            
            # Update load metrics
            if load_metrics:
                registration.load_metrics.update(load_metrics)
            
            # Log significant status changes
            if old_status != status:
                self.logger.info(f"Agent {agent_id} status changed from {old_status.value} to {status.value}")
            
            # Update statistics
            self._update_active_count()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update agent {agent_id} status: {str(e)}")
            return False
    
    async def heartbeat(self, agent_id: str, health_data: Optional[Dict[str, Any]] = None) -> bool:
        """
        Record agent heartbeat.
        
        Args:
            agent_id: ID of agent sending heartbeat
            health_data: Optional health data from agent
            
        Returns:
            True if heartbeat recorded, False otherwise
        """
        try:
            if agent_id not in self.agents:
                self.logger.warning(f"Heartbeat from unknown agent: {agent_id}")
                return False
            
            registration = self.agents[agent_id]
            registration.last_heartbeat = datetime.now()
            
            # Update health data if provided
            if health_data:
                registration.health_status.update(health_data)
            
            # Mark as active if it was inactive
            if registration.status == RegistrationStatus.INACTIVE:
                registration.status = RegistrationStatus.ACTIVE
                self.logger.info(f"Agent {agent_id} reactivated via heartbeat")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to record heartbeat for agent {agent_id}: {str(e)}")
            return False
    
    def find_agents_by_capability(self, capability: str) -> List[AgentRegistration]:
        """
        Find agents that have a specific capability.
        
        Args:
            capability: Capability name to search for
            
        Returns:
            List of agent registrations with the capability
        """
        agent_ids = self.capability_index.get(capability, set())
        return [
            self.agents[agent_id] for agent_id in agent_ids 
            if agent_id in self.agents and self.agents[agent_id].status == RegistrationStatus.ACTIVE
        ]
    
    def find_agents_by_type(self, agent_type: str) -> List[AgentRegistration]:
        """
        Find agents of a specific type.
        
        Args:
            agent_type: Agent type to search for
            
        Returns:
            List of agent registrations of the specified type
        """
        agent_ids = self.type_index.get(agent_type, set())
        return [
            self.agents[agent_id] for agent_id in agent_ids 
            if agent_id in self.agents and self.agents[agent_id].status == RegistrationStatus.ACTIVE
        ]
    
    def get_best_agent_for_capability(self, capability: str) -> Optional[AgentRegistration]:
        """
        Get the best available agent for a specific capability.
        
        Args:
            capability: Capability name
            
        Returns:
            Best agent registration or None if no suitable agent found
        """
        candidates = self.find_agents_by_capability(capability)
        
        if not candidates:
            return None
        
        # Filter healthy agents
        healthy_agents = [
            agent for agent in candidates 
            if agent.health_status.get('status') == 'healthy'
        ]
        
        if not healthy_agents:
            return None
        
        # Sort by load (lower is better)
        healthy_agents.sort(key=lambda a: a.load_metrics.get('active_tasks', 0))
        
        return healthy_agents[0]
    
    def get_agent_by_id(self, agent_id: str) -> Optional[AgentRegistration]:
        """
        Get agent registration by ID.
        
        Args:
            agent_id: Agent ID to look up
            
        Returns:
            Agent registration or None if not found
        """
        return self.agents.get(agent_id)
    
    def get_all_active_agents(self) -> List[AgentRegistration]:
        """Get list of all active agents"""
        return [
            registration for registration in self.agents.values()
            if registration.status == RegistrationStatus.ACTIVE
        ]
    
    def get_all_capabilities(self) -> Set[str]:
        """Get set of all available capabilities"""
        return set(self.capability_index.keys())
    
    def get_registry_statistics(self) -> Dict[str, Any]:
        """Get registry statistics"""
        active_count = len([
            a for a in self.agents.values() 
            if a.status == RegistrationStatus.ACTIVE
        ])
        
        return {
            'total_agents': len(self.agents),
            'active_agents': active_count,
            'inactive_agents': len(self.agents) - active_count,
            'total_capabilities': len(self.capability_index),
            'agent_types': len(self.type_index),
            'registry_uptime': (datetime.now() - self.stats['registry_uptime']).total_seconds(),
            'last_updated': datetime.now().isoformat()
        }
    
    def get_capability_coverage(self) -> Dict[str, int]:
        """Get coverage statistics for each capability"""
        coverage = {}
        for capability, agent_ids in self.capability_index.items():
            active_agents = [
                agent_id for agent_id in agent_ids
                if agent_id in self.agents and self.agents[agent_id].status == RegistrationStatus.ACTIVE
            ]
            coverage[capability] = len(active_agents)
        
        return coverage
    
    async def _heartbeat_monitor(self):
        """Monitor agent heartbeats and mark stale agents as inactive"""
        while self.running:
            try:
                current_time = datetime.now()
                timeout_threshold = current_time - timedelta(seconds=self.heartbeat_timeout)
                
                for agent_id, registration in self.agents.items():
                    if (registration.status == RegistrationStatus.ACTIVE and 
                        registration.last_heartbeat < timeout_threshold):
                        
                        self.logger.warning(f"Agent {agent_id} heartbeat timeout, marking as inactive")
                        registration.status = RegistrationStatus.INACTIVE
                
                await asyncio.sleep(self.heartbeat_interval)
                
            except Exception as e:
                self.logger.error(f"Error in heartbeat monitor: {str(e)}")
                await asyncio.sleep(5)
    
    async def _health_checker(self):
        """Perform periodic health checks on registered agents"""
        while self.running:
            try:
                for agent_id, registration in self.agents.items():
                    if registration.status == RegistrationStatus.ACTIVE:
                        health_status = await self._check_agent_health(agent_id)
                        if health_status:
                            registration.health_status.update(health_status)
                
                await asyncio.sleep(self.health_check_interval)
                
            except Exception as e:
                self.logger.error(f"Error in health checker: {str(e)}")
                await asyncio.sleep(10)
    
    async def _check_agent_health(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Check health of a specific agent"""
        try:
            # Get agent reference
            agent_ref = self.agent_references.get(agent_id)
            if not agent_ref:
                return None
            
            agent = agent_ref()
            if not agent:
                # Agent has been garbage collected
                await self.deregister_agent(agent_id)
                return None
            
            # Get agent status
            status = agent.get_status()
            
            return {
                'status': 'healthy' if status['status'] != 'error' else 'unhealthy',
                'active_tasks': status['active_tasks'],
                'success_rate': status['metrics']['success_rate'],
                'last_activity': status['metrics']['last_activity'],
                'checked_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error checking health for agent {agent_id}: {str(e)}")
            return {'status': 'unhealthy', 'error': str(e)}
    
    async def _cleanup_stale_agents(self):
        """Clean up stale agent registrations"""
        while self.running:
            try:
                current_time = datetime.now()
                stale_threshold = current_time - timedelta(hours=24)  # 24 hours
                
                stale_agents = []
                for agent_id, registration in self.agents.items():
                    if (registration.status == RegistrationStatus.INACTIVE and 
                        registration.last_heartbeat < stale_threshold):
                        stale_agents.append(agent_id)
                
                for agent_id in stale_agents:
                    self.logger.info(f"Cleaning up stale agent: {agent_id}")
                    await self.deregister_agent(agent_id)
                
                await asyncio.sleep(3600)  # Check hourly
                
            except Exception as e:
                self.logger.error(f"Error in cleanup task: {str(e)}")
                await asyncio.sleep(300)  # Retry in 5 minutes
    
    def _update_capability_index(self, agent_id: str, capabilities: List[AgentCapability]):
        """Update the capability index with agent capabilities"""
        for capability in capabilities:
            capability_name = capability.name
            if capability_name not in self.capability_index:
                self.capability_index[capability_name] = set()
            self.capability_index[capability_name].add(agent_id)
    
    def _remove_from_capability_index(self, agent_id: str, capabilities: List[AgentCapability]):
        """Remove agent from capability index"""
        for capability in capabilities:
            capability_name = capability.name
            if capability_name in self.capability_index:
                self.capability_index[capability_name].discard(agent_id)
                # Clean up empty sets
                if not self.capability_index[capability_name]:
                    del self.capability_index[capability_name]
    
    def _update_type_index(self, agent_id: str, agent_type: str):
        """Update the type index"""
        if agent_type not in self.type_index:
            self.type_index[agent_type] = set()
        self.type_index[agent_type].add(agent_id)
    
    def _remove_from_type_index(self, agent_id: str, agent_type: str):
        """Remove agent from type index"""
        if agent_type in self.type_index:
            self.type_index[agent_type].discard(agent_id)
            # Clean up empty sets
            if not self.type_index[agent_type]:
                del self.type_index[agent_type]
    
    def _update_active_count(self):
        """Update active agent count in statistics"""
        self.stats['active_agents'] = len([
            a for a in self.agents.values() 
            if a.status == RegistrationStatus.ACTIVE
        ])
        self.stats['total_capabilities'] = len(self.capability_index)
    
    async def _notify_agent_registered(self, registration: AgentRegistration):
        """Notify systems about new agent registration"""
        # This could send notifications to monitoring systems, etc.
        self.logger.debug(f"Agent registration notification: {registration.agent_id}")
    
    async def _notify_agent_deregistered(self, registration: AgentRegistration):
        """Notify systems about agent deregistration"""
        # This could send notifications to monitoring systems, etc.
        self.logger.debug(f"Agent deregistration notification: {registration.agent_id}")
    
    def export_registry_state(self) -> Dict[str, Any]:
        """Export current registry state for persistence or debugging"""
        return {
            'agents': {
                agent_id: {
                    'agent_id': reg.agent_id,
                    'agent_type': reg.agent_type,
                    'capabilities': [asdict(cap) for cap in reg.capabilities],
                    'status': reg.status.value,
                    'registered_at': reg.registered_at.isoformat(),
                    'last_heartbeat': reg.last_heartbeat.isoformat(),
                    'health_status': reg.health_status,
                    'load_metrics': reg.load_metrics,
                    'metadata': reg.metadata,
                    'endpoint': reg.endpoint,
                    'version': reg.version
                }
                for agent_id, reg in self.agents.items()
            },
            'capability_index': {
                cap: list(agents) for cap, agents in self.capability_index.items()
            },
            'type_index': {
                agent_type: list(agents) for agent_type, agents in self.type_index.items()
            },
            'statistics': self.get_registry_statistics(),
            'exported_at': datetime.now().isoformat()
        }

# Global registry instance
_global_registry: Optional[AgentRegistry] = None

def get_global_registry() -> AgentRegistry:
    """Get the global agent registry instance"""
    global _global_registry
    if _global_registry is None:
        _global_registry = AgentRegistry()
    return _global_registry

async def initialize_global_registry():
    """Initialize the global agent registry"""
    registry = get_global_registry()
    await registry.start()
    return registry

async def shutdown_global_registry():
    """Shutdown the global agent registry"""
    global _global_registry
    if _global_registry:
        await _global_registry.stop()
        _global_registry = None






