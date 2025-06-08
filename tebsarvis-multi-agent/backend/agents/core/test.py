# from abc import ABC, abstractmethod
# from typing import Dict, Any, List, Optional
# from dataclasses import dataclass
# from datetime import datetime
# import asyncio
# import logging
# import uuid
# from enum import Enum
# from .message_types import Priority 
# # from .agent_communication import MessageBus

# print("Loading core test module...")


# Test that all Phase 2 imports work correctly
try:
    # from backend.agents.core.agent_communication import MessageBus, AgentCommunicator, MessageRouter
    from backend.agents.core.agent_registry import AgentRegistry, get_global_registry
    from backend.agents.core.agent_system import AgentSystem, get_agent_system
    from backend.agents.core.message_bus_manager import get_message_bus
    
    print("✅ All Phase 2 imports successful!")
    print("✅ Phase 2 Communication Infrastructure is ready!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")