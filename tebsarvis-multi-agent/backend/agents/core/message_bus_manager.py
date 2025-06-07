from typing import Optional
from .agent_communication import MessageBus

_global_message_bus: Optional[MessageBus] = None

async def get_message_bus() -> MessageBus:
    """Get or create the global message bus instance"""
    global _global_message_bus
    if _global_message_bus is None:
        _global_message_bus = MessageBus()
        await _global_message_bus.start()
    return _global_message_bus

async def shutdown_message_bus():
    """Shutdown the global message bus"""
    global _global_message_bus
    if _global_message_bus:
        await _global_message_bus.stop()
        _global_message_bus = None