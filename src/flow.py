"""
Flow Service - Backward compatibility module
This module re-exports classes from the split modules to maintain backward compatibility.
"""
# Import all classes from the split modules
from .flow_service import FlowService
from .special_flow_service import SpecialFlow1Service

# Re-export for backward compatibility
__all__ = [
    "FlowService",
    "SpecialFlow1Service",
]
