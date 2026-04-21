"""
Broker package.

Concrete broker imports are lazy to avoid hard dependency on broker-specific
packages (shioaji, async_rithmic) when only one broker is needed.
"""

from brokers.base import BrokerProtocol, route_order

__all__ = ["BrokerProtocol", "route_order"]


def __getattr__(name: str):
    if name == "ShioajiBroker":
        from brokers.shioaji_broker import ShioajiBroker
        return ShioajiBroker
    if name == "RithmicBroker":
        from brokers.rithmic_broker import RithmicBroker
        return RithmicBroker
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
