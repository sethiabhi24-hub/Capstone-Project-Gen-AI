import operator
from typing import TypedDict, Annotated, List, Any

class CapstoneState(TypedDict):
    messages: Annotated[List[str], operator.add]
    next_node: str
    customer_profile: dict
    client: Any
