"""
Domain Layer

This layer contains the core business logic and interfaces (protocols) of the SLGA architecture.
It is completely independent of frameworks, infrastructure, and implementation details.

Structure:
- entities: Core domain entities (TransformerBlock, Model protocols)
- services: Domain services (Attention, Landmark selection protocols)
- repositories: Data access interfaces (Dataset, Checkpoint, Tokenizer protocols)
- value_objects: Immutable configuration objects
"""

__all__ = [
    "entities",
    "repositories",
    "services",
    "value_objects",
]
