"""
Factory for creating landmark selectors.

Implements registry pattern for easy extension and creation of selectors.
"""

from typing import Any, Dict, Type

from ..landmarks.base import BaseLandmarkSelector
from ..landmarks.gumbel import GumbelSoftmaxSelector
from ..landmarks.heuristic import HeuristicLandmarkSelector
from ..landmarks.learned import (
    HybridLandmarkSelector,
    LearnedLandmarkSelector,
    PositionalLandmarkSelector,
)


class LandmarkSelectorFactory:
    """Factory for creating landmark selectors with registry pattern.

    Supports built-in selectors and registration of custom types.

    Built-in selectors:
    - "gumbel": GumbelSoftmaxSelector (differentiable, stable gradients)
    - "learned": LearnedLandmarkSelector (straight-through estimator)
    - "positional": PositionalLandmarkSelector (learned position patterns)
    - "hybrid": HybridLandmarkSelector (content + positional)
    - "uniform": HeuristicLandmarkSelector (uniform spacing)
    - "random": HeuristicLandmarkSelector (random selection)
    - "first": HeuristicLandmarkSelector (first-K positions)

    Example:
        >>> # Create built-in selector
        >>> factory = LandmarkSelectorFactory()
        >>> selector = factory.create(
        ...     "gumbel",
        ...     embed_dim=512,
        ...     num_landmarks=64,
        ...     temperature=1.0
        ... )
        >>>
        >>> # Register custom selector
        >>> from custom_module import MyCustomSelector
        >>> factory.register("custom", MyCustomSelector)
        >>> custom_selector = factory.create("custom", embed_dim=512, num_landmarks=64)
        >>>
        >>> # List available selectors
        >>> print(factory.list_available())
        ['gumbel', 'learned', 'positional', 'hybrid', 'uniform', 'random', 'first']
    """

    _registry: Dict[str, Type[BaseLandmarkSelector]] = {
        "gumbel": GumbelSoftmaxSelector,
        "learned": LearnedLandmarkSelector,
        "positional": PositionalLandmarkSelector,
        "hybrid": HybridLandmarkSelector,
    }

    @classmethod
    def create(cls, selector_type: str, **kwargs: Any) -> BaseLandmarkSelector:
        """Create a landmark selector instance.

        Args:
            selector_type: Type of selector to create
            **kwargs: Arguments passed to selector constructor

        Returns:
            selector: Landmark selector instance

        Raises:
            ValueError: If selector_type is not registered

        Example:
            >>> selector = LandmarkSelectorFactory.create(
            ...     "gumbel",
            ...     embed_dim=512,
            ...     num_landmarks=64,
            ...     temperature=1.0,
            ...     temperature_decay=0.999
            ... )
        """
        # Handle heuristic selectors (uniform, random, first)
        if selector_type in ["uniform", "random", "first"]:
            return HeuristicLandmarkSelector(
                num_landmarks=kwargs.get("num_landmarks", 64),
                strategy=selector_type,
                seed=kwargs.get("seed"),
            )

        # Handle registered selectors
        if selector_type not in cls._registry:
            raise ValueError(
                f"Unknown selector type: {selector_type}. " f"Available: {cls.list_available()}"
            )

        selector_class = cls._registry[selector_type]
        return selector_class(**kwargs)

    @classmethod
    def register(cls, name: str, selector_class: Type[BaseLandmarkSelector]) -> None:
        """Register a custom landmark selector.

        Args:
            name: Name to register selector under
            selector_class: Selector class (must inherit BaseLandmarkSelector)

        Raises:
            TypeError: If selector_class doesn't inherit BaseLandmarkSelector

        Example:
            >>> from custom_module import MyCustomSelector
            >>> LandmarkSelectorFactory.register("custom", MyCustomSelector)
            >>> selector = LandmarkSelectorFactory.create("custom", num_landmarks=64)
        """
        if not issubclass(selector_class, BaseLandmarkSelector):
            raise TypeError(f"{selector_class.__name__} must inherit from BaseLandmarkSelector")

        cls._registry[name] = selector_class

    @classmethod
    def list_available(cls) -> list:
        """List all available selector types.

        Returns:
            types: List of registered selector type names

        Example:
            >>> available = LandmarkSelectorFactory.list_available()
            >>> print(available)
            ['gumbel', 'learned', 'positional', 'hybrid', 'uniform', 'random', 'first']
        """
        # Include heuristic strategies
        heuristic = ["uniform", "random", "first"]
        return sorted(list(cls._registry.keys()) + heuristic)

    @classmethod
    def unregister(cls, name: str) -> None:
        """Remove a selector from registry.

        Args:
            name: Name of selector to remove

        Example:
            >>> LandmarkSelectorFactory.unregister("custom")
        """
        if name in cls._registry:
            del cls._registry[name]
