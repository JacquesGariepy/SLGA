"""Integration tests for SLGA-Plus pipelines.

This module contains end-to-end integration tests covering:
- Data pipeline (dataset loading, processing, tokenization, collation)
- Training pipeline (model initialization, forward/backward, optimization)
- Model-data integration (loader to model, device transfers)
- Generation pipeline (model to generator, sampling, decoding)
- Attention pipeline (embedding to attention, landmark selection)
"""
