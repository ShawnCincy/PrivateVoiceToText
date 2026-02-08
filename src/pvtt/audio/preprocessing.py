"""Audio preprocessing for pvtt.

Phase 1: preprocessing is handled by faster-whisper internally.
This module is reserved for Phase 2 (streaming pipeline) where
we need explicit audio normalization, resampling, and chunking.
"""

from __future__ import annotations
