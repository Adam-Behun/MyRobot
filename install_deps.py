#!/usr/bin/env python3
"""
Pre-cache models during Docker build to avoid runtime delays.
This script downloads and caches the Silero VAD model that Pipecat uses.
"""

import torch
import sys

print("=" * 60)
print("Pre-caching models for production deployment...")
print("=" * 60)

try:
    # Download and cache Silero VAD model
    # This is used by pipecat.audio.vad.silero
    print("\n1. Downloading Silero VAD model...")
    model = torch.hub.load(
        'snakers4/silero-vad',
        'silero_vad',
        force_reload=False,
        trust_repo=True  # Required for newer PyTorch versions
    )
    print("   ✓ Silero VAD model cached successfully")
    
    # Verify the model loads correctly
    print("\n2. Verifying model loads correctly...")
    if model is not None:
        print("   ✓ Model verification successful")
    else:
        print("   ✗ Model verification failed")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("Model pre-caching completed successfully!")
    print("=" * 60)
    
except Exception as e:
    print(f"\n✗ Error during model caching: {e}")
    print("\nThis might not be critical - continuing with build")
    # Don't fail the build if model caching fails
    # The model will be downloaded at runtime instead
    sys.exit(0)