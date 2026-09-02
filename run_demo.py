#!/usr/bin/env python3
"""
CausalPrune-AI Launcher Script
Starts the FastAPI server and serves the visual dashboard.
"""

import sys
import os
import uvicorn

def main():
    print("=" * 70)
    print("  🚀 CausalPrune-AI: Task-Aware Interventional LLM Pruning Engine")
    print("=" * 70)
    print("Starting FastAPI Server & Glassmorphic Visual Dashboard...")
    print("  ► Dashboard URL: http://localhost:8000")
    print("  ► REST API Docs: http://localhost:8000/docs")
    print("=" * 70)

    # Ensure root path is in sys.path
    root_dir = os.path.dirname(os.path.abspath(__file__))
    if root_dir not in sys.path:
        sys.path.insert(0, root_dir)

    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    main()
