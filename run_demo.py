#!/usr/bin/env python3
"""Launch the HeadRouter API + live dashboard (http://localhost:8000)."""

import os
import sys

import uvicorn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    from headrouter.config import load_config

    cfg = load_config(os.environ.get("HEADROUTER_CONFIG"))
    print(f"HeadRouter dashboard: http://localhost:{cfg['server']['port']}   API docs: /docs")
    uvicorn.run("api.main:app", host=cfg["server"]["host"], port=cfg["server"]["port"])
