#!/usr/bin/env python3
"""
Test ChromaDB connection
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import chromadb
    from chromadb.config import Settings
    import time
    
    print("Testing ChromaDB connection...")
    
    # Try to connect
    client = chromadb.HttpClient(
        host="localhost",
        port=8001,
        settings=Settings(allow_reset=True)
    )
    
    # Test heartbeat
    heartbeat = client.heartbeat()
    print(f"✅ ChromaDB connection successful!")
    print(f"   Heartbeat: {heartbeat}")
    
    # List collections
    collections = client.list_collections()
    print(f"   Collections: {len(collections)} found")
    
    for coll in collections:
        print(f"   - {coll.name}: {coll.count()} items")
    
except Exception as e:
    print(f"❌ ChromaDB connection failed: {e}")
    print("\nTo start ChromaDB:")
    print("1. docker run -d -p 8001:8000 chromadb/chroma:latest")
    print("2. Then run this test again")