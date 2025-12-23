# Advanced Sentiment Analysis System

A complete containerized NLP application featuring transformer-based sentiment analysis with vector database integration.

## Features

- **Transformer-based Sentiment Analysis**: Uses DistilBERT for fast and accurate sentiment detection
- **Vector Database Integration**: ChromaDB for storing and retrieving semantic embeddings
- **Modern Web Interface**: Gradio-based UI with real-time visualizations
- **Containerized Deployment**: Docker & Docker Compose for easy deployment
- **Batch Processing**: Analyze multiple texts simultaneously
- **Semantic Search**: Find similar texts using vector similarity

## Architecture

## how to Run
First go Powershel
    docker ps 
    docker run -d -p 8001:8000 chromadb/chroma:latest   
    Test ChromaDB connection
    python test_chromadb.py
    then run main file
        python main.py

## Final Running
    # check all running containers
        docker ps
        docker stop containers_name
    # Now start your app
        docker-compose up
        or
        docker-compose up --build



