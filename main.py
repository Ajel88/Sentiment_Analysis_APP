#!/usr/bin/env python3
"""
Main entry point for Advanced Sentiment Analysis System
"""
import os
import sys
import argparse
from typing import Dict, Any, List
import time

# Add src to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils import setup_logging, load_config
from src.data_preprocessor import TextPreprocessor
from src.embedding_generator import EmbeddingGenerator
from src.vector_db import VectorDatabase
import gradio as gr
import plotly.graph_objects as go
from transformers import pipeline
import numpy as np

logger = setup_logging()

class SentimentAnalysisSystem:
    """Complete sentiment analysis system with vector database"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize components
        logger.info("Initializing Sentiment Analysis System...")
        
        # Load sentiment model
        self.sentiment_model_name = config['model']['sentiment']
        logger.info(f"Loading sentiment model: {self.sentiment_model_name}")
        self.sentiment_nlp = pipeline(
            "sentiment-analysis",
            model=self.sentiment_model_name,
            device=-1  # Use CPU (set to 0 for GPU if available)
        )
        
        # Initialize preprocessor
        self.preprocessor = TextPreprocessor(self.sentiment_model_name)
        
        # Initialize embedding generator
        self.embedding_model_name = config['model']['embedding']
        logger.info(f"Loading embedding model: {self.embedding_model_name}")
        self.embedding_generator = EmbeddingGenerator(self.embedding_model_name)
        
        # Initialize vector database
        chroma_config = config['chroma']
        logger.info(f"Initializing ChromaDB connection at {chroma_config['host']}:{chroma_config['port']}")
        try:
            self.vector_db = VectorDatabase(
                host=chroma_config['host'],
                port=chroma_config['port'],
                collection_name=chroma_config['collection_name']
            )
            
            # Test connection
            collection_info = self.vector_db.get_collection_info()
            logger.info(f"ChromaDB connection successful. Collection: {collection_info}")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            logger.info("\n" + "="*60)
            logger.info("CHROMADB SETUP INSTRUCTIONS:")
            logger.info("="*60)
            logger.info("Option 1: Start ChromaDB with Docker:")
            logger.info("  docker run -d -p 8001:8000 chromadb/chroma:latest")
            logger.info("\nOption 2: Use Docker Compose:")
            logger.info("  docker-compose up")
            logger.info("\nOption 3: Check if ChromaDB is running:")
            logger.info("  docker ps | findstr chroma")
            logger.info("="*60)
            raise
        
        # Load sample data
        self._load_sample_data()
        
        logger.info("System initialization complete!")
    
    def _load_sample_data(self):
        """Load sample data into vector database"""
        sample_data = [
            {"text": "I absolutely love this product! It's amazing and works perfectly.", "label": "POSITIVE"},
            {"text": "This is the worst experience I've ever had. Terrible service.", "label": "NEGATIVE"},
            {"text": "The movie was okay, nothing special but not bad either.", "label": "NEUTRAL"},
            {"text": "I'm extremely happy with my purchase, would definitely recommend!", "label": "POSITIVE"},
            {"text": "Very disappointed with the quality, it broke after one day.", "label": "NEGATIVE"},
            {"text": "The food was delicious and the service was excellent!", "label": "POSITIVE"},
            {"text": "Waste of money, completely useless product.", "label": "NEGATIVE"},
            {"text": "Great value for the price, I'm satisfied with my decision.", "label": "POSITIVE"},
            {"text": "The customer support was horrible, they didn't help at all.", "label": "NEGATIVE"},
            {"text": "Best decision I've made, this changed my life for the better!", "label": "POSITIVE"}
        ]
        
        # Generate embeddings for sample data
        embedding_results = []
        for item in sample_data:
            metadata = {
                'label': item['label'],
                'source': 'sample_data',
                'length': len(item['text']),
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
            }
            result = self.embedding_generator.generate_embedding(item['text'], metadata)
            embedding_results.append(result)
        
        # Store in vector database
        try:
            self.vector_db.store_embeddings(embedding_results)
            logger.info(f"Loaded {len(sample_data)} sample texts into ChromaDB")
        except Exception as e:
            logger.error(f"Failed to load sample data: {e}")
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of input text"""
        if not text or not text.strip():
            return {
                "label": "NO_INPUT", 
                "score": 0.0, 
                "note": "Please enter some text.",
                "similar_texts": []
            }
        
        try:
            # Preprocess text
            cleaned_text, _ = self.preprocessor.preprocess_pipeline(text)
            
            # Get sentiment prediction
            max_chars = 1000
            safe_text = cleaned_text if len(cleaned_text) <= max_chars else cleaned_text[:max_chars]
            
            raw_result = self.sentiment_nlp(safe_text)
            result = raw_result[0] if isinstance(raw_result, list) else raw_result
            
            # Generate embedding
            embedding_result = self.embedding_generator.generate_embedding(
                cleaned_text,
                metadata={
                    'label': result.get('label', 'UNKNOWN'),
                    'score': float(result.get('score', 0.0)),
                    'input_length': len(text),
                    'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                    'model': self.sentiment_model_name
                }
            )
            
            # Store in vector database
            try:
                self.vector_db.store_embeddings([embedding_result])
            except Exception as e:
                logger.warning(f"Could not store embedding: {e}")
            
            # Find similar texts
            similar_texts = []
            try:
                similar_texts = self.vector_db.search_similar(
                    embedding_result.embedding,
                    top_k=3
                )
            except Exception as e:
                logger.warning(f"Could not search similar texts: {e}")
            
            return {
                "label": result.get("label", "UNKNOWN"),
                "score": float(result.get("score", 0.0)),
                "cleaned_text": cleaned_text,
                "similar_texts": similar_texts,
                "note": f"Model: {self.sentiment_model_name} | ChromaDB: Connected"
            }
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            return {
                "label": "ERROR",
                "score": 0.0,
                "cleaned_text": text[:100] + "..." if len(text) > 100 else text,
                "similar_texts": [],
                "note": f"Error: {str(e)}"
            }
    
    def batch_analyze(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Analyze sentiment for multiple texts"""
        results = []
        for text in texts:
            result = self.analyze_sentiment(text)
            results.append(result)
        return results
    
    def get_database_info(self) -> Dict[str, Any]:
        """Get database information"""
        try:
            return self.vector_db.get_collection_info()
        except:
            return {"status": "Not connected"}
    
    def get_model_info(self):
        """Get model information for display"""
        return [
            ["🤖 Sentiment Model", self.sentiment_model_name],
            ["🔤 Embedding Model", self.embedding_model_name],
            ["🗄️ Vector Database", "ChromaDB"],
            ["🔗 ChromaDB Host", f"{self.config['chroma']['host']}:{self.config['chroma']['port']}"],
            ["📁 Collection", self.config['chroma']['collection_name']],
            ["⚙️ Framework", "Hugging Face Transformers"],
            ["🎨 UI Framework", "Gradio"],
            ["🐳 Containerization", "Docker + Docker Compose"]
        ]

def build_gradio_interface(system: SentimentAnalysisSystem):
    """Build modern Gradio interface"""
    theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="gray",
        font=["Inter", "sans-serif"]
    )
    
    with gr.Blocks(
        title="Advanced Sentiment Analyzer",
        theme="soft"  # Simple string theme
    ) as demo:
        
        # CSS Styling
        gr.HTML("""
        <style>
            .fade-in {
                animation: fadeIn 1.2s ease-in-out;
            }
            
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(10px); }
                to { opacity: 1; transform: translateY(0); }
            }
            
            .hover-glow:hover {
                box-shadow: 0px 0px 18px rgba(0,140,255,0.5);
                transition: 0.3s;
            }
            
            .sentiment-bar {
                height: 22px;
                border-radius: 8px;
                transition: width 0.7s ease, background-color 0.7s ease;
                margin: 10px 0;
            }
            
            .positive { 
                color: #10b981; 
                font-weight: bold;
                background: linear-gradient(135deg, #10b98122 0%, #10b98111 100%);
                padding: 10px 15px;
                border-radius: 8px;
                border-left: 4px solid #10b981;
            }
            
            .negative { 
                color: #ef4444; 
                font-weight: bold;
                background: linear-gradient(135deg, #ef444422 0%, #ef444411 100%);
                padding: 10px 15px;
                border-radius: 8px;
                border-left: 4px solid #ef4444;
            }
            
            .neutral { 
                color: #6b7280; 
                font-weight: bold;
                background: linear-gradient(135deg, #6b728022 0%, #6b728011 100%);
                padding: 10px 15px;
                border-radius: 8px;
                border-left: 4px solid #6b7280;
            }
            
            .similar-item {
                background: #f8f9fa;
                border-radius: 8px;
                padding: 12px;
                margin: 8px 0;
                border-left: 4px solid #4f46e5;
                transition: transform 0.2s;
            }
            
            .similar-item:hover {
                transform: translateX(5px);
                background: #f1f5f9;
            }
            
            .status-connected {
                color: #10b981;
                font-weight: bold;
            }
            
            .status-disconnected {
                color: #ef4444;
                font-weight: bold;
            }
            
            .sentiment-display {
                padding: 15px;
                border-radius: 10px;
                font-size: 20px;
                font-weight: bold;
                text-align: center;
                margin: 10px 0;
                transition: all 0.3s ease;
            }
            
            .model-info-table {
                width: 100%;
                border-collapse: collapse;
                margin: 15px 0;
                background: white;
                border-radius: 8px;
                overflow: hidden;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }
            
            .model-info-table th {
                background: var(--block-title-background-fill);
                color: white;
                padding: 12px;
                text-align: left;
                font-weight: 600;
                font-size: 14px;
            }
            
            .model-info-table td {
                padding: 12px;
                border-bottom: 1px solid #e5e7eb;
                font-size: 13px;
            }
            
            .model-info-table tr:last-child td {
                border-bottom: none;
            }
            
            .model-info-table tr:hover {
                background: #f9fafb;
            }
            
            .model-icon {
                font-size: 16px;
                margin-right: 8px;
            }
            
            .model-name {
                color: #374151;
                font-weight: 600;
            }
            
            .model-value {
                color: var(--input-background-fill);
                font-family: monospace;
                font-size: 12px;
            }
        </style>
        """)
        
        # Header with DB status
        db_info = system.get_database_info()
        db_status = "Connected" if "count" in db_info else "Disconnected"
        db_count = db_info.get("count", 0) if "count" in db_info else 0
        
        gr.HTML(f"""
        <div class="fade-in" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding:25px; border-radius:12px; margin-bottom:20px">
            <h1 style="color:white; text-align:center; font-size:36px; margin:0;">
                🧠 Advanced Sentiment Analysis System
            </h1>
            <p style="color:#e2e8f0; text-align:center; margin-top:10px; font-size:16px;">
                Transformer Model + ChromaDB Vector Database + Real-time Analysis
            </p>
            <div style="background: rgba(255,255,255,0.1); padding: 10px; border-radius: 8px; margin-top: 15px;">
                <p style="color:white; text-align:center; margin:5px 0;">
                    🗄️ Vector Database: <span class="status-{db_status.lower()}">{db_status}</span> | 
                    📊 Embeddings: {db_count} | 
                    🔗 {system.config['chroma']['host']}:{system.config['chroma']['port']}
                </p>
            </div>
        </div>
        """)
        
        with gr.Row():
            # Input Column
            with gr.Column(scale=2):
                with gr.Group():
                    input_text = gr.Textbox(
                        lines=8,
                        label="📝 Enter Text for Analysis",
                        placeholder="Type or paste your text here...",
                        elem_classes="fade-in"
                    )
                    
                    with gr.Row():
                        analyze_btn = gr.Button(
                            "🔍 Analyze Sentiment",
                            variant="primary",
                            size="lg",
                            elem_classes=["hover-glow", "fade-in"]
                        )
                        clear_btn = gr.Button(
                            "🗑️ Clear",
                            variant="secondary",
                            size="lg"
                        )
                
                # Model Information Table (replacing Batch Analysis)
                with gr.Group():
                    gr.HTML("<h3 class='fade-in'>🤖 Model Information</h3>")
                    
                    # Get model information
                    model_info = system.get_model_info()
                    
                    # Create HTML table for model information
                    model_table_html = """
                    <table class="model-info-table">
                        <thead>
                            <tr>
                                <th>Component</th>
                                <th>Details</th>
                            </tr>
                        </thead>
                        <tbody>
                    """
                    
                    for item in model_info:
                        model_table_html += f"""
                        <tr>
                            <td style="color:var(--input-background-fill)" class="model-name">
                                <span class="model-icon"></span>
                                {item[0]}
                            </td>
                            <td style="color:var(--input-background-fill)" class="model-value">{item[1]}</td>
                        </tr>
                        """
                    
                    model_table_html += """
                        </tbody>
                    </table>
                    """
                    
                    model_info_display = gr.HTML(model_table_html)
            
            # Results Column
            with gr.Column(scale=2):
                # Sentiment Results
                with gr.Group():
                    gr.HTML("<h3 class='fade-in'>🎯 Analysis Results</h3>")
                    
                    # Sentiment Label Display (using HTML for styling)
                    label_display_html = gr.HTML(
                        value="""
                        <div class="sentiment-display" style="background: #f8f9fa; color: #6b7280;">
                            <div style="font-size: 32px; margin-bottom: 5px;">⏳</div>
                            <div>Waiting for analysis...</div>
                        </div>
                        """,
                        label=""
                    )
                    
                    # Confidence Score
                    score_output = gr.Number(
                        label="Confidence Score",
                        interactive=False,
                        precision=4
                    )
                    
                    # Confidence Bar
                    bar_html = gr.HTML("""
                    <div style="margin: 20px 0;">
                        <p style="margin-bottom: 8px; font-weight: bold;">Confidence Level:</p>
                        <div class="sentiment-bar" style="width:0%; background:#d1d5db; padding: 4px; border-radius: 8px;">
                            <span style="color: white; padding: 0 10px; font-weight: bold;">0%</span>
                        </div>
                    </div>
                    """)
                    
                    # Plotly Graph
                    plot_output = gr.Plot(label="Sentiment Distribution")
                
                # Similar Texts Section
                with gr.Group():
                    gr.HTML("<h3 class='fade-in'>🔍 Similar Texts from ChromaDB</h3>")
                    similar_output = gr.HTML("""
                    <div style="text-align: center; padding: 20px; background: #f8f9fa; border-radius: 8px;">
                        <div style="font-size: 32px; margin-bottom: 10px;">🔍</div>
                        <p style="color: #6b7280; margin: 0;">
                            Similar texts will appear here after analysis
                        </p>
                    </div>
                    """)
        
        # Footer
        gr.HTML("""
        <hr>
        <div style="text-align: center; color: gray; font-size: 14px; padding-top: 15px;">
            <p>🚀 Powered by: Transformers • ChromaDB • Gradio • Docker</p>
            <p>📊 Model: DistilBERT • Embeddings: Sentence Transformers • Vector Search: ChromaDB</p>
            <p>📈 Containerized Deployment • Real-time Semantic Similarity</p>
        </div>
        """)
        
        # Callback functions
        def analyze_callback(text):
            """Handle sentiment analysis"""
            result = system.analyze_sentiment(text)
            
            # Prepare confidence bar
            score_percent = int(result["score"] * 100)
            
            # Determine color, label, and styling based on sentiment
            if result["label"] == "POSITIVE":
                color = "#10b981"
                gradient_color = "#10b98122"
                emoji = "😊"
                sentiment_class = "positive"
                label_text = "POSITIVE"
            elif result["label"] == "NEGATIVE":
                color = "#ef4444"
                gradient_color = "#ef444422"
                emoji = "😠"
                sentiment_class = "negative"
                label_text = "NEGATIVE"
            else:
                color = "#6b7280"
                gradient_color = "#6b728022"
                emoji = "😐"
                sentiment_class = "neutral"
                label_text = result["label"]
            
            # Create styled label display with HTML
            label_display = f"""
            <div class="sentiment-display {sentiment_class}" style="
                background: linear-gradient(135deg, {gradient_color} 0%, {color}11 100%);
                border-left: 6px solid {color};
                color: {color};
                box-shadow: 0 4px 12px {color}33;
            ">
                <div style="font-size: 36px; margin-bottom: 8px;">{emoji}</div>
                <div style="font-size: 22px;">{label_text}</div>
                <div style="font-size: 14px; margin-top: 5px; opacity: 0.8;">Sentiment Analysis Result</div>
            </div>
            """
            
            bar_content = f"""
            <div style="margin: 20px 0;">
                <p style="margin-bottom: 8px; font-weight: bold; color: #374151;">Confidence Level:</p>
                <div class="sentiment-bar" style="width:{score_percent}%; background:{color}; padding: 4px; border-radius: 8px;">
                    <span style="color: white; padding: 0 10px; font-weight: bold;">{score_percent}%</span>
                </div>
            </div>
            """
            
            # Prepare plot
            positive_score = result["score"] if result["label"] == "POSITIVE" else (1 - result["score"])
            negative_score = 1 - positive_score
            
            fig = go.Figure(data=[
                go.Bar(
                    x=['Positive', 'Negative'],
                    y=[positive_score, negative_score],
                    marker_color=['#10b981', '#ef4444'],
                    text=[f'{positive_score:.2%}', f'{negative_score:.2%}'],
                    textposition='auto'
                )
            ])
            
            fig.update_layout(
                title="Sentiment Probability Distribution",
                yaxis=dict(range=[0, 1], title="Probability"),
                height=300,
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            
            # Prepare similar texts display
            similar_items = []
            for idx, item in enumerate(result.get("similar_texts", []), 1):
                sentiment = item['metadata'].get('label', 'Unknown')
                sentiment_class = "positive" if sentiment == "POSITIVE" else "negative" if sentiment == "NEGATIVE" else "neutral"
                similarity_percent = int(item['similarity_score'] * 100)
                sentiment_emoji = "😊" if sentiment == "POSITIVE" else "😠" if sentiment == "NEGATIVE" else "😐"
                sentiment_color = "#10b981" if sentiment == "POSITIVE" else "#ef4444" if sentiment == "NEGATIVE" else "#6b7280"
                
                similar_items.append(f"""
                <div class="similar-item">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                        <span style="
                            background: {sentiment_color}15; 
                            color: {sentiment_color};
                            padding: 4px 10px; 
                            border-radius: 4px; 
                            font-size: 12px;
                            font-weight: bold;
                        ">
                            #{idx} • Similarity: {similarity_percent}%
                        </span>
                        <span style="
                            color: {sentiment_color};
                            font-weight: bold;
                            padding: 4px 10px;
                            border-radius: 4px;
                            background: {sentiment_color}10;
                        ">
                            {sentiment_emoji} {sentiment}
                        </span>
                    </div>
                    <p style="margin: 0; font-size: 14px; color: #374151; line-height: 1.5;">
                        "{item['document'][:120]}..."
                    </p>
                </div>
                """)
            
            if similar_items:
                similar_content = f"""
                <div style="max-height: 300px; overflow-y: auto; padding-right: 10px;">
                    <p style="margin-bottom: 12px; color: #4b5563; font-size: 14px; font-weight: 500;">
                        🔍 Found {len(similar_items)} similar texts in database:
                    </p>
                    {''.join(similar_items)}
                </div>
                """
            else:
                similar_content = f"""
                <div style="text-align: center; padding: 30px; background: #f8f9fa; border-radius: 8px;">
                    <div style="font-size: 48px; margin-bottom: 10px; color: {color};">📝</div>
                    <p style="color: #6b7280; margin: 0; font-size: 14px; line-height: 1.5;">
                        No similar texts found yet.<br>
                        This text has been stored in the database for future similarity searches.
                    </p>
                </div>
                """
            
            return (
                label_display,
                result["score"],
                bar_content,
                fig,
                similar_content
            )
        
        def clear_callback():
            """Clear all inputs"""
            empty_fig = go.Figure()
            empty_fig.update_layout(
                title="Sentiment Probability Distribution",
                yaxis=dict(range=[0, 1], title="Probability"),
                height=300,
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            
            default_label = """
            <div class="sentiment-display" style="background: #f8f9fa; color: #6b7280;">
                <div style="font-size: 32px; margin-bottom: 5px;">⏳</div>
                <div>Waiting for analysis...</div>
            </div>
            """
            
            default_similar = """
            <div style="text-align: center; padding: 20px; background: #f8f9fa; border-radius: 8px;">
                <div style="font-size: 32px; margin-bottom: 10px;">🔍</div>
                <p style="color: #6b7280; margin: 0;">
                    Similar texts will appear here after analysis
                </p>
            </div>
            """
            
            return (
                "",  # input_text
                default_label,  # label_display_html
                0.0,  # score_output
                """
                <div style="margin: 20px 0;">
                    <p style="margin-bottom: 8px; font-weight: bold; color: #374151;">Confidence Level:</p>
                    <div class="sentiment-bar" style="width:0%; background:#d1d5db; padding: 4px; border-radius: 8px;">
                        <span style="color: white; padding: 0 10px; font-weight: bold;">0%</span>
                    </div>
                </div>
                """,  # bar_html
                empty_fig,  # plot_output
                default_similar  # similar_output
            )
        
        # Connect buttons to functions
        analyze_btn.click(
            fn=analyze_callback,
            inputs=input_text,
            outputs=[label_display_html, score_output, bar_html, plot_output, similar_output]
        )
        
        clear_btn.click(
            fn=clear_callback,
            outputs=[input_text, label_display_html, score_output, bar_html, plot_output, similar_output]
        )
        
        # Add example inputs
        gr.Examples(
            examples=[
                ["I absolutely love this product! It's amazing and works perfectly."],
                ["This is the worst experience I've ever had. Terrible service."],
                ["The movie was okay, nothing special but not bad either."],
                ["Very disappointed with the quality, it broke after one day."],
                ["Excellent service and high quality product, highly recommended!"]
            ],
            inputs=input_text,
            label="Try these examples:"
        )
    
    return demo

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Advanced Sentiment Analysis System with ChromaDB")
    parser.add_argument("--share", action="store_true", help="Create public link")
    parser.add_argument("--model", type=str, default="distilbert-base-uncased-finetuned-sst-2-english",
                       help="Sentiment model name")
    parser.add_argument("--embedding-model", type=str, default="all-MiniLM-L6-v2",
                       help="Embedding model name")
    parser.add_argument("--chroma-host", type=str, default="localhost",
                       help="ChromaDB host (default: localhost)")
    parser.add_argument("--chroma-port", type=int, default=8001,
                       help="ChromaDB port (default: 8001)")
    parser.add_argument("--no-chroma", action="store_true", 
                       help="Run without ChromaDB (not recommended)")
    
    args = parser.parse_args()
    
    # Check if ChromaDB should be disabled
    if args.no_chroma:
        logger.warning("Running without ChromaDB. Vector database features will be disabled.")
        # You would need to modify the system to use a fallback
    
    # Load configuration
    config = {
        'model': {
            'sentiment': args.model,
            'embedding': args.embedding_model
        },
        'chroma': {
            'host': args.chroma_host,
            'port': args.chroma_port,
            'collection_name': 'sentiment_embeddings'
        }
    }
    
    try:
        # Initialize system
        logger.info("Starting Advanced Sentiment Analysis System...")
        system = SentimentAnalysisSystem(config)
        
        # Build and launch interface
        interface = build_gradio_interface(system)
        
        logger.info("="*60)
        logger.info("Application Ready!")
        logger.info(f"🌐 Gradio UI: http://localhost:7860")
        logger.info(f"🗄️ ChromaDB: {args.chroma_host}:{args.chroma_port}")
        logger.info(f"🤖 Model: {args.model}")
        logger.info("="*60)
        
        interface.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=args.share,
            debug=False,
            show_error=True
        )
        
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        logger.info("\n" + "="*60)
        logger.info("TROUBLESHOOTING:")
        logger.info("="*60)
        logger.info("1. Start ChromaDB server:")
        logger.info("   docker run -d -p 8001:8000 chromadb/chroma:latest")
        logger.info("\n2. Check if ChromaDB is running:")
        logger.info("   docker ps")
        logger.info("\n3. Check ChromaDB logs:")
        logger.info("   docker logs <chroma-container-id>")
        logger.info("\n4. Test ChromaDB connection:")
        logger.info("   curl http://localhost:8001/api/v1/heartbeat")
        logger.info("="*60)
        sys.exit(1)

if __name__ == "__main__":
    main()