#!/usr/bin/env python3
"""
Interactive Chatbot for Native Ads Detection
Agentic orchestrator with natural language understanding
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import logging

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from agents.orchestrator_agent import OrchestratorAgent

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_welcome():
    """Print welcome message."""
    print("\n" + "="*80)
    print("🤖 NATIVE ADS DETECTION ASSISTANT")
    print("="*80)
    print("\nSaya bisa membantu Anda:")
    print("  1. 🌐 Scraping - Ambil konten dari URL")
    print("  2. 🔧 Preprocessing - Bersihkan & analisis teks")
    print("  3. 🔍 Retrieval - Cari contoh di database")
    print("  4. 🤖 Classification - Klasifikasi native ads/berita murni")
    print("  5. 📝 Explanation - Jelaskan hasil klasifikasi")
    print("  6. 🚀 Full Pipeline - Analisis lengkap")
    print("\nContoh:")
    print("  - \"ambil konten dari https://example.com\"")
    print("  - \"klasifikasikan berita ini\"")
    print("  - \"jelaskan kenapa ini native ads\"")
    print("  - \"analisis lengkap url https://example.com\"")
    print("\nCommands:")
    print("  /help   - Tampilkan bantuan")
    print("  /clear  - Hapus history percakapan")
    print("  /memory - Lihat history percakapan")
    print("  /exit   - Keluar")
    print("="*80 + "\n")


def print_response(response: dict):
    """Print agent response in formatted way."""
    status = response.get('status', 'unknown')
    message = response.get('message', '')
    data = response.get('data', {})
    
    # Print message
    if message:
        print(f"\n{message}\n")
    
    # Print data if available
    if data and status == 'success':
        if 'title' in data:
            print(f"   📄 Title: {data['title']}")
        
        if 'word_count' in data:
            print(f"   📊 Word count: {data['word_count']}")
        
        if 'features' in data and data['features']:
            features = data['features']
            print(f"   📈 Features:")
            if 'word_count' in features:
                print(f"      - Words: {features['word_count']}")
            if 'sentence_count' in features:
                print(f"      - Sentences: {features['sentence_count']}")
            if 'lexical_diversity' in features:
                print(f"      - Lexical diversity: {features['lexical_diversity']:.2f}")
        
        if 'summary' in data and data['summary']:
            print(f"   📝 Summary: {data['summary']}")
        
        if 'label' in data:
            print(f"   🏷️  Label: {data['label']}")
            if 'confidence' in data:
                print(f"   💯 Confidence: {data['confidence']:.2%}")
            if 'reasoning' in data:
                print(f"   💭 Reasoning: {data['reasoning']}")
        
        if 'explanation' in data:
            print(f"\n{data['explanation']}")
        
        if 'results' in data:
            results = data['results']
            for i, result in enumerate(results, 1):
                print(f"\n   [{i}] {result.get('label', 'unknown').upper()}")
                print(f"       {result.get('content', '')}")
                if 'similarity' in result:
                    print(f"       Similarity: {result['similarity']:.2f}")
    
    print()


def save_conversation(orchestrator: OrchestratorAgent, output_dir: str = "../results/conversations"):
    """Save conversation to file."""
    try:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = output_path / f"conversation_{timestamp}.txt"
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("CONVERSATION HISTORY\n")
            f.write("="*80 + "\n\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n\n")
            
            memory = orchestrator.get_memory()
            for msg in memory:
                role = msg['role'].upper()
                content = msg['content']
                f.write(f"{role}: {content}\n\n")
            
            f.write("="*80 + "\n")
        
        print(f"💾 Conversation saved to: {filename}")
        
    except Exception as e:
        logger.error(f"Failed to save conversation: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Interactive Native Ads Detection Chatbot'
    )
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo',
                       help='LLM model name (for classification/explanation)')
    parser.add_argument('--provider', type=str, default='openai',
                       choices=['openai', 'openrouter', 'local'],
                       help='LLM provider (for classification/explanation)')
    parser.add_argument('--api-key', type=str, 
                       help='API key (optional - only for classification/explanation)')
    parser.add_argument('--no-vectorstore', action='store_true',
                       help='Disable vector store (retrieval will not work)')
    parser.add_argument('--use-instructor', action='store_true',
                       help='Use Instructor-based classification with Gemma v7')
    parser.add_argument('--model-path', type=str, default='../models/gemma-native-ads-v7_merged_16bit',
                       help='Path to the model (for local/instructor mode)')
    
    args = parser.parse_args()
    
    # Auto-set local provider if using instructor and no provider specified
    if args.use_instructor and args.provider == 'openai' and not os.getenv("OPENAI_API_KEY"):
        print("💡 Instructor mode requested but no OpenAI key found. Defaulting to local provider...")
        args.provider = 'local'
    
    # Note: Intent detection is always local (no API needed)
    logger.info("Using local keyword-based intent detection (no API required)")
    
    # Initialize vectorstore if not disabled
    vectorstore = None
    if not args.no_vectorstore:
        try:
            from langchain_chroma import Chroma
            from langchain_openai import OpenAIEmbeddings
            
            vectorstore = Chroma(
                persist_directory="../vector_stores/native_ads_db",
                embedding_function=OpenAIEmbeddings(api_key=args.api_key)
            )
            logger.info("Vector store loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load vector store: {e}")
            logger.warning("Retrieval features will be disabled")
    
    # Initialize orchestrator
    print("\n🔄 Initializing orchestrator...")
    orchestrator = OrchestratorAgent(
        model_name=args.model,
        provider=args.provider,
        api_key=args.api_key,
        vectorstore=vectorstore,
        use_instructor=args.use_instructor,
        model_path=args.model_path if args.use_instructor or args.provider == 'local' else None
    )
    
    # Print welcome message
    print_welcome()
    
    # Main interaction loop
    try:
        while True:
            # Get user input
            try:
                user_input = input("You: ").strip()
            except EOFError:
                break
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.startswith('/'):
                command = user_input[1:].lower()
                
                if command == 'exit' or command == 'quit':
                    print("\n👋 Terima kasih! Sampai jumpa!")
                    save_conversation(orchestrator)
                    break
                
                elif command == 'help':
                    print_welcome()
                    continue
                
                elif command == 'clear':
                    orchestrator.clear_memory()
                    print("\n✅ History percakapan dihapus.\n")
                    continue
                
                elif command == 'memory':
                    memory = orchestrator.get_memory()
                    if not memory:
                        print("\n📭 History kosong.\n")
                    else:
                        print("\n📜 History percakapan:")
                        print("-"*80)
                        for msg in memory:
                            role = "You" if msg['role'] == 'user' else "Bot"
                            print(f"{role}: {msg['content'][:100]}...")
                        print("-"*80 + "\n")
                    continue
                
                else:
                    print(f"\n❌ Command tidak dikenali: /{command}\n")
                    continue
            
            # Process user input
            print()  # Newline for better formatting
            response = orchestrator.process(user_input)
            
            # Print response
            print_response(response)
    
    except KeyboardInterrupt:
        print("\n\n👋 Terima kasih! Sampai jumpa!")
        save_conversation(orchestrator)
    
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"\n❌ Error: {e}\n")


if __name__ == "__main__":
    main()
