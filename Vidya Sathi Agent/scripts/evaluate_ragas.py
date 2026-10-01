import os
import argparse
from datasets import Dataset, Features, Value, Sequence
try:
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )
except ImportError as e:
    print(f"ImportError details: {e}")
    print("Please install ragas: pip install ragas datasets")
    exit(1)

# In a real scenario, this function would call the actual Vidya Sathi Agent
# Here we simulate the agent's RAG pipeline output for testing purposes
def simulate_agent_response(question: str):
    # This is a dummy implementation. 
    # To run this properly, you would import the agent from agent.student_agent and call it.
    dummy_responses = {
        "What is VSEPR theory?": {
            "answer": "VSEPR theory states that electron pairs around a central atom repel each other, determining the molecule's shape.",
            "contexts": ["Valence Shell Electron Pair Repulsion (VSEPR) theory provides a simple model for predicting the shapes of molecules. It is based on the idea that electron pairs in the valence shell of a central atom repel one another."]
        }
    }
    return dummy_responses.get(question, {"answer": "I don't know.", "contexts": []})

import json

def run_evaluation(args):
    dataset_path = args.dataset
    print("Initializing RAGAS evaluation...")
    
    # 1. Load the golden dataset of questions and expected answers
    try:
        with open(dataset_path, "r", encoding="utf-8") as f:
            eval_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Dataset not found at {dataset_path}")
        return
        
    questions = [item["question"] for item in eval_data]
    ground_truths = [item["ground_truth"] for item in eval_data]
    
    # 2. Collect Agent answers and retrieved contexts
    answers = []
    contexts = []
    
    print(f"Generating agent responses for {len(questions)} test questions...")
    for q in questions:
        result = simulate_agent_response(q)
        answers.append(result["answer"])
        contexts.append(result["contexts"])
        
    # 3. Format as a HuggingFace Dataset
    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    }
    
    features = Features({
        "question": Value("string"),
        "answer": Value("string"),
        "contexts": Sequence(Value("string")),
        "ground_truth": Value("string")
    })
    
    dataset = Dataset.from_dict(data, features=features)
    
    # 4. Run Evaluation
    print("Running RAGAS metrics: Faithfulness, Answer Relevancy, Context Precision, Context Recall...")
    
    evaluate_kwargs = {
        "dataset": dataset,
        "metrics": [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ]
    }
    
    if args.ollama:
        print(f"Using local Ollama models (LLM: {args.ollama_llm}, Embeddings: {args.ollama_embed})...")
        try:
            from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        except ImportError:
            print("Please install langchain-openai: pip install langchain-openai")
            return
            
        local_llm = ChatOpenAI(
            model=args.ollama_llm,
            base_url="http://localhost:11434/v1",
            api_key="ollama" # Dummy key
        )
        local_embeddings = OpenAIEmbeddings(
            model=args.ollama_embed,
            base_url="http://localhost:11434/v1",
            api_key="ollama"
        )
        evaluate_kwargs["llm"] = local_llm
        evaluate_kwargs["embeddings"] = local_embeddings
    elif args.hf:
        print(f"Using local Hugging Face models (LLM: {args.hf_llm}, Embeddings: {args.hf_embed})...")
        try:
            from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
            from transformers import BitsAndBytesConfig
            import torch
        except ImportError:
            print("Please install requirements: pip install langchain-huggingface transformers accelerate bitsandbytes torch")
            return
        
        # We load the model in 4-bit quantization so a 70B model can fit in VRAM
        # Use BitsAndBytesConfig for newer transformers versions
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4"
        )
        
        local_llm = HuggingFacePipeline.from_model_id(
            model_id=args.hf_llm,
            task="text-generation",
            pipeline_kwargs={"max_new_tokens": 512, "temperature": 0.1},
            model_kwargs={"quantization_config": bnb_config, "device_map": "auto"}
        )
        local_embeddings = HuggingFaceEmbeddings(
            model_name=args.hf_embed,
            model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
        )
        evaluate_kwargs["llm"] = local_llm
        evaluate_kwargs["embeddings"] = local_embeddings
    else:
        if not os.environ.get("OPENAI_API_KEY"):
            print("\n[WARNING] OPENAI_API_KEY is not set. RAGAS requires an LLM to evaluate the responses.")
            print("Set it via: export OPENAI_API_KEY='your-key' or run with --ollama or --hf\n")
            return

    result = evaluate(**evaluate_kwargs)
    
    print("\n=== EVALUATION RESULTS ===")
    print(result)
    
    # In a full hackathon setup, you would save this to a CSV or generate a graph using matplotlib
    df = result.to_pandas()
    df.to_csv("evaluation_results.csv", index=False)
    print("Results saved to evaluation_results.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation on Vidya Sathi Agent")
    parser.add_argument("--dataset", type=str, help="Path to golden dataset JSON", default="data/eval_dataset.json")
    
    parser.add_argument("--ollama", action="store_true", help="Use local Ollama instance for evaluation")
    parser.add_argument("--ollama-llm", type=str, default="llama3.1:70b", help="Ollama LLM model name")
    parser.add_argument("--ollama-embed", type=str, default="nomic-embed-text", help="Ollama Embeddings model name")
    
    parser.add_argument("--hf", action="store_true", help="Use Hugging Face models via transformers")
    parser.add_argument("--hf-llm", type=str, default="meta-llama/Meta-Llama-3.1-70B-Instruct", help="HF LLM model ID")
    parser.add_argument("--hf-embed", type=str, default="BAAI/bge-small-en-v1.5", help="HF Embeddings model ID")
    
    args = parser.parse_args()
    
    run_evaluation(args)
