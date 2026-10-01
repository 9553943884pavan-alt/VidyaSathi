import os
import argparse
from datasets import Dataset

try:
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )
except ImportError:
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

def run_evaluation(dataset_path: str = "data/eval_dataset.json"):
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
    
    dataset = Dataset.from_dict(data)
    
    # 4. Run Evaluation
    print("Running RAGAS metrics: Faithfulness, Answer Relevancy, Context Precision, Context Recall...")
    # Note: RAGAS requires OPENAI_API_KEY to be set in the environment to use its LLM as an evaluator
    if not os.environ.get("OPENAI_API_KEY"):
        print("\n[WARNING] OPENAI_API_KEY is not set. RAGAS requires an LLM to evaluate the responses.")
        print("Set it via: export OPENAI_API_KEY='your-key'\n")
        return

    result = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
    )
    
    print("\n=== EVALUATION RESULTS ===")
    print(result)
    
    # In a full hackathon setup, you would save this to a CSV or generate a graph using matplotlib
    df = result.to_pandas()
    df.to_csv("evaluation_results.csv", index=False)
    print("Results saved to evaluation_results.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation on Vidya Sathi Agent")
    parser.add_argument("--dataset", type=str, help="Path to golden dataset JSON", default=None)
    args = parser.parse_args()
    
    run_evaluation(args.dataset)
