from langsmith import Client as LangSmithClient
from langsmith.evaluation import evaluate
from dotenv import load_dotenv
from src.service_wrapper import process_chat_request

load_dotenv()

ls_client = LangSmithClient()

def evaluate_agent_pipeline(inputs: dict) -> dict:
    user_query = inputs.get("query")
    
    response_text = process_chat_request(user_id="langsmith_eval_runner", user_message=user_query)
    
    return {"output": response_text}

def accuracy_intent_scorer(run, example) -> dict:
    agent_output = run.outputs.get("output", "")
    expected_node = example.outputs.get("expected_agent", "")
    
    is_correct = 0
    if expected_node == "support_agent":
        if "hello" in agent_output or "hi" in agent_output or "assist" in agent_output or "support" in agent_output:
            is_correct = 1
            
    elif expected_node == "retrieval_agent":
        if "flicker" in agent_output or "troubleshooting" in agent_output or "ticket" in agent_output:
            is_correct = 1
            
    elif expected_node == "escalation_agent":
        if "escalat" in agent_output or "refund" in agent_output or "supervisor" in agent_output or "live support" in agent_output:
            is_correct = 1
        
    return {"key": "correct_routing_intent", "score": is_correct}

def run_automated_evaluation():
    dataset_name = "NovaTech-Capstone-Validation-Suite"
    
    test_dataset = [
        {"query": "hi", "expected_agent": "support_agent"},
        {"query": "want to know about screen flicker troubleshooting", "expected_agent": "retrieval_agent"},
        {"query": "I am furious, give me a human supervisor refund right now!", "expected_agent": "escalation_agent"}
    ]
    
    if not ls_client.has_dataset(dataset_name=dataset_name):        
        dataset = ls_client.create_dataset(
            dataset_name=dataset_name,
            description="End-to-End verification rules for NovaTech multi-agent orchestration routing matrix."
        )
        
        for case in test_dataset:
            ls_client.create_example(
                inputs={"query": case["query"]},
                outputs={"expected_agent": case["expected_agent"]},
                dataset_id=dataset.id
            )
    else:
        print(f"Found existing tracking dataset '{dataset_name}' in cloud workspace dashboard.")

    print(f"\nLaunching automated evaluation threads across target dataset...")
    
    results = evaluate(
        evaluate_agent_pipeline,
        data=dataset_name,
        evaluators=[accuracy_intent_scorer],
        experiment_prefix="MultiAgent-RAG-Run"
    )

    total_score = 0
    count = 0
    
    for execution in results:
        feedback = execution.get("evaluation_results", {}).get("results", [])
        score = feedback[0].get("score", 0) if feedback and isinstance(feedback[0], dict) else (feedback[0].score if feedback else 0)          
        total_score += score
        count += 1

    final_percentage = (total_score / count) * 100 if count > 0 else 0
    print(f"System Accuracy KPI: {final_percentage:.1f}% ({total_score}/{count} Passed)")    
    print(f"\nEvaluation loop completed successfully!")


if __name__ == "__main__":
    run_automated_evaluation()