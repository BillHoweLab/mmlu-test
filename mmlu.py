import argparse
import json
import time

from huggingface_hub import login
from deepeval.benchmarks import MMLU
from deepeval.benchmarks.mmlu.task import MMLUTask

from llm import CustomLlama3

def main():
    
    # Parse arguments   
    parser = argparse.ArgumentParser()
    parser.add_argument('--params', type=str, default='8B')
    parser.add_argument('--shots', type=int, default=0)
    parser.add_argument('--tasks', type=str, default='single')
    parser.add_argument('--hftoken', type=str, default='')
    args = parser.parse_args()

    # Check for HF token
    if not args.hftoken:
        raise ValueError("Hugging Face token is required. Please set the --hftoken argument.")

    # Login to Hugging Face
    login(token=args.hftoken)
    print("Logged in to Hugging Face successfully.")

    # Initialize model
    model = CustomLlama3(args.params, args.hftoken)

    # Test the model's generation function
    test_output = model.generate("What is the answer to 2 + 2?")
    print(f'Model Output: {test_output}')

    # Get MMLU benchmark
    if args.tasks.lower() == 'single':
        benchmark = MMLU(
            tasks=[MMLUTask.HIGH_SCHOOL_COMPUTER_SCIENCE],
            n_shots=args.shots
        )
    elif args.tasks.lower() == 'all':
        benchmark = MMLU(
            n_shots=args.shots
        )
    else:
        raise ValueError("Invalid tasks argument. Use 'single' or 'all'.")

    # Evaluate model
    start_time = time.time()
    print("Starting evaluation...")
    benchmark.evaluate(model=model)
    end_time = time.time()
    time_taken = end_time - start_time
    print(f'Overall Score: {benchmark.overall_score}')
    print(f'Evaluation completed in {time_taken:.2f} seconds.')

    score_df = benchmark.task_scores

    json_score = {
        "model": f'Llama-3.1 {args.params}',
        "params": args.params,
        "shots": args.shots,
        "tasks": args.tasks,
        "time_taken": end_time - start_time,
        "overall_score": benchmark.overall_score
    }

    # Save results
    with open(f'results/{args.params}_{args.shots}-shot_mmlu-{args.tasks}.json', 'w') as f:
        json.dump(json_score, f)

    score_df.to_csv(f'results/{args.params}_{args.shots}-shot_mmlu-{args.tasks}.csv', index=False)

if __name__ == '__main__':
    main()