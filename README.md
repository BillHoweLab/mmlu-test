## UW AI Clinic -- LLM Benchmarking

This repo uses the DeepEval framework to evaluate Llama-3.1 models running in 4-bit quantization. It is intended to serve as a simple means for testing UW AI Clinic infrastructure.

To run, first make sure you have a conda environment called mmlu_test with the requirements installed (pip install -r requirements.txt).

Then, run: "bash run_benchmark.sh", passing in number of Llama-3 model parameters ("8B", "70B", or "405B" - defaults to "8B") with the --p flag; MMLU tasks (either "single" for just High School CS or "all" for the entire benchmark - defaults to "single") with the --t flag; the number of shots (between 0 and 5 - defaults to 0) with the --s flag; and your Hugging Face access token with the --h flag. Of these, only the HF access token lacks a default and is required to run the evaluation.