# Parse options using getopts
while getopts ":p:t:s:h:q:" opt; do
  case $opt in
    p) params="$OPTARG"
       ;;
    t) tasks="$OPTARG"
       ;;
    s) shots="$OPTARG"
       ;;
    h) hftoken="$OPTARG"
       ;;
    q) quantization="$OPTARG"
       ;;
  esac
done

# Set default values if not provided
if [[ -z "$shots" ]]; then
  shots=0  # Default to 0 if not specified
fi
if [[ $shots -lt 0 ]]; then
  echo "Error: Number of shots cannot be negative."
  exit 1
fi
if [[ $shots -gt 5 ]]; then
  echo "Warning: Number of shots is too high ($shots). Setting to 5."
  shots=5
fi

if [[ -z $params ]]; then
  echo "Defaulting to Llama-3.1-8B-Instruct"
  params="8B"
fi

if [[ $params != "8B" && $params != "70B" && $params != "405B" ]]; then
  echo "Error: Invalid model parameter '$params'. Must be '8B', '70B', or '405B'."
  exit 1
fi

if [[ -z $tasks ]]; then
  echo "Defaulting to single MMLU tasks High School Computer Science"
  tasks="single"
fi

if [[ $tasks != "single" && $tasks != "all" ]]; then
  echo "Error: Invalid tasks option '$tasks'. Must be 'single' for just High School CS or 'all' for all benchmarks."
  exit 1
fi

if [[ -z $hftoken ]]; then
  echo "Error: HF access token is required. Please pass it as an argument with -h."
  exit 1
fi

if [[ $quantization != "8bit" && $quantization != "4bit" && $quantization != "full" ]]; then
  echo "Error: Invalid quantization option '$quantization'. Must be '8bit', '4bit', or 'full' (for full precision)."
  exit 1
fi

# Load conda
source "$(conda info --base)/etc/profile.d/conda.sh"

# Activate environment
conda activate mmlu_test

# Confirm activate environment
echo "Active environment: $CONDA_DEFAULT_ENV"

# Run MMLU script
python3 mmlu.py --params "$params" \
                --tasks "$tasks" \
                --shots "$shots" \
                --hftoken "$hftoken" \
                --quantization "$quantization"

# Deactivate conda
conda deactivate

# Confirm activate environment
echo "Active environment: $CONDA_DEFAULT_ENV"