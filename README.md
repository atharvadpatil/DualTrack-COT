# Dual-Track Chain-of-Thought (Dual-Track CoT)

A collaborative reasoning system that uses two specialized language models working together to solve mathematical word problems step-by-step. The system features a **Decomposer** model that generates solution steps and an **Evaluator** model that assesses the quality of each step, creating an iterative refinement process.

### Key Workflow

1. **Step Generation**: The Decomposer generates a candidate step based on the problem and previously accepted steps
2. **Step Evaluation**: The Evaluator assesses the step using a detailed scoring rubric
3. **Decision Logic**:
   - If score ≥ threshold (typically 2.0): Accept step and continue
   - If score < threshold: Provide feedback and retry (up to `max_retries`)
4. **Termination**: Stop when a `FINAL_ANSWER` is reached or `max_steps` is exceeded

## 🔧 Components

### 1. Decomposer (Step Generator)

- **Role**: Generates the next logical step in solving the problem
- **Input Format**:

  ```
  Problem: <problem text>

  Steps completed so far:
  STEP 1: <step 1>
  STEP 2: <step 2>
  ...
  ```

- **Output Format**:
  - `STEP: <step description>` for intermediate steps
  - `FINAL_ANSWER: <answer>` when the problem is solved
- **Features**:
  - Incorporates feedback from the Evaluator when retrying
  - Avoids rejected steps using rejection cache
  - Can be fine-tuned or used with prompting

### 2. Evaluator (Step Assessor)

- **Role**: Evaluates the quality and correctness of generated steps
- **Scoring Rubric**:
  - **Score 0**: Irrelevant or nonsensical step
  - **Score 1**: Weak/incorrect reasoning
  - **Score 2**: Partially correct but needs improvement
  - **Score 3**: Good step with minor issues
  - **Score 4**: Perfect step, clear and correct
- **Output**: Provides both a numerical score and textual feedback for improvement

### 3. TokenBudget

A utility class that tracks token usage during problem-solving to enforce computational constraints.

```python
class TokenBudget:
    def __init__(self, max_tokens: int)
    def consume(self, tokens: int) -> bool  # Returns False if budget exhausted
    @property
    def exhausted(self) -> bool
    @property
    def remaining(self) -> int
```

**Features**:

- Monitors token consumption for both Decomposer and Evaluator
- Prevents generation when budget is exhausted
- Supports strict token constraints in resource-limited scenarios

### 4. RejectionCache

Prevents the generation of repetitive or previously rejected steps using a mathematical fingerprint-based approach.

```python
class RejectionCache:
    def __init__(self)
    def add(text: str)  # Add rejected step to cache
    def is_duplicate(text: str) -> bool  # Check if step matches a rejected one
```

**Features**:

- Uses `_normalize_for_repetition()` to extract a mathematical fingerprint
- Normalization extracts only numbers and operators (e.g., `"9 * 2 = 18"`), ignoring wording
- Exact string matching after normalization (not semantic similarity)
- Prevents exact computational duplicates while allowing similar semantic content with different math
- Repetition guard: Detects if a step repeats a previously _accepted_ step using the same normalization

**Note on EmbeddingRejectionCache**: An embedding-based cache (`EmbeddingRejectionCache`) was experimented with but found to be too aggressive—it would reject steps that are semantically similar to rejected steps, even if the new step is correct. For example, if a partially wrong step was rejected, the embedding cache might also reject a correct step that happens to be semantically similar, since embeddings can't distinguish between "partially wrong" and "correct but similar." The string-based `RejectionCache` avoids this issue by focusing on exact mathematical computations rather than semantic similarity.

### 5. Collaborative Solver

The main orchestration function that coordinates the Decomposer and Evaluator:

```python
def solve_problem_collaboratively(
    problem: str,
    decomp_model,
    decomp_tokenizer,
    eval_model,
    eval_tokenizer,
    max_steps: int = 10,
    score_threshold: float = 2.0,
    max_retries: int = 2,
    max_token_budget: int = None,  # Optional
) -> Dict
```

**Return Value**:

```python
{
    "problem": str,
    "final_steps": List[str],  # Accepted steps
    "all_evaluations": List[Dict],  # All evaluation results
    "num_steps": int,
    "final_answer": str | None,
}
```

## 🧪 Experiment Variations

The project includes several experiment configurations, progressively adding features:

### 1. Baseline Simple CoT

- **File**: `notebooks/experiments/Baseline Simple CoT.ipynb`
- Single model with standard Chain-of-Thought prompting
- Baseline for comparison

### 2. Dual-Track-CoT Only Prompting

- **File**: `notebooks/experiments/Dual-Track-CoT Only Prompting.ipynb`
- Two-model collaboration (Decomposer + Evaluator)
- Both models use base Llama 3.1 8B with prompting only (no fine-tuning)

### 3. Dual-Track-CoT Fine-tuned Only Decomposer

- **File**: `notebooks/experiments/Dual-Track-CoT Fine-tuned Only Decomposer.ipynb`
- Decomposer is fine-tuned, Evaluator uses base model with prompting

### 4. Dual-Track-CoT Fine-tuned Decomposer and Evaluator

- **File**: `notebooks/experiments/Dual-Track-CoT Fine-tuned Decomposer and Evaluator.ipynb`
- Both Decomposer and Evaluator are fine-tuned
- Core dual-track implementation

### 5. Dual-Track-CoT with Token Budget

- **File**: `notebooks/experiments/Dual-Track-CoT Fine-tuned Decomposer and Evaluator with Token Budget.ipynb`
- Adds `TokenBudget` class to enforce token limits
- Useful for resource-constrained scenarios

### 6. Dual-Track-CoT with Token Budget and Rejection Cache

- **File**: `notebooks/experiments/Dual-Track-CoT Fine-tuned Decomposer and Evaluator with Token Budget and Rejection Cache.ipynb`
- **Recommended**: Uses `RejectionCache` (string-based) to prevent duplicate steps
- Mathematical fingerprint-based duplicate detection
- Most advanced production configuration

### 7. Dual-Track-CoT with Token Budget and Embeddings Rejection Cache

- **File**: `notebooks/experiments/Dual-Track-CoT Fine-tuned Decomposer and Evaluator with Token Budget and Embeddings Rejection Cache.ipynb`
- **Experimental/Omitted**: Uses `EmbeddingRejectionCache` with semantic similarity
- This approach was found to be too aggressive, rejecting correct steps that are semantically similar to rejected steps
- Included for completeness but not recommended for production use

### 8. Dual-Track-CoT Strict Token Constraint

- **File**: `notebooks/experiments/Dual-Track-CoT Fine-tuned Decomposer and Evaluator Strict Token Constraint.ipynb`
- Enforces strict token limits with early stopping

## 🚀 Setup

### Prerequisites

- Python 3.8+
- Jupyter Notebook or JupyterLab
- CUDA-capable GPU (recommended for model inference)
- Hugging Face account and token (for accessing models and datasets)

### Quick Start

Since the experiments are self-contained Jupyter notebooks, minimal setup is required:

1. Clone the repository:

```bash
git clone <repository-url>
cd DualTrack-COT
```

2. Set up Hugging Face authentication:

```python
import os
os.environ["HF_TOKEN"] = "your_huggingface_token_here"
```

Or use the Hugging Face CLI:

```bash
huggingface-cli login
```

3. **Prepare the Dataset** (Required for fine-tuning experiments):

Run the dataset preparation script to create the stepwise training data:

```bash
python src/dataset-preparation/gsm8k_stepwise_dataset.py
```

The script will:

- Load the GSM8K training dataset from Hugging Face
- Transform it into stepwise format (see [Dataset Preparation](#-dataset-preparation) section for details)
- Save the result to `datasets/gsm8k_stepwise.csv`

**Required dependencies**: Install `datasets`, `pandas`, and `tqdm` if not already available:

```bash
pip install datasets pandas tqdm
```

**Note**: Each experiment notebook installs its own dependencies at runtime, so no global installation is needed. The notebooks will install packages like `transformers`, `accelerate`, `bitsandbytes`, `peft`, and `torch` as needed.

### Model Setup

The experiments use Llama 3.1 8B as the base model with optional fine-tuned LoRA adapters:

- **Base Model**: `unsloth/meta-Llama-3.1-8B-Instruct`
- **Fine-tuned Adapters**: Paths to saved LoRA adapters (configure in notebook)

Models are loaded with 4-bit quantization for memory efficiency:

```python
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)
```

## 📊 Running Experiments

### Basic Usage

1. **Open the desired experiment notebook** (e.g., `Dual-Track-CoT Fine-tuned Decomposer and Evaluator with Token Budget and Embeddings Rejection Cache.ipynb`)

2. **Configure model paths**:

   - Update `decomp_adapter_path` to your Decomposer fine-tuned adapter path
   - Update `eval_adapter_path` to your Evaluator fine-tuned adapter path

3. **Run the notebook cells in order**:
   - Cell 1: Install dependencies and authenticate
   - Cell 2-3: Load embedding model (if using rejection cache)
   - Cell 4: Define utility classes (`TokenBudget`, `RejectionCache`)
   - Cell 5: Load models (Decomposer and Evaluator)
   - Cell 6: Define `generate_next_step` function
   - Cell 7: Define `evaluate_step` function
   - Cell 8: Define `solve_problem_collaboratively` function
   - Cell 9+: Testing and evaluation functions

### Testing on GSM8K

Each experiment notebook includes a `test_on_gsm8k` function:

```python
results = test_on_gsm8k(
    decomp_model,
    decomp_tokenizer,
    eval_model,
    eval_tokenizer,
    num_samples=50,
    max_steps=10,
    score_threshold=2.0,
    max_retries=2,
    max_token_budget=1600  # Optional
)
```

### Configuration Parameters

Key parameters you can adjust:

- `max_steps`: Maximum number of steps to generate (default: 10)
- `score_threshold`: Minimum score for step acceptance (default: 2.0)
- `max_retries`: Maximum retries per step when score is low (default: 2)
- `max_token_budget`: Token budget limit (optional, default: None)

### Example Output

```
================================================================================
PROBLEM: Janet has 3 apples. She buys 5 more. How many does she have?
================================================================================

--- Generating Step 1 ---
STEP: Janet starts with 3 apples.

--- Evaluating Step 1 ---
Scores: {'Relevance': 4, 'Correctness': 4, 'Clarity': 4, 'Raw_Score': 4.0}
Feedback: Excellent step. Clear and relevant.
✓ Step accepted!

--- Generating Step 2 ---
STEP: She buys 5 more apples, so she now has 3 + 5 = 8 apples.
FINAL_ANSWER: 8

✓ Final answer reached: 8
```

## 📁 Dataset Preparation

### Training (Evaluator Fine-Tuning)

- **PRM800K**: A subset of this process-supervised dataset is used to fine-tune the Evaluator, providing step-level supervision signals aligned with logical validity and efficiency.

### Custom Stepwise Dataset (Decomposer Fine-Tuning)

The project includes a dataset preparation script for creating stepwise training data:

**Script**: `src/dataset-preparation/gsm8k_stepwise_dataset.py`

This script transforms the standard GSM8K dataset into a custom stepwise format suitable for training the Decomposer model on incremental reasoning.

#### Process

1. **Load GSM8K Training Data**: Loads the original GSM8K training split (7,473 problems)

2. **Extract Rationales and Steps**:

   - Extracts the rationale portion (everything before the `####` marker that contains the final answer)
   - Splits rationales into individual steps using:
     - **Line-based splitting**: If the rationale contains multiple lines, each line becomes a step
     - **Sentence-based splitting**: If the rationale is a single paragraph, split on sentence boundaries (`.`, `!`, `?`)

3. **Create Progressive Training Examples**:
   For each problem, creates multiple training examples that progressively build up the solution:

   - **Example 1**: Problem only → First step
   - **Example 2**: Problem + Step 1 → Second step
   - **Example 3**: Problem + Steps 1-2 → Third step
   - ...
   - **Final Example**: Problem + All steps → `FINAL_ANSWER: <answer>`

4. **Format**:

   - **Input Format** (`Problem` column):

     ```
     Problem: <question text>

     Steps completed so far:
     STEP 1: <step 1>
     STEP 2: <step 2>
     ...
     ```

   - **Target Format** (`Next Step` column):
     - Intermediate steps: `STEP: <step description>`
     - Final answer: `FINAL_ANSWER: <numeric answer>`

#### Output

- **File**: `datasets/gsm8k_stepwise.csv`
- **Rows**: ~34,193 training examples (from 7,473 problems)
- **Columns**: `Problem`, `Next Step`

#### Benefits

This custom dataset format enables the Decomposer to learn:

- **Incremental reasoning**: How to generate the next logical step given previous steps
- **Proper formatting**: To output steps in the expected `STEP:` format and final answers as `FINAL_ANSWER:`
- **Context awareness**: To consider the problem statement and solution history when generating new steps

The stepwise format is used for fine-tuning the Decomposer model using Supervised Fine-Tuning (SFT), enabling it to generate coherent, sequential solution steps rather than complete answers at once.

### Evaluation Benchmarks

The framework is evaluated on its ability to solve math problems at fixed token budgets (1.0k/1.5k/2.0k tokens).

- **GSM8K**: High-school level math problems testing multi-hop reasoning.

## 🔍 Key Features

### Retry Mechanism

When a step scores below the threshold, the Evaluator's feedback is incorporated into the next generation attempt, allowing the Decomposer to improve.

### Rejection Cache

Prevents the model from generating steps that repeat previously rejected steps using mathematical fingerprinting. The cache extracts numbers and operators from steps, allowing it to detect exact computational duplicates while ignoring wording differences.

### Token Budget Management

Enforces computational constraints, making the system suitable for resource-limited environments.

## 🛠️ Tools & Tech Stack

This project leverages open-source deep learning frameworks and libraries to ensure reproducibility and efficiency.

- **Deep Learning Framework**: [PyTorch](https://pytorch.org/) for dynamic computation and GPU acceleration.
- **Decomposer Model**: [Llama 3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) (Instruction-tuned) for step-wise problem breakdown.
- **Evaluator Model**: [Llama 3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) for scoring and logical verification.
- **Fine-Tuning**: [trl](https://github.com/huggingface/trl) for LoRA/QLORA training.
- **Deployment**: Hugging Face **Transformers** and **Accelerate** for multi-GPU inference and tokenization.
- **Environment**: Google Colab Pro+ (NVIDIA A100 GPUs).

---

## 👥 Authors

- **Atharva Patil** (adpatil@umass.edu)
- **Sricharan Ramesh** (sricharanram@umass.edu)
- **Sagnik Chatterjee** (sagnikchatte@umass.edu)
