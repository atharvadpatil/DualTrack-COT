Here is the clean version of the README for the **DualTrack-COT** project.

# DualTrack-COT: Budget-Aware Stepwise Guidance for Small LMs

## 📖 Introduction
Large Language Models (LLMs) have demonstrated success in reasoning tasks via Chain-of-Thought (CoT) prompting, but smaller models (7-8B parameters) often struggle with multi-step reasoning, especially under tight compute and token budgets. Existing solutions like Self-Consistency or Tree-of-Thoughts (ToT) offer improvements but frequently incur high token costs or lack fine-grained, step-level intervention.

**Dual-Track CoT** is a two-agent test-time reasoning framework designed to address these limitations. It enables Small Language Models (SLMs) to reason reliably by using a **Decomposer** to break problems into verifiable steps and an **Evaluator** to judge and guide each step within a strict token budget.

---

## 🚀 Key Features
* **Two-Agent Framework**:
    * **Decomposer**: Breaks complex problems into short, verifiable reasoning steps with brief rationales.
    * **Evaluator**: Scores each step based on logical validity, relevance, consistency, complexity, efficiency, and token cost.
* **Stepwise Revision**: If a step scores low (criterion $\le 1$), the Evaluator provides a concise hint to guide a single revision before the step is accepted or rejected.
* **Budget-Aware Search**: The system maintains a small beam of partial solutions ($k=2-4$) and enforces a strict per-example token budget using a "token accountant".
* **Semantic Rejection Cache**: Records failed step patterns to prevent the model from revisiting invalid or redundant ideas.

---

## 🏗️ Architecture

The Dual-Track CoT architecture operates as a loop where the Decomposer generates candidate steps, the Evaluator scores them, and a Rejection Cache filters out known failures.



### Workflow
1.  **Generation**: The Decomposer emits a short step and rationale.
2.  **Caching**: The system checks the **Rejection Cache**; if the step matches a failed pattern, it prompts a re-generation.
3.  **Evaluation**: The Evaluator scores the step on a 0-3 rubric.
4.  **Revision**: If the score is low, the Evaluator issues a hint for revision.
5.  **Termination**: The loop ends when a final answer is generated or token/step caps are reached.

---

## 🛠️ Tools & Tech Stack
This project leverages open-source deep learning frameworks and libraries to ensure reproducibility and efficiency.

* **Deep Learning Framework**: [PyTorch](https://pytorch.org/) for dynamic computation and GPU acceleration.
* **Decomposer Model**: [Llama 3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) (Instruction-tuned) for step-wise problem breakdown.
* **Evaluator Model**: [Llama 3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) for scoring and logical verification.
* **Fine-Tuning**: [trl](https://github.com/huggingface/trl) for LoRA/QLORA training.
* **Deployment**: Hugging Face **Transformers** and **Accelerate** for multi-GPU inference and tokenization.
* **Environment**: Google Colab Pro+ (NVIDIA A100 GPUs).

---

## 📊 Datasets

### Training (Evaluator Fine-Tuning)
* **PRM800K**: A subset of this process-supervised dataset is used to fine-tune the Evaluator, providing step-level supervision signals aligned with logical validity and efficiency.

### Evaluation Benchmarks
The framework is evaluated on its ability to solve math problems at fixed token budgets (1.0k/1.5k/2.0k tokens).
* **GSM8K**: High-school level math problems testing multi-hop reasoning.

---

## 🔬 Research Goals
This project aims to answer four primary questions:
1.  **Effectiveness at Equal Cost**: Does Dual-Track CoT outperform baselines (CoT, Self-Consistency) when restricted to the same token budget?
2.  **Value of Stepwise Guidance**: Does judging every intermediate step outperform post-hoc judgment (evaluating only final answers)?
3.  **Efficiency**: How much do the rejection cache and token-aware ranking contribute to stability?
4.  **Process Supervision**: Does fine-tuning the Evaluator on process-labeled data improve accuracy compared to a prompted Evaluator?

---

## 📅 Roadmap (8-Week Timeline)
* **Weeks 1-2**: Setup, data preparation (GSM8K, SVAMP), and implementation of token-accounting utilities.
* **Weeks 3-4**: Development of the core Dual-Track engine (Decomposer + Evaluator loop).
* **Weeks 5-6**: Integration of the semantic rejection cache and fine-tuning the Evaluator with LoRA.
* **Weeks 7-8**: Full evaluation, ablation studies, and final reporting.

---

## 👥 Authors
* **Atharva Patil** (adpatil@umass.edu)
* **Sricharan Ramesh** (sricharanram@umass.edu)
* **Sagnik Chatterjee** (sagnikchatte@umass.edu)