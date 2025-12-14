import re
import pandas as pd
from pathlib import Path
from typing import Generator
from datasets import load_dataset
from tqdm import tqdm

from config import OUTPUT_DIR_NAME, OUTPUT_FILENAME


def extract_final_answer(answer_text: str) -> str | None:
    """Extracts the final numerical answer following ####."""
    match = re.search(r"####\s*(.+)", answer_text)
    return match.group(1).strip() if match else None


def extract_rationale(answer_text: str) -> str:
    """Extracts the reasoning text before the final answer."""
    return answer_text.split("####")[0].strip()


def split_rationale_into_steps(rationale: str) -> list[str]:
    """
    Splits the rationale into discrete steps.
    Prioritizes line breaks, falls back to sentence splitting.
    """
    # 1. Try line-based splitting
    lines = [ln.strip() for ln in rationale.split("\n") if ln.strip()]
    if len(lines) > 1:
        return lines

    # 2. Fallback: sentence-based splitting
    # Replace newlines with spaces to handle wrapped text
    text = rationale.replace("\n", " ").strip()
    # Split on punctuation (.!?) followed by whitespace
    sentences = re.split(r"(?<=[\.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


def format_history(steps: list[str]) -> str:
    """Formats a list of previous steps into a numbered history string."""
    if not steps:
        return ""
    return "\n".join(f"STEP {i}: {step}" for i, step in enumerate(steps, start=1))


def process_example(
    question: str, full_answer: str
) -> Generator[dict[str, str], None, None]:
    """
    Yields training rows (Problem, Next Step) for a single GSM8K example.
    """
    final_answer = extract_final_answer(full_answer)
    if final_answer is None:
        return

    rationale = extract_rationale(full_answer)
    steps = split_rationale_into_steps(rationale)

    if not steps:
        return

    history_steps: list[str] = []

    # Yield one row per reasoning step
    for step in steps:
        history_str = format_history(history_steps)

        input_text = (
            f"Problem: {question}\n\nSteps completed so far:\n{history_str}".strip()
        )
        target_text = f"STEP: {step}"

        yield {
            "Problem": input_text,
            "Next Step": target_text,
        }

        history_steps.append(step)

    # Yield the final answer row
    history_str = format_history(history_steps)
    input_text = (
        f"Problem: {question}\n\nSteps completed so far:\n{history_str}".strip()
    )
    target_text = f"FINAL_ANSWER: {final_answer}"

    yield {
        "Problem": input_text,
        "Next Step": target_text,
    }


def main():
    # 1. Resolve Paths
    # Resolves to: project_root/datasets/
    script_path = Path(__file__).resolve()
    dataset_dir = script_path.parent.parent / OUTPUT_DIR_NAME
    output_path = dataset_dir / OUTPUT_FILENAME

    # Ensure directory exists
    dataset_dir.mkdir(parents=True, exist_ok=True)
    print(f"Target directory ensured at: {dataset_dir}")

    # 2. Load Data
    print("Loading GSM8K dataset...")
    gsm = load_dataset("gsm8k", "main", split="train")

    # 3. Process Data
    print("Processing examples...")
    processed_rows = []

    # Using tqdm for a progress bar
    for row in tqdm(gsm, desc="Generating Steps", unit="ex"):
        question = row["question"]
        answer = row["answer"]

        # Extend the list with all steps generated from this single example
        processed_rows.extend(process_example(question, answer))

    # 4. Save to CSV
    print(f"Total training rows generated: {len(processed_rows)}")
    df = pd.DataFrame(processed_rows)
    df.to_csv(output_path, index=False)
    print(f"Successfully saved dataset to: {output_path}")


if __name__ == "__main__":
    main()
