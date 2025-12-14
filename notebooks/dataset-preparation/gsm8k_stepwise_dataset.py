import re
import pandas as pd
from datasets import load_dataset


def extract_final_answer(answer_text: str) -> str | None:
    m = re.search(r"####\s*(.+)", answer_text)
    if not m:
        return None
    return m.group(1).strip()


def extract_rationale(answer_text: str) -> str:
    return answer_text.split("####")[0].strip()


def split_rationale_into_steps(rationale: str) -> list[str]:
    # --- 1) Try line-based splitting ---
    raw_lines = [ln.strip() for ln in rationale.split("\n") if ln.strip()]

    # If we got multiple non-empty lines, assume each line is a step.
    if len(raw_lines) > 1:
        return raw_lines

    # --- 2) Fallback: sentence-based splitting ---
    text = rationale.replace("\n", " ").strip()

    # Split on sentence boundaries ('.', '!', '?') followed by whitespace
    raw_sentences = re.split(r"(?<=[\.!?])\s+", text)
    steps = [s.strip() for s in raw_sentences if s.strip()]

    return steps


if __name__ == "__main__":
    #  Load GSM8K train split
    gsm = load_dataset("gsm8k", "main", split="train")
    print("Loaded GSM8K train size:", len(gsm))

    # -----------------------------------------------------------
    # 3. Build stepwise training rows (input, target)
    # -----------------------------------------------------------

    rows = []

    for idx, row in enumerate(gsm):
        question = row["question"]
        full_answer = row["answer"]

        final_answer = extract_final_answer(full_answer)
        if final_answer is None:
            # Skip weirdly formatted items
            continue

        rationale = extract_rationale(full_answer)
        steps = split_rationale_into_steps(rationale)

        # If we didn't get any steps, skip
        if not steps:
            continue

        # We'll keep a list of previous steps to build history each time.
        history_steps: list[str] = []

        # STEP calls (one row per step)
        for step in steps:
            # Build history string from previous steps
            if history_steps:
                history_str = ""
                for i, h in enumerate(history_steps, start=1):
                    history_str += f"STEP {i}: {h}\n"
                history_str = history_str.rstrip()
            else:
                history_str = ""  # empty string for first call

            # Build full input context
            input_text = (
                f"Problem: {question}\n\nSteps completed so far:\n{history_str}"
            )

            target_text = f"STEP: {step}"

            rows.append(
                {
                    "Problem": input_text,
                    "Next Step": target_text,
                }
            )

            # Add this step to history for the next call
            history_steps.append(step)

        # Final call: only FINAL_ANSWER
        if history_steps:
            history_str = ""
            for i, h in enumerate(history_steps, start=1):
                history_str += f"STEP {i}: {h}\n"
            history_str = history_str.rstrip()
        else:
            history_str = ""  # should basically never happen here

        input_text = f"Problem: {question}\n\nSteps completed so far:\n{history_str}"
        final_target = f"FINAL_ANSWER: {final_answer}"

        rows.append(
            {
                "Problem": input_text,
                "Next Step": final_target,
            }
        )

        if (idx + 1) % 100 == 0:
            print(
                f"Processed {idx + 1}/{len(gsm)} problems, total rows so far: {len(rows)}"
            )

        print("Total training rows:", len(rows))

    #  Save to CSV

    df = pd.DataFrame(rows, columns=["Problem", "Next Step"])
    df.to_csv("gsm8k_stepwise.csv", index=False)
    print("Saved to gsm8k_stepwise.csv")
