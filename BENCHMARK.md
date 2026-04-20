# dev-council Benchmarking Guide

This document outlines the strategy and instructions for benchmarking `dev-council` to measure its proficiency in software engineering tasks, planning, and multi-model consensus.

## 1. Benchmarking Strategy

To evaluate the system for a research paper, we focus on three dimensions:

- **Autonomous Coding**: Ability to solve complex tasks and use tools correctly.
- **BTP Pipeline Efficiency**: Quality of SRS, Milestones, and Tech Stack plans.
- **Council Consensus**: Accuracy gain from multi-model collaboration.

## 2. BigCodeBench (BCB)

BigCodeBench evaluates instruction-following and library usage (pandas, matplotlib, etc.).

### Harness Setup

The harness is located at `tests/benchmark_bcb.py`. It automates task retrieval from HuggingFace and headless execution via the CLI.

**Run Generation:**

```bash
python tests/benchmark_bcb.py --subset instruct --n-tasks 10 --model local/gpt-oss:20b
```

**Evaluation Pipeline:**

1. Generate `samples.jsonl`.
2. Sanitize: `bigcodebench.sanitize --samples samples.jsonl --calibrate`
3. Evaluate: `bigcodebench.evaluate --subset instruct --samples samples-sanitized-calibrated.jsonl`

---

## 3. SWE-bench (Software Engineering)

SWE-bench measures the ability to resolve real GitHub issues in large repositories.

**Strategy:**

- Select a subset of **SWE-bench Verified**.
- Feed the issue description to `dev-council`.
- Evaluate if the generated patch passes the repository's test suite.

---

## 4. Multi-Model Council Evaluation

Measure the effectiveness of the `/council` command.

**Experiment Design:**

1. Pick a task (e.g., from BigCodeBench or a custom complex requirement).
2. Run with a single model: `python dev_council.py --print "TASK"`
3. Run with the Council: `python dev_council.py /council "TASK"`
4. **Metrics:**
   - **Pass Rate**: Does the Council reach a correct solution more often than single models?
   - **Consensus Quality**: Use "LLM-as-a-Judge" to compare single-model proposals vs. the synthesized consensus.

---

## 5. Requirements & Planning (BTP)

Evaluate the quality of the generated BTP artifacts (`btp/srs.md`, etc.).

**Methodology:**

- Use the **SRS-Benchmark** rubric (Completeness, Consistency, Feasibility).
- Employ a "Judge LLM" (e.g., Claude 3.5 Sonnet) to grade 100 generated SRS documents on a scale of 1-5.

---

## 6. Multi-LLM Consensus (Council) Research

When evaluating the `/council` consensus mechanism, it is important to measure not just the code accuracy, but the quality of the collaborative process.

### Specialized Benchmarks

- **MultiAgentBench**: Evaluates interactive, multi-turn coordination and communication between agents. Uses milestone-based KPIs for planning and coordination.
- **CONSENSAGENT**: Specifically designed to evaluate and optimize consensus mechanisms. It measures "sycophancy" (agents blindly agreeing) and debate efficiency.
- **E2EDevBench**: Focuses on how agent teams handle end-to-end software development, including requirement comprehension and labor division.
- **ReConcile**: A standard baseline framework for evaluating multi-agent reasoning and iterative debate.

### Consensus-Specific Metrics

- **Sycophancy Score**: The tendency of secondary models to agree with the first model's proposal regardless of correctness. Lower is better.
- **Convergence Speed**: The number of turns/rounds of debate required to reach a stable consensus.
- **Communication Overhead**: The ratio of tokens spent on coordination vs. tokens spent on the final implementation.
- **Debate Quality (LLM-as-a-Judge)**: Rating the logical soundness and critical thinking in the inter-agent discussion logs.

## 7. Key Metrics for Research

- **Pass@1**: Probability that the first generated solution is correct.
- **Tool Selection Accuracy**: % of correct tool choices (e.g., choosing `Edit` vs `Write`).
- **Token Efficiency**: Consensus gain per token spent.
- **Mean Time to Resolution (MTTR)**: Number of turns taken to solve a task.

---

## 7. Supported Datasets Reference

| Benchmark            | Dataset ID (HuggingFace)           | Primary Language | Description                                                  |
| :------------------- | :--------------------------------- | :--------------- | :----------------------------------------------------------- |
| **BigCodeBench**     | `bigcode/bigcodebench`             | Python           | Hard tasks with library usage (Pandas, Matplotlib, etc).     |
| **SWE-bench**        | `princeton-nlp/SWE-bench_Verified` | Python           | Real-world GitHub issues and PRs.                            |
| **HumanEval**        | `openai/openai-python-humaneval`   | Python           | 164 hand-written algorithmic problems.                       |
| **MBPP**             | `google-research-datasets/mbpp`    | Python           | Mostly Basic Python Problems.                                |
| **DS-1000**          | `xlangai/DS-1000`                  | Python           | Data science tasks (NumPy, SciPy, Pandas, etc).              |
| **RepoBench**        | `microsoft/RepoBench`              | Polyglot         | Repository-level code completion.                            |
| **LiveCodeBench**    | `livecodebench/livecodebench`      | Polyglot         | Contamination-resistant problems from competitive platforms. |
| **Aider Benchmarks** | `aider-ai/aider-bench`             | Polyglot         | Specifically for code editing and refactoring.               |

---

## 8. Integration Map

| dev-council Feature           | Recommended Benchmark          |
| :---------------------------- | :----------------------------- |
| **Python Tool Use**           | BigCodeBench                   |
| **Consensus Accuracy**        | HumanEval / BigCodeBench       |
| **Multi-file Navigation**     | RepoBench / SWE-bench          |
| **Refactoring (`Edit` tool)** | Aider Benchmark                |
| **BTP SRS/Milestones**        | ArchBench (and Custom Rubrics) |

1. pip install bigcodebench
2. bigcodebench.sanitize --samples samples.jsonl --calibrate
3. bigcodebench.evaluate --subset instruct --samples samples-sanitized-calibrated.jsonl
