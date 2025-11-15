## Recommended Tutorials

1. [01_basic_agent.ipynb](../tutorials/01_basic_agent.ipynb) - Foundation: Build your first agent
2. [08_attack_red_teaming.ipynb](../tutorials/08_attack_red_teaming.ipynb) - **Red teaming techniques, PAIR attacks** ⭐
3. [06_benchmark_evaluation.ipynb](../tutorials/06_benchmark_evaluation.ipynb) - LLM-as-a-Judge for measuring ASR

**Additional helpful tutorials**: [02_custom_tools.ipynb](../tutorials/02_custom_tools.ipynb), [03_structured_output.ipynb](../tutorials/03_structured_output.ipynb), [04_model_monitoring.ipynb](../tutorials/04_model_monitoring.ipynb), [05_observability.ipynb](../tutorials/05_observability.ipynb)

## Topics of Interest - Attack Types to Explore

- **Jailbreak attacks**: Bypass safety guardrails
- **Prompt injection**: Manipulate system prompts
- **Reward hacking**: Exploit evaluation metrics
- **PAIR attacks**: Automated iterative refinement
- **Data exfiltration**: Leak training data or secrets
- **Tool misuse**: Exploit agent tool-calling
- **Hidden motivations**: Detect deceptive alignment

## Red Teaming Resources

**→ Start here:** Use our **Red Teaming Datasets** for quick, systematic testing of deployed agents.

### Red Teaming Datasets

**Location**: [`examples/red_teaming_datasets/`](./examples/red_teaming_datasets/)

Standardized test cases for evaluating agent security and robustness. Three datasets covering benign queries, harmful queries, and jailbreak prompts.

**Datasets**:

- **Benign Test Cases** (`benign_test_cases.csv`) - 101 benign test cases across 10 harm categories
- **Harmful Test Cases** (`harmful_test_cases.csv`) - 101 explicitly harmful queries for testing safety guardrails
- **Jailbreak Prompts** (`jailbreak_prompts.csv`) - 100+ jailbreak prompts (DAN, role-playing, prompt injection)

**Key Features**:

- Pre-validated queries for consistent evaluation
- Multiple attack vectors and harm categories
- Structured format for ASR calculation
- Baseline comparisons for agent responses

**How to Use**:

1. Navigate to [`examples/red_teaming_datasets/`](./examples/red_teaming_datasets/)
2. Load datasets using pandas: `pd.read_csv('benign_test_cases.csv')`
3. Test agents systematically across all categories
4. Calculate ASR for different attack types
5. See [`README.md`](./examples/red_teaming_datasets/README.md) for detailed usage examples

**Relevance to Track C**: Provides standardized test cases for systematic red teaming evaluation. Use these datasets to test deployed agents, measure ASR across different attack types, and identify vulnerabilities. Essential for thorough security assessment and comparison across different agents.


🛡️ HarmBench (Advanced Framework - Optional)

### HarmBench

**Location**: [`examples/harmbench/`](./examples/harmbench/)

A standardized evaluation framework for automated red teaming and robust refusal. HarmBench provides a complete evaluation pipeline for testing red teaming methods against LLMs and evaluating LLMs against attack methods. Includes 18 red teaming methods, support for 33+ target LLMs, and evaluation classifiers.

**Key Features**:

- Standardized evaluation framework for automated red teaming
- 18 red teaming methods (GCG, PAIR, TAP, AutoDAN, etc.)
- Support for transformers-compatible LLMs and closed-source APIs
- Evaluation pipeline: generate test cases → generate completions → evaluate completions
- HarmBench classifiers for assessing attack success
- Adversarial training methods for improving robustness

**How to Use**:

1. Navigate to [`examples/harmbench/`](./examples/harmbench/)
2. Install dependencies: `pip install -r requirements.txt`
3. Run evaluation pipeline: `python ./scripts/run_pipeline.py --methods GCG --models llama2_7b --step all`
4. See [`baselines/`](./examples/harmbench/baselines/) for red teaming method implementations
5. See [`configs/`](./examples/harmbench/configs/) for method and model configurations

**Resources**:

- **Website**: [harmbench.org](https://www.harmbench.org/)
- **Paper**: [arXiv:2402.04249](https://arxiv.org/abs/2402.04249) - HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal
- **GitHub**: [centerforaisafety/HarmBench](https://github.com/centerforaisafety/HarmBench)





⚔️ AgentHarm Benchmark (Reference)

### AgentHarm Benchmark

**Location**: [`examples/agentharm/`](./examples/agentharm/)

A benchmark for measuring harmfulness of LLM agents, including 110 explicitly malicious agent tasks (440 with augmentations) covering 11 harm categories including fraud, cybercrime, and harassment. Evaluates whether models refuse harmful agentic requests and whether jailbroken agents maintain their capabilities following an attack to complete multi-step tasks.

**Key Features**:

- 110 malicious agent tasks across 11 harm categories
- Attack patterns and jailbreak templates
- ASR (Attack Success Rate) evaluation methods
- Multi-step task completion testing

**How to Use**:

1. Navigate to [`examples/agentharm/`](./examples/agentharm/)
2. See [`src/inspect_evals/agentharm/`](./examples/agentharm/src/inspect_evals/agentharm/) for implementation
3. Load dataset from [Hugging Face](https://huggingface.co/datasets/ai-safety-institute/AgentHarm)
4. Adapt attack patterns for your red teaming tests

**Resources**:

- **Paper**: [arXiv:2410.09024](https://arxiv.org/abs/2410.09024) - AgentHarm: A Benchmark for Measuring Harmfulness of LLM Agents (ICLR 2025)
- **Dataset**: [Hugging Face](https://huggingface.co/datasets/ai-safety-institute/AgentHarm)
- **Documentation**: [Inspect Evals](https://ukgovernmentbeis.github.io/inspect_evals/evals/safeguards/agentharm/)




## Example Systems & Resources

**AgentSeer** ([Track B](../track_b_glass_box/README.md#agentseer))

An observability-based evaluation framework that decomposes agentic executions into granular action and component graphs, enabling systematic agentic-situational assessment of model- and agentic-level vulnerabilities in LLMs.

- **Paper**: [arXiv:2509.04802](https://arxiv.org/abs/2509.04802) - Mind the Gap: Evaluating Model- and Agentic-Level Vulnerabilities in LLMs with Action Graphs
- **Awards**: AAAI 2026 Demo Track, NeurIPS 2025 LLMEval, OpenAI RedTeaming Challenge Winner
- **Demo**: [Hugging Face Spaces](https://huggingface.co/spaces/holistic-ai/AgentSeer)
- **Video**: [![AgentSeer Demo Video](https://img.youtube.com/vi/8pDTIIVRwmQ/0.jpg)](https://www.youtube.com/watch?v=8pDTIIVRwmQ)


