## Recommended Tutorials

1. [01_basic_agent.ipynb](../tutorials/01_basic_agent.ipynb) - Foundation: Introduction to LangGraph ReAct agents
2. [02_custom_tools.ipynb](../tutorials/02_custom_tools.ipynb) - Build production tools: Create custom tools for your agents
3. [03_structured_output.ipynb](../tutorials/03_structured_output.ipynb) - Efficient data handling: Get validated JSON responses
4. [04_model_monitoring.ipynb](../tutorials/04_model_monitoring.ipynb) - **Performance tracking: Track tokens, costs, and carbon emissions** ⭐
5. [06_benchmark_evaluation.ipynb](../tutorials/06_benchmark_evaluation.ipynb) - Evaluation: Measure agent performance and improvements
6. [07_reinforcement_learning.ipynb](../tutorials/07_reinforcement_learning.ipynb) - Advanced optimization: RL concepts for agent training (Optional)

**Additional helpful tutorials**: [05_observability.ipynb](../tutorials/05_observability.ipynb), [08_attack_red_teaming.ipynb](../tutorials/08_attack_red_teaming.ipynb)

## AWS Tools & Frameworks

**Building Agents on AWS** — Comprehensive Guide

A complete guide to building production-ready AI agents on AWS infrastructure. Covers agent architecture, deployment strategies, and AWS service integration.

- **Guide**: [`Building Agents on AWS.pdf`](Building%20Agents%20on%20AWS.pdf) - AWS agent development tutorial


**AWS Strands Agents SDK**

A model-driven SDK for building and running AI agents with a flexible, lightweight, and model-agnostic approach. Supports multi-agent workflows, MCP server integration, and deployment to AWS Bedrock AgentCore.

- **Documentation**: [AWS Prescriptive Guidance - Strands Agents](https://docs.aws.amazon.com/prescriptive-guidance/latest/agentic-ai-frameworks/strands-agents.html)
- **Quick Start**: [Build and Deploy Production-Ready AI Assistant](https://aws.amazon.com/getting-started/hands-on/strands-agentic-ai-assistant/)
- **GitHub**: [strands-agents/docs](https://github.com/strands-agents/docs)
- **Installation**: `pip install strands-agents`
- **Key Features**: Multi-agent workflows, sequential pipelines, orchestrator patterns, MCP integration, Bedrock AgentCore deployment


**Amazon Nova**

A family of multimodal generative AI models supporting text, image, and video generation. Includes Nova Lite (fast, cost-effective) and Nova Premier (advanced capabilities with 1M token context window).

- **Documentation**: [Amazon Nova User Guide](https://docs.aws.amazon.com/nova/latest/userguide/what-is-nova.html)
- **Models**: Nova Lite (`us.amazon.nova-lite-v1:0`), Nova Premier (`us.amazon.nova-premier-v1:0`)
- **APIs**: Converse API (conversational), Invoke API (direct inference)
- **Features**: Multimodal support (text, image, video), large context windows, fine-tuning capabilities


**AWS Bedrock**

Fully managed service for building and scaling generative AI applications. Provides access to foundation models from Amazon, Anthropic, Meta, Mistral AI, and more.

- **Documentation**: [AWS Bedrock Documentation](https://docs.aws.amazon.com/bedrock/)
- **Models**: Claude (Anthropic), Llama (Meta), Mistral, Titan (Amazon), and more
- **Features**: Model access, fine-tuning, RAG, agents, guardrails


**AWS Bedrock AgentCore**

Runtime for deploying and running AI agents at scale. Supports Strands Agents SDK and provides infrastructure for agent execution.

- **Documentation**: [Bedrock AgentCore Runtime](https://docs.aws.amazon.com/bedrock/latest/userguide/agentcore.html)
- **Integration**: Works seamlessly with Strands Agents SDK
- **Features**: Serverless deployment, auto-scaling, monitoring


## Example Systems & Resources

**Small Language Models are the Future of Agentic AI** (NVIDIA Research)

A position paper arguing that small language models (SLMs) are sufficiently powerful, inherently more suitable, and necessarily more economical for many invocations in agentic systems. Discusses the operational and economic impact of shifting from LLMs to SLMs, outlines a general LLM-to-SLM agent conversion algorithm, and advocates for heterogeneous agentic systems leveraging SLMs for routine tasks and LLMs for complex reasoning.

- **Website**: [NVIDIA Research](https://research.nvidia.com/labs/lpr/slm-agents/)
- **Paper**: [arXiv:2506.02153](https://arxiv.org/abs/2506.02153) - Small Language Models are the Future of Agentic AI
- **Key Recommendations**: Prioritize SLMs for cost-effective deployment, design modular agentic systems, leverage SLMs for rapid specialization


**AgentHarm Benchmark**

A benchmark for measuring harmfulness of LLM agents, including 110 explicitly malicious agent tasks (440 with augmentations) covering 11 harm categories including fraud, cybercrime, and harassment. Evaluates whether models refuse harmful agentic requests and whether jailbroken agents maintain their capabilities following an attack to complete multi-step tasks.

- **Paper**: [arXiv:2410.09024](https://arxiv.org/abs/2410.09024) - AgentHarm: A Benchmark for Measuring Harmfulness of LLM Agents (ICLR 2025)
- **Dataset**: [Hugging Face](https://huggingface.co/datasets/ai-safety-institute/AgentHarm)
- **Documentation**: [Inspect Evals](https://ukgovernmentbeis.github.io/inspect_evals/evals/safeguards/agentharm/)
- **Codebase**: [GitHub](https://github.com/UKGovernmentBEIS/inspect_evals/tree/main/src/inspect_evals/agentharm)


