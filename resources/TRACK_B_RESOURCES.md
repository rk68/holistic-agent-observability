## Recommended Tutorials

1. [01_basic_agent.ipynb](../tutorials/01_basic_agent.ipynb) - Foundation
2. [02_custom_tools.ipynb](../tutorials/02_custom_tools.ipynb) - Tool integration
3. [03_structured_output.ipynb](../tutorials/03_structured_output.ipynb) - Structured data for better tracing
4. [04_model_monitoring.ipynb](../tutorials/04_model_monitoring.ipynb) - Monitor tokens, costs, and performance
5. [05_observability.ipynb](../tutorials/05_observability.ipynb) - **Deep tracing with LangSmith** ⭐

## AWS Observability Tools

AWS provides multiple services for monitoring and tracing applications, including CloudWatch for logs and metrics, and X-Ray for distributed tracing.

- **AWS CloudWatch**: [CloudWatch Documentation](https://docs.aws.amazon.com/cloudwatch/)
  - Logs, metrics, alarms, dashboards
  - Integration with AWS services
  - Custom metrics and log insights
- **AWS X-Ray**: [X-Ray Documentation](https://docs.aws.amazon.com/xray/)
  - Distributed tracing for microservices
  - Service map visualization
  - Performance bottleneck identification
- **AWS Bedrock Observability**: [Bedrock Monitoring](https://docs.aws.amazon.com/bedrock/latest/userguide/monitoring.html)
  - CloudWatch integration for Bedrock API calls
  - Token usage and latency metrics
  - Cost tracking and usage analytics
- **OpenTelemetry + AWS**: [AWS Distro for OpenTelemetry](https://aws-otel.github.io/)

  - Pre-configured OpenTelemetry collector
  - Integration with CloudWatch and X-Ray
  - Automatic instrumentation for AWS services



## Getting Started with Observability

**→ Start with Tutorial 05: [05_observability.ipynb](../tutorials/05_observability.ipynb)**

The tutorial uses **LangSmith** for tracing and visualization:

- **Documentation**: [LangSmith Documentation](https://docs.smith.langchain.com/)
- **Installation**: `pip install langsmith`
- **Key Features**: Automatic tracing, execution visualization, prompt management, evaluation
- **Quick Start**: Set environment variables (`LANGSMITH_API_KEY`, `LANGSMITH_TRACING=true`) and LangSmith will automatically capture all agent traces

## Additional Observability Tools

**OpenTelemetry**

Vendor-neutral, open-source observability framework for cloud-native software. Standardizes collection of traces, metrics, and logs across languages and platforms.

- **Documentation**: [OpenTelemetry Docs](https://opentelemetry.io/docs/)
- **Python SDK**: `pip install opentelemetry-api opentelemetry-sdk`
- **Key Features**: Standardized telemetry collection, vendor-neutral, supports traces/metrics/logs, context propagation, wide language support

**LangFuse**

Open-source LLM engineering platform for debugging, analyzing, and iterating on LLM applications. Offers tracing, prompt management, evaluations, and analytics.

- **Documentation**: [LangFuse Docs](https://langfuse.com/docs)
- **GitHub**: [langfuse/langfuse](https://github.com/langfuse/langfuse)
- **Installation**: `pip install langfuse` or self-hosted deployment
- **Key Features**: Automatic tracing with `@observe()` decorator, nested LLM call tracking, OpenTelemetry support, prompt versioning, cost tracking

## Featured Research Platforms

**AgentGraph** — Trace-to-Graph Visualization Platform

A trace-to-graph platform for interactive analysis and robustness testing in agentic AI systems (AAAI 2026 Demo Track). Converts execution logs into interactive knowledge graphs with nodes representing agents, tasks, tools, inputs/outputs, and humans, enabling both qualitative failure detection and quantitative robustness evaluation.

- **Paper**: [`examples/agent_graph_AAAI.pdf`](examples/agent_graph_AAAI.pdf) - AAAI 2026 Demo Track
- **Video**: [![AgentGraph Demo Video](https://img.youtube.com/vi/btrS9pfDYJY/0.jpg)](https://www.youtube.com/watch?v=btrS9pfDYJY)
- **Key Features**:
  - Converts execution traces into interactive knowledge graphs
  - Visual analysis of agent behavior and decision paths
  - Failure detection across five risk categories
  - Quantitative robustness evaluation


**AgentSeer** — Observability-Based Evaluation Framework

An observability-based evaluation framework that decomposes agentic executions into granular action and component graphs, enabling systematic agentic-situational assessment of model- and agentic-level vulnerabilities in LLMs.

- **Paper**: [arXiv:2509.04802](https://arxiv.org/abs/2509.04802) - Mind the Gap: Evaluating Model- and Agentic-Level Vulnerabilities in LLMs with Action Graphs
- **Awards**: AAAI 2026 Demo Track, NeurIPS 2025 LLMEval, OpenAI RedTeaming Challenge Winner
- **Demo**: [Hugging Face Spaces](https://huggingface.co/spaces/holistic-ai/AgentSeer)
- **Video**: [![AgentSeer Demo Video](https://img.youtube.com/vi/8pDTIIVRwmQ/0.jpg)](https://www.youtube.com/watch?v=8pDTIIVRwmQ)
- **Key Features**:
  - Decomposes agent executions into granular action graphs
  - Systematic assessment of model-level vs. agentic-level vulnerabilities
  - Identifies "agentic-only" vulnerabilities that emerge in agentic contexts
  - Action and component graph visualization
