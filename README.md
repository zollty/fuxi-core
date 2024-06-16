<div id="top"></div>
<div align="center">
  <h2>HP-deploy</h2>

[English](README.md) | 简体中文

</div>
**HP**-deploy的目标是成为一个优秀的模型部署平台，它建立在多种高性能开源推理框架之上，提供多模型实例、多推理框架、多云、自动扩缩容与可观测性等能力。   

______________________________________________________________________

## 功能更新 🎉
 - 支持sglang等模型运行框架，支持Qwen2等模型
 - 支持多模型、多卡、多机分布式推理服务
 - 支持Huggingface、vllm等模型运行框架
 - 支持PyTorch、TurboMind等推理引擎
 - 具备动态拆分和融合，张量并行，高效计算等特性
 - 支持flash-attention2、Paged Attention、attention kernel、GQA等技术
 - 支持使用 [AWQ](https://arxiv.org/abs/2306.00978) 算法进行 4-bit 量化
 - 支持 Llama-2 7B/13B/70B、Baichuan2-7B、InternLM-20B等模型（参见附表）
 - 支持 Qwen-7B，Qwen-14B等，支持动态NTK-RoPE缩放，动态logN缩放
 - 计划支持多模态，包括：视觉-语言模型（VLM）

______________________________________________________________________

# 简介

**HP**-deploy，即 （**High Performance**） deploy，主打高性能运行模型，简洁纯粹，为模型提供高效稳定的运行服务。

HP-deploy 涵盖了 LLM 任务的全套轻量化、部署和服务解决方案，并通过接口和CLI轻松管理模型的发布和运行。
这个强大的工具箱提供以下核心功能：

- **高效的推理**：HP-deploy 底层采用各种最先进的推理运行框架和引擎，集成多种优化技术，且对模型运行服务进行极致优化，推理性能做到比FastChat、 vLLM更好，快20%甚至更多。

- **轻松的管理**：支持部署工具服务和远程访问，通过CLI命名或者API调用，一键启动、停止、重启、配置模型，十分的方便。

- **便捷的服务**：通过注册中心、多进程worker管理、请求分发服务，HP-deploy 支持多模型在多机、多卡上的推理服务。

- **可靠的量化**：HP-deploy 支持权重量化和 k/v 量化，性能好且稳定。

- **有状态推理**：通过缓存多轮对话过程中 attention 的 k/v，记住对话历史，从而避免重复处理历史会话。显著提升长文本多轮对话场景中的效率。

它是“风后®AI”服务开发所用的模型运行框架，具备企业级产品支撑能力。
它也可以和其他框架协作和集成，目前主要集成了vLLM和FastChat，优化了相关功能。

# 性能

HP-deploy使用各种底层优化技术，例如TurboMind 引擎的卓越推理能力，每秒处理的请求数是 vLLM 的 1.36 ~ 1.85 倍。

支持Nvidia多种显卡，例如：
- A100
- 4090
- 3090
- 2080

# 支持的模型

|    Model     |    Size    |
|:------------:|:----------:|
|    Llama     |  7B - 65B  |
|    Llama2    |  7B - 70B  |
|   ChatGLM2   |     6B     |
|   ChatGLM3   |     6B     |
|     QWen     |  7B - 72B  |
|   QWen-VL    |     7B     |
|   QWen1.5    | 0.5B - 72B |
|   Baichuan   |  7B - 13B  |
|  Baichuan2   |  7B - 13B  |
|   InternLM   |  7B - 20B  |
|  InternLM2   |  7B - 20B  |
|  Code Llama  |  7B - 34B  |
|    Falcon    | 7B - 180B  |
|      YI      |  6B - 34B  |
|   Mistral    |     7B     |
| DeepSeek-MoE |    16B     |
|   Mixtral    |    8x7B    |
|    Gemma     |   2B-7B    |

HP-deploy 支持 2 种推理引擎： TurboMind 和 PyTorch，它们侧重不同。前者追求推理性能的极致优化，后者纯用python开发，着重降低开发者的门槛。 
它们在支持的模型类别、计算精度方面有所差别。

另外支持多种推理优化和量化技术，用户可并根据实际需求选择合适的。

# 快速开始

## 安装

使用 pip ( python 3.8+) 安装 HP-deploy，或者 推荐 [源码安装]

```shell
pip install HP-deploy
```

## 贡献指南

我们感谢所有的贡献者为改进和提升 HP-deploy 所作出的努力。欢迎参与项目贡献。

## 致谢

- [FasterTransformer](https://github.com/NVIDIA/FasterTransformer)
- [llm-awq](https://github.com/mit-han-lab/llm-awq)
- [vLLM](https://github.com/vllm-project/vllm)
- [DeepSpeed-MII](https://github.com/microsoft/DeepSpeed-MII)
- [FastChat](https://github.com/lm-sys/FastChat)

## License

该项目采用 [Apache 2.0 开源许可证](LICENSE)。

<p align="right"><a href="#top">🔼 Back to top</a></p>