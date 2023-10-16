<h1 align="center"> 🛠️LLM Builder <h1/>
<p>
    <p align="center"> Facilitate the Seamless Construction of Robust Large Language Models <p/>
<p/>

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Getting Started](#getting-started)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)
- [Documentation](#documentation)
- [Support](#support)
- [FAQ](#faq)
- [Acknowledgments](#acknowledgments)

## Introduction

LLM Builder is an open-source deep learning training library specifically for building Large Language Models (LLMs) by [Faheem](https://github.com/TheFaheem). A visionary library designed to make the complex and often formidable task of building and training Large Language Models (LLMs) accessible to all, from fledgling enthusiasts to seasoned researchers. It have painstakingly removed the barriers of intricate code work, enabling a streamlined journey towards LLM construction without the burden of extensive code wrangling.

Our repository empowers users to harness the potential of state-of-the-art language models without the need for extensive technical expertise. At its core, **LLM Builder** stands as an all-encompassing solution, encapsulating every facet of the LLM development process, spanning from architecting customizable model structures, handling dataset preparation to paving the way for a smoother transition to the training optimization phase, and meticulous logging, allowing you to focus on the big picture of your research.

Liberated from hardware constraints, LLM Builder extends its embrace to the entire spectrum of training environments, encompassing single GPUs, TPU hosts, TPU cores, and the orchestration of multi-GPU and TPU devices.

## Features

Check out [FEATURES.md](https://github.com/TheFaheem/llm-builder/blob/main/FEATURES.md) for more detailed and elaborate features of this repo.

## Getting Started

### Installation

```shell
# Example installation command
pip install llm-builder
```

### Usage

```python
from llm_builder import LLMBuilderConfig, LLMBuilder

# Build the LLM
config = LLMBuilderConfig({'train': [("tinystories", 1.0)], 'validation': [("tinystories", 1.0)]})
llm_builder = LLMBuilder(config)
llm_builder.build()
```

### Configuration

Configuring LLM Builder to meet your specific needs is made easy with the `LLMBuilderConfig` class. Below, we explain how to fine-tune the settings for your language model construction:

**Language Model Configuration**

- **Customizable Architecture**: LLM Builder empowers you to craft a language model architecture that suits your specific research goals. Experiment with different model configurations, including the number of layers, attention mechanisms, and embedding layers.

- **Effortless Dataset Preparation**: Simplify the data preparation process with dataset_preparation_config that allow you to control dataset file paths, vocabulary size, and special tokens (e.g., eos, bos, pad, unk). This ensures that your data is compatible with the model.

- **Versatile Training Optimization**: Fine-tune your model's performance with control over hyperparameters such as learning rate, weight decay, gradient clipping, and more.

- **Multi-Device Training**: LLM Builder allows you to specify the hardware environment for training, whether it's a single CPU, GPU, TPU host, TPU core, or a multi-GPU setup. You can scale training efficiently across multiple devices, making the most of your available resources.

- **Checkpointing and Resuming**: Take advantage of checkpointing to save and resume your training from the last checkpoint. This feature preserves your training progress, even in cases of interruptions or system failures.

- **Comprehensive Logging**: Gain insights into your model's performance with detailed logging. Track training progress, hyperparameter settings, and evaluation metrics. Visualize training statistics and loss curves for a deeper understanding of your model's behavior.

**Additional Configuration Settings**

- **WandB Integration**: Optionally, you can enable logging to Weights and Biases (WandB) for enhanced project monitoring and collaboration. Configure the project name and run name as needed.

- **Reproducibility**: Specify a random seed for reproducibility, ensuring consistent results across training runs.

- **Compile and Dataset Preparation Flags**: Choose whether to compile the model before training and specify whether dataset preparation is required if not already. These flags give you full control over the model development process.
  
With LLM Builder's comprehensive configuration options, you can create and train your Large Language Models with precision and flexibility.

For more detailed information about each configuration attributes, please refer to the [`LLMBuilderConfig class`](https://github.com/TheFaheem/llm-builder/blob/main/llm-builder/llm_builder.py#L38) docstring.


## Contributing

We highly welcome contributions to LLM-Builder! Whether you want to report issues, submit pull requests, include specific functionality or provide feedback, please follow our [Contributing Guidelines](https://github.com/TheFaheem/llm-builder/blob/main/CONTRIBUTION.md).

## License

This project is licensed under the [Your License Name](LICENSE) - see the [LICENSE](https://github.com/TheFaheem/llm-builder/blob/main/LICENSE) file for more details.

## Documentation

Contribution Needed!

## Support

If you have questions or run into issues, please visit our [Issue Tracker]([link-to-issues](https://github.com/TheFaheem/llm-builder/issues)) for support and bug reporting.


