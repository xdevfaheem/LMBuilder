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

## Introduction

LLM Builder is an open-source deep learning training library designed exclusively for constructing Large Language Models (LLMs). It aims to democratize the often complex and daunting task of LLM development, making it accessible to both novices and seasoned researchers. By eliminating the need for intricate code work, LLM Builder streamlines the path to constructing LLMs.

Our repository empowers users to leverage cutting-edge language models without requiring extensive technical expertise. At its core, **LLM Builder** offers a comprehensive solution, addressing every aspect of the LLM development process, from customizable model design, dataset preparation, to seamless training optimization, meticulous logging, checkpointing, distributed training, and more. This allows you to focus on the broader aspects of your research, unburdened by technical complexities.

## Features

- **Customizable Architecture:** Customize your own LLM architecture with pre-defined architecture or (pre-built or custom) modules.
  
- **Effortless Dataset Preparation:** Simplify data preparation from preprocessing, tokenization and building the dataset files in suitable format for ypur LLM pre-training.
  
- **Distributed Training:** Optimize training across diverse devices for superior performance, with support for various hardware configurations and efficient scaling across multiple devices.

- **Comprehensive Logging and Checkpointing:** Track every aspect of LLM development and resume training seamlessly.

For elaborate information on each feature, please refer to our [FEATURES.md](https://github.com/TheFaheem/llm-builder/blob/main/FEATURES.md).

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

Configuring LLM Builder to meet your specific needs is made easy with the `LLMBuilderConfig` class. Below, we explain some configuration settings to build your LLM:

**Language Model Configuration Attributes**

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

For more detailed information about each configuration attributes, please refer to the [`LLMBuilderConfig class`](https://github.com/TheFaheem/llm-builder/blob/main/llm-builder/llm_builder.py#L38) docstring (as for now).


## Contributing

We highly welcome contributions to LLM-Builder! Whether you want to report issues, submit pull requests, include specific functionality or provide feedback, please follow our [Contributing Guidelines](https://github.com/TheFaheem/llm-builder/blob/main/CONTRIBUTION.md).

## License

[![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

This work is licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa] - see the [LICENSE](https://github.com/TheFaheem/llm-builder/blob/main/LICENSE) file for details.

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg

## Documentation

Contribution Needed!

## Support

If you have questions or run into issues, please visit our [Issue Tracker](https://github.com/TheFaheem/llm-builder/issues) for support and bug reporting.


