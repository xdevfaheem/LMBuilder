<h1 align="center"> 🛠️LLM Builder <h1/>
<p>
    <p align="center"> Facilitate Seamless Construction of Robust Large Language Models <p/>
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

For elaborate information, please refer to our [FEATURES.md](https://github.com/TheFaheem/llm-builder/blob/main/FEATURES.md).

## Getting Started

### Installation

```shell
# Example installation command
pip install llm-builder
```

### Usage

```python
from llm_builder import LLMBuilderConfig, LLMBuilder

# Configuring the llm-builder
config = LLMBuilderConfig({'train': [("tinystories", 1.0)], 'validation': [("tinystories", 1.0)]})

# Build LLM and Train
llm_builder = LLMBuilder(config)
llm_builder.build()
```

## Configuration

Configuring LLM Builder to suit your specific needs is straightforward with the `LLMBuilderConfig` class. which provides settings, hyperparameters for your LLM and key attributes for precise control. This section highlights some of it's key attributes and settings:

- `model_configs (dict)`: Tailor your LLM's architecture, adjust the number of layers, attention mechanisms, and more to align with your research objectives.
- `dataset_config (dict)`: A dictionary containing a list of tuples specifying the prefixes of the memap files and their distribution weights for each data split.
- `data_prep_config (dict)`: Configure data preparation by specifying dataset file paths, block size, vocabulary settings, and special tokens for compatibility with your data.
- `device_type (str)`: Type of device for training (e.g., 'cpu', 'gpu', 'tpu').
- `out_dir (str)`: The directory where model checkpoints and logs will be saved.
- `data_dir (str)`: The directory containing the data and dataset files.
- `wandb_log (bool)`: A flag for logging to WandB (Weights and Biases).
- `wandb_project (str)`: Weights and Biases project name.
- `wandb_run_name (str)`: Name for the Weights and Biases run.
- `batch_size_per_device (int)`: Batch size per device used in training.
- `init_from (str)`: Initialization of model from ('scratch' or 'resume').
- `seed (int)`: Random seed for reproducibility.
- `learning_rate (float)`: Learning rate for training.
- `max_iters (int)`: Maximum number of training iterations.
- `weight_decay (float)`: Weight decay for optimization.
- `beta1 (float)`: Beta1 parameter for optimization.
- `beta2 (float)`: Beta2 parameter for optimization.
- `grad_clip_value (float)`: Gradient clipping value.
- `decay_lr (bool)`: A flag for learning rate decay.
- `warmup_iters (int)`: Number of warm-up iterations for learning rate scheduling.
- `lr_decay_iters (int)`: Number of iterations for learning rate decay.
- `min_lr (float)`: Minimum learning rate.
- **And more...**

With LLM Builder's extensive configuration options, you can build and train your Large Language Models with precision and flexibility.

For details about other configurations attributes, please refer to the [LLMBuilderConfig class documentation](https://github.com/TheFaheem/llm-builder/blob/main/llm-builder/llm_builder.py#L38).

## Contributing

We highly welcome contributions to LLM-Builder! Whether you're a seasoned developer or just getting started, your involvement in our open-source project is highly appreciated inspite of whether you want to report issues, report bug, or report any unexpected behavior, submit pull requests which include specific functionality or provide feedback,

Detailed instructions on how you can contribute can be found in our [Contributing Guidelines](https://github.com/TheFaheem/llm-builder/blob/main/CONTRIBUTING.md). Join me in making LLM-Builder a more efficient tool for building and training Large Language Models (LLMs) and collaborating with the community.


## License

[![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

This work is licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa] - see the [LICENSE](https://github.com/TheFaheem/llm-builder/blob/main/LICENSE) file for details.

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg

## Documentation

Still haven't done yet. Contribution needed!

## Support

If you have questions or run into issues, please visit our [Issue Tracker](https://github.com/TheFaheem/llm-builder/issues) for support and bug reporting.

## Citation

This project is currently contributed by [Faheem](https://github.com/TheFaheem). If you find LLM-Builder helpful for your research or work, please cite:
```bibtex
@online{llmbuilder,
    author = {Mohammed Faheem},
    title = {🛠️LLM-Builder: Facilitate Seamless Construction of Robust Large Language Models},
    url = {https://github.com/TheFaheem/llm-builder},
    year = {2023},
    month = {oct}
}
```
Your contribution is greatly appreciated, and I encourage you to acknowledge it when using 🛠️LLM-Builder in your research or projects.

