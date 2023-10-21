<h1 align="center"> 🛠️LMBuilder <h1/>
<p align="center"> Facilitate Seamless Construction of Robust Large Language Models <p/>

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

Welcome to 🛠LMBuilder, an open-source deep learning training library specifically for constructing Large Language Models (LLMs) which is designed to democratize the often complex and daunting task of LLM development, making it accessible to both novices and seasoned researchers. By painstakingly removing the barriers of intricate code work, 🛠LMBuilder enables a streamlined journey towards LLM construction.

Our library empowers users to build and harness the potential of state-of-the-art language models without requiring extensive technical expertise. At its core, 🛠LMBuilder aims to encapsulates every aspect of the LLM development process, spanning from customizable model architectures to handling dataset preparation, seamless training optimization, meticulous logging, checkpointing, distributed training, etc and allows you to focus on the broader aspects of your research, unburdened by technical complexities.

## Features

- **Customizable Architecture:** Customize your LLM architecture with pre-defined module or -built or custom) modules.
  
- **Effortless Dataset Preparation:** Simplify data preparation from preprocessing, tokenization and building the dataset files in suitable format for your LLM pre-training.
  
- **Distributed Training:** Optimize training across diverse devices with support for various hardware configurations and efficient scaling across multiple devices.

- **Comprehensive Logging and Checkpointing:** Track every aspect of LLM development and resume training seamlessly.

For elaborate information, please refer to our [FEATURES.md](https://github.com/TheFaheem/LMBuilder/blob/main/FEATURES.md).

## Getting Started

### Installation

```shell
# Example installation command
pip install lmbuilder
```

### Usage

```python
from lmbuilder import LMBuilderConfig, LMBuilder

# Configuring the lmbuilder
config = LMBuilderConfig({'train': [("tinystories", 1.0)], 'validation': [("tinystories", 1.0)]})

# Build LLM and Train
llm_builder = LMBuilder(config)
llm_builder.build()
```

## Configuration

Configuring 🛠️LMBuilder to suit your specific needs is straightforward with the `LMBuilderConfig` class. which provides settings, hyperparameters for your LLM for precise control. Here are some of it's key attributes and settings:

- `model_configs (dict)`: Tailor your LLM's architecture, adjust the number of layers, attention mechanisms, and more to align with your research objectives.
- `dataset_config (dict)`: A dictionary containing a list of tuples specifying the prefixes of the dataset files and their distribution weights for each data split.
- `data_prep_config (dict)`: Configure data preparation by specifying dataset file paths, block size, vocabulary settings, and special tokens for compatibility with your data.
- `device_type (str)`: Type of device for training (e.g., 'cpu', 'gpu', 'tpu').
- **...(And more)**

With 🛠️LMBuilder's extensive configuration options, you can build and train your Large Language Models with precision and flexibility. For details about other configurations attributes, please refer to the [LMBuilderConfig class documentation](https://github.com/TheFaheem/LMBuilder/blob/main/LMBuilder/lm_builder.py#L38).

## Contributing

Want to help us improve 🛠️LMBuilder and make LLM development more easy and efficient? We highly welcome contributions, Whether you're a seasoned developer or beginners alike and just getting started, your involvement in our open-source project is highly appreciated inspite of whether you want to report issues, bugs, submit pull requests, or provide feedback. [Learn more about how you can contribute here](https://github.com/TheFaheem/LMBuilder/blob/main/CONTRIBUTING.md).

## License

[![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

This work is licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa] - see the [LICENSE](https://github.com/TheFaheem/LMBuilder/blob/main/LICENSE) file for details.

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg

## Documentation

Still haven't done yet. Contribution needed!

## Support

For assistance, feedback, or inquiries related to 🛠️LMBuilder, We're here to help. Explore the options below to get the support you need:

- **Issue Tracker**: Encountered a bug or unexpected behaviours? Check our [Issue Tracker](https://github.com/TheFaheem/LMBuilder/issues) to report the issue or find solutions.

- **Discussion Forum**: Join our [Discussion](https://github.com/TheFaheem/LMBuilder/discussions) where you can inquire, collaborate, learn, sharing ideas, support the community.

- **Contact Us**: For sensitive or private matters, you can reach out directly via [mail](faheem.llmbuilder@gmail.com).

- **Feature Requests**: Have ideas for new features, improvements, or enhancements? Please visit our [Issue Tracker](https://github.com/TheFaheem/LMBuilder/issues) to share your suggestions. We value your input and appreciate your suggestions to make our tool even better.

We're committed to providing you with the best support possible, and we appreciate your feedback and contributions to the 🛠️LMBuilder community.

## Citation

This project is currently contributed by [Faheem](https://github.com/TheFaheem). If you find 🛠️LMBuilder helpful for your research or work, please cite:
```bibtex
@online{llmbuilder,
    author = {Mohammed Faheem},
    title = {🛠️LMBuilder: Facilitate Seamless Construction of Robust Large Language Models},
    url = {https://github.com/TheFaheem/LMBuilder},
    year = {2023},
    month = {Oct}
}
```
Your contribution is greatly appreciated, and we encourage you to acknowledge it when using 🛠️LMBuilder in your research or projects.

