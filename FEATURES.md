# List of Features and Capabilities offered by 🛠️LLM Builder.

## Customizable Architecture

With LLM Builder, you're not bound by a pre-built architecture. We provide you our pre-defined modules which you can use to craft your own language model architecture, tailored to your research objectives. You can either use pre-built model architecture or you can customize your model using pre-defined modules or you can use your own custom module in your model architecture.

- **Transformative Model Configurations**: Experiment with different model configurations, from transformer-based architectures to hybrid models that combine various components either pre-defined modules or your own modules. Check out [contribution guide](https://github.com/TheFaheem/llm-builder/blob/main/CONTRIBUTION.md) to contribute your custom module or architecture for other to use. 

## Effortless Dataset Preparation

LLM Builder's `PrepareDataset` class simplifies the process of preprocessing, training tokenizer and preparing your dataset seamlessly either concurrently or parallely, allowing you to focus on the quality and curation of your data, rather than the complexities of data preparation.

- **Dataset Transformation**: Convert your raw data into the format required to pre-train your Large Language Model. LLM Builder handles everything, from text tokenization, data proportionality (you can control the propotionality of the specific set within your dataset) to creating data pipelines (data iterators and loaders) with prefetch for efficient training.

## Versatile Training Optimization

Train your model for optimal results and efficiency with llm-builder's range of training optimization techniques.

- **Adaptive Training Strategies**: Experiment with various training strategies such as learning rate schedules, gradient clipping, and distributed training. Our library adapts and supports your requirements, allowing you achieve the best possible model performance. Check out [contribution guide](https://github.com/TheFaheem/llm-builder/blob/main/CONTRIBUTION.md) to contribute your desired training optimization technique which you don't find and you know how to code and where to plug that in and help other <!fork this repo incorporate that specific funtionality where it needs to be>

- **Fine-Tune for Excellence**: Coming soon... <!Achieving optimal results often involves extensive fine-tuning. LLM Builder streamlines this process, offering a range of training optimization techniques. Whether you need to improve the model's accuracy, reduce training time, or optimize hyperparameters, our platform provides the necessary tools and guidance.>

## Multi-Device Training

Liberated from hardware constraints, LLM Builder accommodates a variety of hardware configurations, from single CPUs and GPUs to TPUs and multi-device setups.

- **Flexible Hardware Support**: Train on single CPU, GPU, TPU hosts or TPU cores, and even orchestrate training across multiple GPUs and TPU hosts. Our library seamlessly accommodates diverse hardware environments.

- **Efficient Scaling**: Utilize the power of distributed training on multi-GPU setups and TPU clusters to train your models efficiently. LLM Builder simplifies the process of scaling your training across multiple devices (upto 2048 TPU chips if training on TPU cluster), enabling you to take full advantage of your hardware resources.

## Checkpointing and Resuming

Auto checkpointing ensures that you can save and resume training with ease.

**Seamless Progress Saving**: Auto checkpointing (`save_every` attribute in `LLMBuilderConfig`) allows you to save the model's state, also including the current epoch, global iteration and step, loss, current best validation loss, model configs (rebuild the model as same as from the scratch), optimizer's statedict and scaler's (if training on cuda-enabled GPU) state dict at desired intervals. You can easily resume training from the exact point where you left off, ensuring that no progress is lost in case of interruptions or system failures.

**Precise Model Recovery**: Resume training with confidence by loading the model's state dictionary from the last checkpoint. LLM Builder ensures that you can pick up right where you left off, saving you valuable time and resources

## Comprehensive Logging

Gain deeper insights into your model's performance with our comprehensive logging features.

- **In-Depth Progress Tracking**: LLM Builder meticulously logs every aspect of your model development by keeping a detailed record of training progress, hyperparameter settings, and evaluation metrics. This data allows for easy analysis later.

- **Visualized Model Understanding**: Enhance your understanding of your model's performance by logging training statistics and visualizing loss curves. llm-builder equips you with the insights needed to make informed decisions throughout the training process.

