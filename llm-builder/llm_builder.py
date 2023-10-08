import os
import time
import math
import pickle
from pathlib import Path
from contextlib import nullcontext
import numpy as np
import torch
import torch_xla
from torch.utils.data import Dataset, DataLoader, Sampler, BatchSampler
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
import torch_xla.distributed.parallel_loader as pl
import torch_xla.runtime as xr
from torch_xla.experimental import pjrt
import torch_xla.experimental.pjrt_backend # moved to torch_xla.runtime
import torch_xla.distributed.xla_backend
import torch_xla.utils.utils as xu
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.test.test_utils as test_utils
from torch_xla.amp import autocast, GradScaler
import sentencepiece as spm
from trainer import Trainer
from dataclasses import dataclass
from prepare_dataset import PrepareDataset
from tokenizer import Tokenizer
from typing import Tuple
try:
  from torch_xla.amp import syncfree
except ImportError:
  assert False, "Missing package syncfree; the package is available in torch-xla>=1.11. upgrade the library using `pip install torch-xla --upgrade`"
from model import GPTConfig, GPT


class LLMBuilderConfig:
    
    def __init__(self, dataset_config: dict, **kwargs):
        
        self.dataset_config = dataset_config
        for key, value in kwargs.items():
            setattr(self, key, value)

    out_dir = 'out'
    data_dir="./"
    model_name="test_model"
    data_prep_config = dict(
        hf_dataset="Bingsu/openwebtext_20p",
        from_disk=False,
        local_dataset_path=None,
        dataset_files= None,
        num_blocks=1024,
        train_data_percentage= 0.8,
        dset_prefix="tinystories",
        build_vocab=False,
        vocab_size=8000,
        vocab_type="yt",
        eos=1,
        bos=0,
        pad=2,
        unk= 3,
        
    )
    model_configs = dict(
        max_seq_len=1024,
        vocab_size=8000,
        dropout=0.1,
        bias=True,
        num_heads = 8,
        num_blocks = 4,
        d_model = 512,
        expansion_factor = 4,
        activation_type="relu",
        embedding_dim = 512,
        pretrained_embeddings = None,
        freeze_embeddings = False,
    )
    pl_kwargs = dict(
        batchdim=0,
        batches_per_execution=1,
        loader_prefetch_size=8,
        device_prefetch_size=4,
        host_to_device_transfer_threads=1
    )
    eval_interval = 2000
    total_epochs=2
    log_interval = 1
    eval_iters = 200
    eval_only = False
    always_save_checkpoint = True
    init_from = 'scratch'
    wandb_log = False
    wandb_project = 'gpt_builder'
    wandb_run_name = 'gpt2'
    batch_size_per_device = 12
    block_size=1024
    tpu_ddp=False
    pjrt_dist=True # Required for DDP on TPU v2/v3 when using PJRT.
    gradient_accumulation_steps=40
    num_samples = int(1e10)
    learning_rate = 6e-4
    max_iters = 600000
    weight_decay = 1e-1
    beta1 = 0.9
    beta2 = 0.95
    grad_clip_value = 1.0
    decay_lr=True
    warmup_iters = 2000
    lr_decay_iters = 600000
    min_lr = 6e-5
    backend = 'nccl'
    device_type = 'cpu'
    seed = 1337
    is_compile=True
    prepare_dataset=True

"""
Example usage:
  #TODO: get a single config yaml file as an arg instead of all attributes
  gptconfig = GPTBuilderConfig({"train": [("tinystories", 1.0)], "test": [("tinystories", 1.0)]})
"""

class LLMBuilder:
    
    
    def __init__(self, builder_config):
        
# -------------------------------------configurations----------------------------------------------

        self.out_dir = Path(builder_config.out_dir)
        self.model_name = builder_config.model_name
        self.data_dir = builder_config.data_dir
        self.dataset_dir = os.path.join(self.data_dir, "dataset_files")
        self.model_args = builder_config.model_configs
        self.paraloader_kwargs = pl_kwargs
        self.eval_interval = builder_config.eval_interval
        self.num_epochs = builder_config.total_epochs
        self.log_interval = builder_config.log_interval
        self.eval_iters = builder_config.eval_iters
        self.eval_only = builder_config.eval_only
        self.always_save_checkpoint = builder_config.always_save_checkpoint
        self.init_from = builder_config.init_from
        self.wandb_log = builder_config.wandb_log
        self.wandb_project = builder_config.wandb_project
        self.wandb_run_name = builder_config.wandb_run_name
        self.tpu_ddp = tpu_ddp
        self.pjrt_dist = pjrt_dist
        self.batch_size_per_device = builder_config.batch_size_per_device
        self.gradient_accumulation_steps = builder_config.gradient_accumulation_steps
        self.seed = builder_config.seed
        self.block_size = builder_config.block_size
        self.learning_rate = builder_config.learning_rate
        self.max_iters = builder_config.max_iters
        self.weight_decay = builder_config.weight_decay
        self.beta1 = builder_config.beta1
        self.beta2 = builder_config.beta2
        self.grad_clip_value = builder_config.grad_clip_value
        self.decay_lr = builder_config.decay_lr
        self.warmup_iters = builder_config.warmup_iters
        self.lr_decay_iters = builder_config.lr_decay_iters
        self.min_lr = builder_config.min_lr
        self.backend = builder_config.backend
        self.device_type = builder_config.device_type
        self.compile= builder_config.is_compile
        self.dataset_preparation_config = builder_config.data_prep_config
        self.prepare_dataset = builder_config.prepare_dataset
        self.train_data_config = builder_config.dataset_config["train"]
        self.val_data_config = builder_config.dataset_config["validation"]
        self.setup()
        
    def setup(self):
        
        self.logger = self._configure_logging(log_dir, "setup")
        train_logger = self._configure_logging(log_dir, "training")
        
        self.logger.info("Setting up host device...")
        device = torch.cuda.current_device() if self.device_type == 'gpu' else xm.xla_device() if self.device_type == 'tpu' else torch.device('cpu') if self.device_type == 'cpu' else None
        assert device is not None, "Unsupported device!"
        dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
        self.logger.info(f"Device: {device}, Datatype: {dtype}")

        # init these up here, can override if init_from='resume' (i.e. from a checkpoint)
        global_iter=0
        initial_iter=0
        global_step=0
        best_val_loss = float("inf")
        current_epoch=0

        if self.device_type == "gpu":

            # various inits, derived attributes, I/O setup
            if ddp := int(os.environ.get('RANK', -1)) != -1:
                self.logger.info("Training on multiple GPU!")
                init_process_group(backend=backend)
                rank = int(os.environ['RANK'])
                local_rank = int(os.environ['LOCAL_RANK'])
                world_size = int(os.environ['WORLD_SIZE'])
                device = f'cuda:{local_rank}'
                torch.cuda.set_device(device)
                master_process = (rank==0) # this process will do logging, checkpointing etc.
                seed_offset = rank # each process gets a different seed
                # world_size number of processes will be training simultaneously, so we can scale
                # down the desired gradient accumulation iterations per process proportionally
            
            else:# if not ddp, we are running on a single gpu, and one process
                master_process = True
                rank = 0
                local_rank = 0
                device = f'cuda:{local_rank}'
                torch.cuda.set_device(device)
                self.logger.info("Training on single GPU!")
                seed_offset = rank
                world_size = 1

        elif self.device_type == "tpu":

            #for more info check out here -> https://github.com/pytorch/xla/blob/master/docs/pjrt.md
            
            # distributed training over single host with muti device (eg., TPU-v3 has 4 chip (device) with two core each)
            if self.pjrt_dist:
                self.logger.info("Training on single tpu host!")
                # Recommended: set PJRT_DEVICE to your local device type
                os.environ['PJRT_DEVICE'] = 'TPU'
                rank = xm.get_ordinal()
                world_size = xm.xrt_world_size()
                dist.init_process_group('xla', init_method='xla://') # The xla:// init_method automatically finds replica IDs, world size, and master IP by querying the runtime.
                master_process = (rank==0)
                seed_offset = rank

            # distributed training in multi host tpu devices with muti chips
            elif self.tpu_ddp:
                self.logger.info("Training on multi tpu host!")
                os.environ['XRT_TPU_CONFIG'] = 'localservice;0;localhost:51011'
                os.environ['MASTER_ADDR'] = 'localhost'
                os.environ['MASTER_PORT'] = '12355'
                rank = xm.get_ordinal()
                world_size = xm.xrt_world_size()
                dist.init_process_group('xla', world_size=world_size, rank=rank)
                master_process = (rank==0)
                seed_offset = rank

            # training on single tpu core
            else:
                self.logger.info("Training on single tpu core!")
                master_process = xm.is_master_ordinal()
                rank = xm.get_ordinal()
                world_size = xm.xrt_world_size()
                seed_offset = rank
                
        elif self.device_type == "cpu":

            self.logger.info("Training on single cpu device!")
            master_process = True
            rank = 0
            seed_offset = rank
            world_size = 1

        else:
            raise NotImplementedError("No other device types supported currently!")

        
        assert ((self.batch_size_per_device * world_size) // world_size) == self.batch_size_per_device, "batch size per gpu should be divisible by world size"
        self.batch_size = self.batch_size_per_device
        assert self.gradient_accumulation_steps % world_size == 0 and self.gradient_accumulation_steps > 0, "gradient accumulation step should be divisible by world size and should be greater than 1."
        self.gradient_accumulation_steps //= world_size

        tokens_per_iter = self.gradient_accumulation_steps * world_size * self.batch_size * self.block_size
        self.logger.info(f"Total Tokens Per Iteration : {tokens_per_iter:,}")

        if master_process:
            # model out directory
            self.logger.info("Setting up model and log directories")
            model_dir = os.path.join(self.out_dir.absolute().resolve(), self.model_name)
            os.makedirs.(model_dir, exist_ok=True)
            log_dir = os.path.join(Path(model_dir).absolute().resolve(), "log")
            os.makedirs.(log_dir, exist_ok=True)
        torch.manual_seed(self.seed + seed_offset)
        torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
        torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn

        # note: float16 data type will automatically use a GradScaler
        ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
        ctx_mnr = torch.cpu.amp.autocast(device_type=device, dtype=torch.bfloat16) if self.device_type == 'cpu' else torch.cuda.amp.autocast(device_type=device, dtype=ptdtype) if self.device_type == 'cuda' else torch_xla.amp.autocast(device) if self.device_type == 'tpu' else None
        assert ctx_mnr is not None, "Device type: {self.device_type} is not supported yet!"
        # checkout here for more info about xla autocast-> https://pytorch.org/docs/stable/amp.html#automatic-mixed-precision-package-torch-amp
        
        if not os.path.exists(os.path.join(self.dataset_dir, 'train')) and not os.path.exists(os.path.join(self.dataset_dir, 'val')):
            self.logger("Dataset directory doen't exist!")
            #y_n = str(input("Whether to start preparing the dataset now or not (y/n): "))
            if self.prepare_dataset:
                self.logger("Starting to prepare the dataset...")
                prep_dataset = PrepareDataset(self.data_dir, **self.dataset_preparation_config)
                train_data_dir, val_data_dir = prep_dataset.prepare(self.block_size, num_blocks=self.dataset_preparation_config["num_blocks"])
                assert os.path.exists(os.path.join(self.dataset_dir, 'train')) and os.path.exists(os.path.join(self.dataset_dir, 'val')), "The path to the dataset files (either train or val) directory doesn't exist, Make sure the dataset_folder and arguments given to `data_prep_dict` is correct."                
            else:
                sys.exit(1)
            
        else:
            train_data_dir = os.path.join(self.dataset_dir, 'train')
            val_data_dir = os.path.join(self.dataset_dir, 'val')
            
        self.logger.info("Setting up the data loader...")
        train_dataloader, val_dataloader = self.create_dataloaders(
                                                self.batch_size,
                                                self.block_size,
                                                train_data_dir,
                                                world_size,
                                                rank,
                                                device,
                                                val_data_dir=val_data_dir,
                                                seed=(self.seed + seed_offset)
        )
        if self.device_type == "tpu":
            train_dataloader, val_dataloader = train_dataloader.per_device_loader(device), val_dataloader.per_device_loader(device)
        
        self.logger.info("Setting up the tokenizer.")
        # Get the Vocablary Size of the dataset
        if self.dataset_preparation_config["build_vocab"]:
            
            if os.path.exists(vocab_path := glob.glob(os.path.join(self.data_dir, "*.model"))[0]):
                sp = True if self.dataset_preparation_config["vocab_type"] == "sp" else False
                yt = True if self.dataset_preparation_config["vocab_type"] == "yt" else False
                tokenizer = Tokenizer(model_path=vocab_path, youtokenizer=yt, sp_tokenizer=sp)
                if self.tokenizer.backend == "youtoken":
                    setattr(self.tokenizer, "eos_id", self.dataset_preparation_config["eos"])
                    setattr(self.tokenizer, "bos_id", self.dataset_preparation_config["eos"])
                vocab_size = tokenizer.vocab_size()
            else:
                raise ValueError("Tokenizer model doesn't exist. Check if the tokenizer is trained correctly!")
        
        else:
            # TODO: custom encoding
            tokenizer = Tokenizer(tiktokenizer=True)
            vocab_size = tokenizer.vocab_size()

        self.logger.info("Setting up the model...")
        #model_args = dict(num_block=self.num_block, num_heads=self.num_heads, d_model=self.d_model, block_size=self.block_size, bias=self.bias, vocab_size=vocab_size, dropout=self.dropout) # start with model_args from command line

        if self.init_from == 'scratch':
            # initialize a new model from scratch
            self.logger.info("\nInitializing a new model from scratch...")
            # determine the vocab size we'll use for from-scratch training
            self.model_args["vocab_size"] = vocab_size
            gptconfig = GPTConfig(**self.model_args)
            self.model = GPT(gptconfig)

        elif self.init_from == 'resume':
            # resume training from a checkpoint.
            self.logger.info(f"Resuming training from {self.model_dir}")
            ckpt_path = sorted(Path(self.model_dir).glob("*.pt"))[-1]
            checkpoint = torch.load(ckpt_path, map_location=device)
            checkpoint_model_args = self.checkpoint['model_args']
            # force these config attributes to be equal otherwise we can't even resume training
            # the rest of the attributes (e.g. dropout) can stay as desired from command line
            for k in ['max_seq_len', 'vocab_size', 'dropout', 'bias', 'num_heads', 'num_blocks', 'd_model', 'expansion_factor', 'activation_type', 'embedding_dim', 'pretrained_embeddings', 'freeze_embeddings']:
                self.model_args[k] = checkpoint_model_args[k]
            # init the model
            gptconfig = GPTConfig(**self.model_args)
            self.model = GPT(gptconfig)
            state_dict = checkpoint['model']
            # fix the keys of the state dictionary :(
            # honestly no idea how checkpoints sometimes get this prefix, have to debug more
            unwanted_prefix = '_orig_mod.'
            for k,v in list(state_dict.items()):
                if k.startswith(unwanted_prefix):
                    state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
            model.load_state_dict(state_dict)
            global_iter = checkpoint['global_iter'] # global iteration. this will be updated during training
            initial_iter = checkpoint['global_iter'] # initial iteration. this will stay the same until training stops to identify no. of iteration in one training life
            global_step = checkpoint['global_step']
            best_val_loss = checkpoint['best_val_loss']
            current_epoch = checkpoint['curr_epoch']

        else:
            self.logger.error("No other model initialization is currently supported!")

        self.logger.info("Moving the model to host device...")
        self.model.to(device) # move the initialized model to host device

        if self.device_type == "tpu" and xr.using_pjrt()):
            xm.broadcast_master_param(self.model)

        # compile the model
        if self.compile:
            self.logger.info("Compiling the model... (might take a ~minute)")
            unoptimized_model = self.model
            if hasattr(torch, 'compile'):
                self.model = torch.compile(unoptimized_model) # requires >= PyTorch 2.0 
                unoptimized_model = None # free up the memory
            else:
                self.logger.warning("WARNING: your current pytorch version does'nt have compile method. continuing setup without compiling the model...")
                #sys.exit(1)
        
        # wrap model into DDP container
        if self.ddp:
            self.model = DDP(self.model, device_ids=[local_rank])
        elif self.tpu_ddp:
            self.model = DDP(self.model, gradient_as_bucket_view=True)
        
        self.model = self.model.module if self.ddp else self.model # plug the model from ddp container if training over multi-gpu setup or keep the model as it is.

        # initializing gradient scaler
        self.logger.info("Setting up gradient scaler...")
        
        if self.device_type == 'tpu' or self.device_type == 'cpu':
            scaler = None
        elif self.device_type == 'cuda':
            # using GradScaler if data type is float16 otherwise there will be no-ops
            scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'), use_zero_grad=True)

        # loading scaler state dict
        if self.init_from == 'resume' and (self.device_type != 'tpu' and self.device_type != 'cpu'):
            scaler.load_state_dict(self.checkpoint['grad_scaler'])

        # initializing the optimizer
        self.logger.info("Setting up the optimizer..")
        self.optimizer = Trainer.configure_optimizers(self.weight_decay, self.learning_rate, (self.beta1, self.beta2), self.device_type)

        # loading optimizer state dict
        if self.init_from == 'resume':
            self.optimizer.load_state_dict(self.checkpoint['optimizer'])
        checkpoint = None # free up memory
        
        # initialiting the Trainer class instance
        self.logger.info("Setting up the trainer...")
        self.trainer = Trainer(
                trainer_cfg,
                train_dataloader,
                val_dataloader,
                wandb_log=self.wandb_log,
                wandb_project=self.wandb_project,
                wandb_run_name=self.wandb_run_name,
                out_dir=model_dir,
                log_dir=log_dir,
                logger=train_logger
                global_step=global_step,
                global_iter=global_iter,
                initial_iter=initial_iter
                best_val_loss=best_val_loss,
                curr_epoch=current_epoch,
                total_epochs=self.num_epochs,
                scaler=scaler,
                device=device,
                ddp=ddp,
                tpu_ddp = self.tpu_ddp,
                decay_lr=self.decay_lr,
                eval_interval=self.eval_interval,
                always_save_checkpoint=self.always_save_checkpoint,
                model_args=self.model_args,
                eval_only=self.eval_only,
                gradient_accumulation_steps=self.gradient_accumulation_steps,
                grad_clip=self.grad_clip_value,
                ctx=ctx_mnr,
                log_interval=self.log_interval,
                max_iters=self.max_iters,
                mastr_proc=master_process,
            )
            
    def _configure_logging(self, log_dir, file_name):
        # Create a logger
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)  # Set the lowest log level (DEBUG)

        # Create a formatter for log messages
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        formatter = logging.Formatter(log_format)

        # Create a file handler to save logs to a file
        log_file = os.path.join(log_dir, f"{file_name}.log")
        file_handler = logging.RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=3)  # Rotate after 10MB
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)  # Log all messages to the file

        # Create a console handler for displaying log messages on the console
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.INFO)  # Log INFO and above to the console

        # Add the handlers to the logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def create_dataloader(
        self,
        batch_size: int,
        block_size: int,
        num_chunks,
        dataset_dir: Path,
        n_process: int,
        proc_rank: int,
        device: torch.device,
        shuffle: bool = True,
        seed: int = 12345,
        split="train",
    ) -> DataLoader:

        datasets = []
        wts = []
        data_config = self.train_data_config if split == "train" else self.val_data_config if split == "val" else None
        assert data_config is not None
        
        for prefix, weight in data_config:
            filenames = glob.glob(os.path.join(dataset_dir,f"{prefix}*"))
            random.seed(seed)
            random.shuffle(filenames)

            dataset = PackedDataset(
                filenames,
                # n_chunks control the buffer size. 
                # Note that the buffer size also impacts the random shuffle
                # (PackedDataset is an IterableDataset. So the shuffle is done by prefetch a buffer and shuffle the buffer)
                n_chunks=num_chunks,
                block_size=block_size,
                shuffle=shuffle,
                seed=seed,
                num_processes=n_process,
                process_rank=proc_rank
                )
            datasets.append(dataset)
            wts.append(weight)

        if not datasets:
            raise RuntimeError(f"No data found at {data_dir}. Make sure everything worked perfectly while creating the dataset.")

        sum_weights = sum(wts)
        weights = [el / sum_weights for el in wts]
        
        def collater(batch):
            x = torch.stack([arr[:-1] for arr in batch])
            y = torch.stack([arr[1:] for arr in batch])
            if self.device_type == 'cuda':
                #pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
                x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
            else:
                x, y = x.to(device), y.to(device)
            return x, y

        combined_dataset = CombinedDataset(datasets=datasets, seed=seed, weights=weights)
        data_loader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=False, collate_fn=collater, drop_last=True)
        
        if self.device_type == "tpu":
            return pl.ParallelLoader(data_loader, device, **self.paraloader_kwargs)
        elif self.device_type == "cuda" or self.device_type == "cpu":
            return data_loader
        

    def create_dataloaders(
                self,
                batch_size: int,
                block_size: int,
                train_data_dir: Path,
                num_process: int,
                process_rank: int,
                device: torch.device,
                val_data_dir: Optional[Path] = None,
                seed: int = 12345,
                ) -> Tuple[DataLoader, DataLoader]:
        
        # Increase by one because we need the next token as well
        effective_block_size = block_size + 1
        train_dataloader = self.create_dataloader(
                    batch_size,
                    effective_block_size,
                    8,
                    train_data_dir,
                    num_process,
                    process_rank,
                    device,
                    shuffle=True,
                    seed=seed,
                    split="train"
                )
        val_dataloader = (
                self.create_dataloader(
                    batch_size,
                    effective_block_size,
                    2,
                    val_data_dir,
                    num_process,
                    process_rank,
                    device,
                    shuffle=True,
                    seed=seed,
                    split="val"
                ) if val_data_dir else None
                )
        return train_dataloader, val_dataloader
        
    def build(self):
        self.logger.info("Firing up the training...")
        tloss = self.trainer.train(self.model, self.optimizer)
