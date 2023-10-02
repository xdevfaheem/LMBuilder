
import os
import time
import math
import pickle
from contextlib import nullcontext
import numpy as np
import torch
import torch_xla
from torch.utils.data import Dataset, DataLoader, Sampler, BatchSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch_xla.distributed.parallel_loader as pl
import torch_xla.utils.utils as xu
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.test.test_utils as test_utils
from torch_xla.amp import autocast, GradScaler
import sentencepiece as spm
from trainer import Trainer
from dataclass import dataclass
try:
  from torch_xla.amp import syncfree
except ImportError:
  assert False, "Missing package syncfree; the package is available in torch-xla>=1.11. upgrade the library using `pip install torch-xla --upgrade`"
from model import GPTConfig, GPT

# -------------------------------------configurations-------------------------------------------------
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'gpt_builder'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
# data
dataset = 'openwebtext_20p'
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
num_samples = int(1e10)
# model
num_block = 4
num_heads = 8
d_model = 512
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip_value = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = torch.device('cuda') if torch.cuda.is_available() else xm.xla_device()  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', 'tpu' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster

# if not ddp, we are running on a single gpu, and one process
master_process = True
seed_offset = 0
ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"Total Tokens Per Iteration : {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'gpu' if 'cuda' in device else 'tpu' if 'tpu' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx_mnr = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == 'gpu' else torch_xla.amp.autocast(device) if device_type == 'tpu' else None
assert ctx_mnr is not None, "Unsupported device"

# daaata looader...
# Torch Dataset
class CustomDataset(Dataset):
    
    def __init__(self, mmap_path, block_size):
        self.block_size = block_size
        self.data = np.memmap(mmap_path, dtype=np.uint16, mode="r")

    def __len__(self):
        return len(self.data) // self.block_size

    def __getitem__(self, idx):
        z = self.data[idx:idx+self.block_size+1].astype(np.int64)
        return z
    
data_dir = os.path.join('data', dataset)
# Create dataset
train_dataset = CustomDataset(os.path.join(data_dir, 'openwebtext_20p_train.bin'), block_size)
val_dataset = CustomDataset(os.path.join(data_dir, 'openwebtext_20p_val.bin'), block_size)

# collator function to batch up
def collate_fn(batch):
    x = torch.stack([torch.from_numpy(item[:-1]) for item in batch])
    y = torch.stack([torch.from_numpy(item[1:]) for item in batch])
    return x, y

# Custom Sampler
class CustomSampler(Sampler):
    
    def __init__(self, data, total_samples, block_size):
        self.data = data
        self.block_size = block_size
        self.num_samples = total_samples
        seed = int(torch.empty((), dtype=torch.int64).random_().item())
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)

    def __iter__(self):
        high = (len(self.data) - (self.block_size+1)) # as we fetch 1 more than block size in __getitem__
        for _ in range(self.num_samples // 32):
            yield from torch.randint(high=high, size=(32,), dtype=torch.int64, generator=self.generator).tolist()
        yield from torch.randint(high=high, size=(self.num_samples % 32,), dtype=torch.int64, generator=self.generator).tolist()

    def __len__(self):
        return self.num_samples

# batch sampler using our custom sampler
train_batch_sampler = BatchSampler(CustomSampler(train_dataset.data, num_samples, block_size=block_size), batch_size=batch_size, drop_last=True)
val_batch_sampler = BatchSampler(CustomSampler(val_dataset.data, num_samples, block_size=block_size), batch_size=batch_size, drop_last=True)

# Create DataLoader
train_dataloader = DataLoader(train_dataset, batch_sampler=train_batch_sampler, collate_fn=collate_fn, pin_memory=True)
val_dataloader = DataLoader(val_dataset, batch_sampler=val_batch_sampler, collate_fn=collate_fn, pin_memory=True)

if device_type == 'cuda' or device_type == 'cpu':
    pass
elif device_type == 'tpu':
    train_dataloader = pl.MpDeviceLoader(train_dataloader, device)
    val_dataloader = pl.MpDeviceLoader(val_dataloader, device)
    
# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
total_iter_num = 0
best_val_loss = 1e9

vocab_path = os.path.join(data_dir, 'openwebtext_20p.model')
try:
    vocab = spm.SentencePieceProcessor(model_file=vocab_path)
    vocab_size = vocab.get_piece_size()
except:
    vocab = tiktoken.get_encoding("gpt2")
    vocab_size = None

# model init
model_args = dict(num_block=num_block, num_heads=num_heads, d_model=d_model, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout) # start with model_args from command line

if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = vocab_size if vocab_size is not None else 50304
    gptconfig = GPTConfig(**model_args)
    model = GPT(gptconfig)
        
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['num_block', 'num_heads', 'd_model', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    total_iter_num = checkpoint['total_iter_num']
    best_val_loss = checkpoint['best_val_loss']
    
else:
    print("No other Model Initialization is Currently Supported")

# move the initialized model to host device
model.to(device)

if device_type == 'tpu':
    scaler = None
elif device_type == 'gpu':
    # GradScaler only used for GPU
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
else:
    print("Only TPU or GPU supported for AMP.")
sys.exit(1)

@dataclass
class TrainerConfig:
    wandb_log=wandb_log
    wandb_project=wandb_project
    wandb_run_name=wandb_run_name
    total_iter_num=total_iter_num
    scaler=scaler
    decay_lr=decay_lr
    learning_rate=learning_rate
    eval_interval=eval_interval
    best_val_loss=best_val_loss
    always_save_checkpoint=always_save_checkpoint
    model_args=model_args
    eval_only=eval_only
    gradient_accumulation_steps=gradient_accumulation_steps
    grad_clip=grad_clip_value
    ctx=ctx_mnr
    log_interval=log_interval
    max_iters=max_iters
trai_cfg = TrainerConfig

# initialiting the Trainer class instance
trainer = Trainer(trai_cfg, train_dataloader, val_dataloader)
# initiating the optimizer
optimizer = trainer.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)

if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (might take a ~minute)")
    unoptimized_model = model
    if hasattr(torch, 'compile'):
        model = torch.compile(unoptimized_model) # requires PyTorch 2.0
    else:
        print("pytorch does'nt have compile attribute. upgrade the pytorch package and try again!")
        
trainer.train(model, optimizer)
