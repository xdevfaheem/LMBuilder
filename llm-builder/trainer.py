# imports necessary packages
import torch_xla.core.xla_model as xm
import torch
import matplotlib.pyplot as plt
import wandb
import time
import math
import os
import logging
from contextlib import nullcontext

                                                                
class Trainer:
    
    def __init__(self,
                 train_loader,
                 val_loader,
                 wandb_log=False,
                 wandb_project="",
                 wandb_run_name="",
                 out_dir="./",
                 log_dir="./log",
                 logger=None,
                 global_step=0,
                 global_iter=0,
                 initial_iter=0,
                 best_val_loss=float(1000000.0),
                 losses_list=None
                 curr_epoch=0,
                 total_epochs=3,
                 scaler=None,
                 device=None,
                 ddp=False,
                 tpu_ddp = False,
                 pjrt_dist = True
                 decay_lr=False,
                 eval_interval=2000,
                 always_save_checkpoint=True,
                 model_args: dict = {},
                 eval_only=False,
                 gradient_accumulation_steps=5,
                 grad_clip=1.0,
                 ctx=nullcontext(),
                 log_interval=5,
                 max_iters=600000,
                 mastr_proc=False,
        ):

        """
        Initialize the Trainer.

        Args:
            train_loader (DataLoader): Training data loader.
            val_loader (DataLoader): Validation data loader.
            wandb_log (bool): Whether to log to WandB.
            wandb_project (str): Weights and Biases project name.
            wandb_run_name (str): Weights and Biases run name.
            out_dir (str): Output directory.
            log_dir (str): Log directory.
            logger (Logger): Logging object.
            global_step (int): Global step.
            global_iter (int): Global iteration.
            initial_iter (int): Initial iteration.
            best_val_loss (float): Best validation loss.
            losses_list (list): list of training loss
            curr_epoch (int): Current epoch.
            total_epochs (int): Total number of epochs.
            scaler: Scaler for gradient scaling.
            device: Device for training.
            ddp (bool): Use DistributedDataParallel for GPU training.
            tpu_ddp (bool): Use TPU DistributedDataParallel for training.
            pjrt_dist (bool): Use PJRT (PyTorch JIT-Ready Training) for TPU.
            decay_lr (bool): whether to Apply learning rate decay.
            eval_interval (int): Evaluation interval.
            always_save_checkpoint (bool): Always save model checkpoints.
            model_args (dict): Model configuration.
            eval_only (bool): Enable evaluation mode only.
            gradient_accumulation_steps (int): Number of gradient accumulation steps.
            grad_clip (float): Gradient clipping value.
            ctx: Context manager for auto mixed precision (amp) training.
            log_interval (int): Log interval.
            max_iters (int): Maximum iterations before stoppage.
            mastr_proc (bool): Flag to indicate the master process.
        """
            
        self.tbatch_genarator = train_loader
        self.vbatch_generator = val_loader
        self.wandb_log = wandb_log
        self.wandb_project = wandb_project
        self.wandb_run_name = wandb_run_name
        self.out_dir = out_dir
        self.log_dir = log_dir
        self.logger = logger
        self.global_step = global_step
        self.global_iter = global_iter
        self.initial_iter = initial_iter
        self.best_val_loss = best_val_loss
        self.losses_list = losses_list
        self.curr_epoch = curr_epoch
        self.total_epochs = total_epochs
        self.scaler = scaler
        self.device = device
        self.ddp = ddp
        self.tpu_ddp = tpu_ddp
        self.pjrt_dist = pjrt_dist
        self.decay_lr = decay_lr
        self.warmup_iters = warmup_iters
        self.learning_rate = learning_rate
        self.lr_decay_iters = lr_decay_iters
        self.min_lr = min_lr
        self.eval_interval = eval_interval
        self.always_save_checkpoint = always_save_checkpoint
        self.model_args = model_args
        self.eval_only = eval_only
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.grad_clip = grad_clip
        self.ctx = ctx
        self.log_interval = log_interval
        self.max_iters = max_iters=600000
        self.mastr_proc = mastr_proc=False
        self.epochs = epochs
        self.plot_dir =  os.path.join(self.config.log_dir, "plots")
        os.makedirs(self.plot_dir, exist_ok=True)
            
        # logging
        if self.wandb_log and self.mastr_proc:
            wandb.init(project=self.wandb_project, name=self.wandb_run_name, config=self.logging_config)
    
    # helps estimate an arbitrarily accurate loss over either split using many batches
    @torch.no_grad()
    def validate_model(self, model, global_step, epoch, plt_label="Loss", marker='o', linestyle='-'):
        
        """
        Estimate loss over train and val splits using many batches.

        Args:
            model: Model to validate.
            global_step (int): Global step.
            epoch (int): Current epoch.
            plt_label (str, optional): Label for the plot curve. Defaults to "Loss".
            marker (str, optional): Marker style for the data points. Defaults to 'o'.
            linestyle (str, optional): Line style for the curve. Defaults to '-'.
        Returns:
            out (dict): Validated loss dictionary.
        """
        
        # dict for saving the loss values
        out = {}
        model.eval()
        
        for split in ['train', 'val']:
            
            losses = torch.zeros(self.eval_iters)
            generator = iter(self.tbatch_genarator if split=="train" else self.vbatch_generator)
            
            for k, (X, Y) in enumerate(generator):
                if k > self.eval_iters:
                    break
                with self.ctx:
                    _, loss = model(X, Y)
                losses[k] = loss.item()
            
            out[split] = losses
            # save loss curves to visualize later
            self.save_loss_curve(range(1, self.eval_iters+1), losses, x_label="Evaluation Iterations", y_label=f"{split.capitalize()} Loss", plt_title=f"Evaluation Iterations vs {split.capitalize()} Loss", plt_label=plt_label, marker=marker, linestyle=linestyle, plot_name=f"E{epoch}_It{global_step}_S{split}_loss_curve")
            out[split] = out[split].mean()
            
        del model # free up memory
        return out
    

    # learning rate decay scheduler (cosine with warmup)
    def curr_lr(self, it, warmup_iters, lr_decay_iters, learning_rate, min_lr):
        
        """
        Learning rate decay scheduler (cosine with warmup).
    
        Args:
            it (int): Current iteration.
            warmup_iters (int): Number of warm-up iterations.
            lr_decay_iters (int): Number of iterations for learning rate decay.
            learning_rate (float): Initial learning rate.
            min_lr (float): Minimum learning rate.
    
        Returns:
            lr (float): Learning rate for the current iteration.
        """
        
        # Linear warm-up for warmup_iters steps
        if it < warmup_iters:
            lr = learning_rate * it / warmup_iters
            
        # If it > lr_decay_iters, return min learning rate
        elif it > lr_decay_iters:
            lr = min_lr
        
        # In between, use cosine decay down to min learning rate
        else:
            decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
            decay_coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
            lr = min_lr + decay_coeff * (learning_rate - min_lr)
    
        return lr

    
    def save_loss_curve(self, x, y, x_label, y_label, plt_title="X vs Y", plt_label="Plot", marker='o', linestyle='-', plot_name="plt"):
        
    """
    Save a loss curve as a plot.

    Args:
        x (list or array-like): x-axis data.
        y (list or array-like): y-axis data.
        x_label (str): Label for the x-axis.
        y_label (str): Label for the y-axis.
        plt_title (str, optional): Title for the plot. Defaults to "X vs Y".
        plt_label (str, optional): Label for the plot curve. Defaults to "Plot".
        marker (str, optional): Marker style for the data points. Defaults to 'o'.
        linestyle (str, optional): Line style for the curve. Defaults to '-'.
        plot_name (str, optional): Name for the saved plot file. Defaults to "plt".
    """

    # Create a plot for the loss curve
    plt.figure(figsize=(10, 5))
    plt.plot(x, y, label=plt_label, marker=marker, linestyle=linestyle)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(plt_title)
    plt.legend()
    plt.grid(True)
    
    # Save the plot as an image file
    plt.savefig(os.path.join(self.plot_dir, f'{plot_name}.png'))

        
    @staticmethod
    def configure_optimizers(weight_decay, learning_rate, betas, device_type):

        """
        Configure the optimizer based on provided parameters.

        Args:
            weight_decay (float): Weight decay.
            learning_rate (float): Learning rate.
            betas: Beta parameters for Adam optimizer.
            device_type (str): Type of training device.

        Returns:
            optimizer: Optimizer for training.
        """
        
        # first of all filter out all non-trainable parameters
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        self.logger.info(f"Number of tensors's weights will be decayed: {len(decay_params)}, with {num_decay_params:,} parameters")
        self.logger.info(f"Number of tensors's weights will not be decayed: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        
        # Create AdamW optimizer and use the fused version if it is available
        if device_type == "cpu":
          optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, weight_decay=weight_decay)

        elif device_type == "cuda":
            fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
            use_fused = fused_available and device_type == 'cuda'
            extra_args = dict(fused=True) if use_fused else dict()
            optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, weight_decay=weight_decay, **extra_args)
            
        elif device_type == "tpu": # using syncfree optimizer to avoid the additional sync between device and host.
            optimizer = syncfree.AdamW(optim_groups, lr=learning_rate, betas=betas, weight_decay=weight_decay)
        
        return optimizer
    
    def save_checkpoint(self, out_dir, model, optimizer, global_iter, global_step, best_val_loss, losses_list, epoch, scaler=None):

        """
        Save model checkpoint.

        Args:
            out_dir (str): Output directory.
            model: Model to be saved.
            optimizer: Optimizer.
            scaler: Scaler for gradient scaling.
            global_iter (int): Global iteration.
            global_step (int): Global step.
            best_val_loss (float): Best validation loss.
            epoch (int): Current epoch.
        """
        
        checkpoint_dir = os.path.join(out_dir, f"E{epoch}_It{global_iter}_ckpt.pt")
        checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'grad_scaler': scaler.state_dict(),
                    'model_args': self.config.model_args,
                    'global_step': global_step,
                    'global_iter': global_iter,
                    'best_val_loss': best_val_loss,
                    'loss' : losses_list,
                    'epoch': epoch,
                    'config': self.config,
        }
        self.logger.info(f"saving checkpoint to {checkpoint_dir}")
        torch.save(checkpoint, checkpoint_dir)
    
    def train(self, model, optimizer):

    """
    Train the model and log training progress.

    Args:
        model (torch.nn.Module): The neural network model to train.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
    """

    # Extract frequently used variables and initialize variables
    scaler = self.scaler  # Grad scaler
    initial_iter = self.initial_iter
    global_iter = self.global_iter  # Total number of iterations
    max_iters = self.max_iters  # Maximum number of iterations
    global_step = self.global_step  # Total model gradient adjustments made so far
    local_iter = 0  # Current iteration number

    # Initialize a list to store training losses
    train_losses = self.losses_list if self.losses_list is not None else []

    # Initialize the training data loader and set the model to training mode
    train_loader = iter(self.tbatch_genarator)
    model.train()

    t0 = time.perf_counter()

    # Iterate through epochs
    for epoch in epochs:

        # Logging for resuming training or starting anew
        if epoch != 0 and global_iter > 0:
            self.logger.info(f"Resuming training from epoch: {epoch} and iteration: {global_iter}")
        elif epoch == 0 and global_iter == 0:
            self.logger.info(f"Training Started!")
            self.logger.info(f"{'-'*25} Epoch:{epoch} {'-'*25}")

        for X, Y in train_loader:

            # Termination conditions
            if self.max_iters is not None and global_step >= self.max_iters:
                self.logger.info("Training Completed!")
                break

            # Determine and set the learning rate for this iteration
            lr = self.curr_lr(global_step, self.warmup_iters, self.lr_decay_iters, self.learning_rate, self.min_lr) if self.decay_lr else self.learning_rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # Evaluate the loss on train/val sets and write checkpoints
            if global_iter % self.eval_interval == 0 and self.mastr_proc:
                losses_dict = self.validate_model(model, global_iter, epoch)
                self.logger.info(f"Epoch: {epoch} Total Iters: {global_iter} Total Steps: {global_step} Train loss: {losses_dict['train']:.4f} Val loss: {losses_dict['val']:.4f}")
                if self.wandb_log:
                    wandb.log({
                        "iters": global_iter,
                        "step": global_step,
                        "train_loss": losses_dict['train'],
                        "val_loss": losses_dict['val'],
                        "lr": lr,
                    })
                if losses_dict['val'] < self.best_val_loss or self.always_save_checkpoint:
                    best_val_loss = losses_dict['val']
                    if global_iter > 0:
                        # Save a checkpoint with loss data
                        self.save_checkpoint(self.out_dir, model, optimizer, global_iter, global_step, best_val_loss, train_losses, epoch, scaler=scaler)
                if self.eval_only and global_step == 0:
                    logger.info(f"Training Stopped at {global_step}")
                    break

            # Forward backward update, with gradient accumulation to simulate larger batch size
            # Check if the gradients are accumulating or accumulated
            iter_t0 = time.perf_counter()
            is_accumulating = ((global_step + 1) % self.gradient_accumulation_steps != 0)  # +1 to prevent scale and backprop at gstep 0

            if is_accumulating:

                # This bloats the code with repeated code. Should be fixed by looking into this (https://pytorch.org/docs/master/_modules/torch/nn/parallel/distributed.html#DistributedDataParallel.no_sync) in the future.
                if self.ddp:
                    with torch.nn.parallel.DistributedDataParallel.no_sync():
                        with self.ctx:
                            logits, loss = model(X, Y)
                            loss = loss / self.gradient_accumulation_steps  # Scale the loss to account for gradient accumulation
                        if scaler:  # For GPU
                            # Backward pass, with gradient scaling if training in FP16
                            scaler.scale(loss).backward()
                        else:  # For TPU and CPU
                            loss.backward()
                else:
                    with self.ctx:
                        logits, loss = model(X, Y)
                        loss = loss / self.gradient_accumulation_steps  # Scale the loss to account for gradient accumulation
                    if scaler:  # For GPU
                        # Backward pass, with gradient scaling if training in FP16
                        scaler.scale(loss).backward()
                    else:  # For TPU and CPU
                        loss.backward()

            if not is_accumulating:

                # Clip the gradient
                if self.grad_clip != 0.0:
                    # Step the optimizer and scaler
                    if scaler:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)

                # Perform optimizer step based on device type
                if self.device_type == "gpu" and scaler is not None:  # For single GPU or multi-GPUs devices within DDP container
                    # Take optimizer step and update the scaling factor if training in FP16
                    scaler.step(optimizer)
                    scaler.update()
                elif self.device_type == "tpu" and self.pjrt_dist:  # For a single TPU host with pjrt
                    xm.optimizer_step()
                elif self.device_type == "tpu" and (not self.tpu_ddp and not self.pjrt_dist):  # For a single TPU core
                    xm.optimizer_step(optimizer, barrier=True)
                elif (self.device_type == "tpu" and self.tpu_ddp) or self.device_type == "cpu":  # For CPU device or single TPU with multi-core or multi-TPU devices with multicore wrapped within DDP container
                    optimizer.step()

                # Flush the gradients as soon as we can, no need for this memory anymore
                optimizer.zero_grad(set_to_none=True)
                global_step += 1  # Update (number of model parameter adjustments) steps

            # Update local and global iteration numbers
            global_iter += 1
            local_iter += 1

            # Timing and logging
            t1 = time.perf_counter()
            iter_time = t1 - iter_t0

            if global_iter % self.log_interval == 0:
                # Get loss as a float. Note: this is a CPU-GPU sync point.
                # Scale up to undo the division above, approximating the true total loss (exact would have been a sum)
                lossf = loss.item() * self.gradient_accumulation_steps
                train_losses.append(lossf)
                if local_iter >= 6:  # Let training settle a bit!
                    self.logger.info(f"Current Iteration: {local_iter}, Overall Iteration: {global_iter}, Total Steps: {global_step}, Loss: {lossf}, Time: {(iter_time * 1000):.2f}ms, Estimated Remaining Hours: {(((t1 - total_t0) / (global_iter - initial_iter)) * (max_iters - global_iter) / 3600):.2f} hours, Estimated Remaining Days: {(((t1 - total_t0) / (global_iter - initial_iter)) * (max_iters - global_iter) / 3600 / 24):.2f} days")

        # Save the training loss curve for the epoch
        self.save_loss_curve(range(1, len(train_losses) + 1), train_losses, x_label=f"Epoch{epoch}", y_label="Training loss", plt_title="One Epoch vs Training Loss", plt_label="Training loss curve", marker='o', linestyle='-', plot_name=f"epoch{epoch}_train_loss")
