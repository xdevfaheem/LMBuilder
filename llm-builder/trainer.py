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
        self.curr_epoch = curr_epoch
        self.total_epochs = total_epochs
        self.scaler = scaler
        self.device = device
        self.ddp = ddp
        self.tpu_ddp = tpu_ddp
        self.pjrt_dist = pjrt_dist
        self.decay_lr = decay_lr
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
    def validate_model(self, model, global_step, epoch):
        
        """
        Estimate loss over train and val splits using many batches
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
        self.save_eval_loss_curves(out, global_step, epoch)
        
        out["train"] = out["train"].mean()
        out["val"] = out["val"].mean()
        
        del model # free up memory
        return out
    
    # learning rate decay scheduler (cosine with warmup)
    def get_lr(self, it):
        
        """
        Learning rate decay scheduler (cosine with warmup)
        """
        
        # 1) linear warmup for warmup_iters steps
        if it < warmup_iters:
            return learning_rate * it / warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        if it > lr_decay_iters:
            return min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
        return min_lr + coeff * (learning_rate - min_lr)
    
    def save_eval_loss_curves(self, loss_data, global_step, epoch):
        
        """
        Save loss curves for visualization
        """

        train_losses = loss_data['train']
        val_losses = loss_data['val']

        eval_iters = range(1, len(train_losses) + 1)  # Assuming len(train_losses) == len(val_losses)

        # Plot training loss
        plt.figure(figsize=(10, 5))
        plt.plot(eval_iters, train_losses, label='Train Loss', marker='o', linestyle='-')
        plt.xlabel('Evaluation Iterations')
        plt.ylabel('Train Loss')
        plt.title('Evaluation Iterations vs Training Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.plot_dir, f'E{epoch}_{global_step}_train_loss_curve.png'))
        #plt.show()

        # Plot evaluation loss
        plt.figure(figsize=(10, 5))
        plt.plot(eval_iters, val_losses, label='Validation Loss', marker='o', linestyle='-')
        plt.xlabel('Evaluation Iterations')
        plt.ylabel('Validation Loss')
        plt.title('Evaluation Iterations vs Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.plot_dir, f'E{epoch}_{global_step}_val_loss_curve.png'))
        #plt.show()

    @classmethod
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):

        """
        Configure the optimizer based on provided parameters
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
    
    def save_checkpoint(self, out_dir, model, optimizer, scaler, global_iter, global_step, best_val_loss, epoch):

        """
        Save model checkpoint
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
                    'epoch': epoch,
                    'config': self.config,
        }
        self.logger.info(f"saving checkpoint to {checkpoint_dir}")
        torch.save(checkpoint, checkpoint_dir)
    
    def train(self, model, optimizer):

        """
        Train the model
        """
        
        # changing the scope of some frequently used variable into local
        scaler = self.scaler # grad scaler
        initial_iter=self.initial_iter
        global_iter=self.global_iter # total number of iteration
        max_iters=self.max_iters # maximum number of iterations
        global_step=self.global_step # total model grad adjustment made so far
        local_iter=0 # curent iteration number
        
        t0 = time.perf_counter()
        
        # initialising CombinedDatasetIterator ( check `__iter__` method in CombinedDataset)
        train_loader = iter(self.tbatch_genarator)
        model.train() # change the model to training mode
        
        for epoch in epochs:
            
            if epoch != 0 and global_iter > 0:
                self.logger.info(f"Resuming training from epoch: {epoch} and iteration: {global_iter}")
            
            elif epoch == 0 and global_iter == 0:
                self.logger.info(f"Training Started!")
                self.logger.info(f"{'-'*25} Epoch:{epoch} {'-'*25}")
            
            for X, Y in train_loader:
            
                # termination conditions
                if self.max_iters is not None and global_step >= self.max_iters:
                    self.logger.info("Training Completed!")
                    break

                # determine and set the learning rate for this iteration
                lr = self.get_lr(global_step) if self.decay_lr else self.learning_rate
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

                # evaluate the loss on train/val sets and write checkpoints
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
                            self.save_checkpoint(self.out_dir, model, optimizer, scaler, global_iter, global_step, best_val_loss, epoch)
                    if self.eval_only and global_step == 0:
                        logger.info(f"Training Stopped at {global_step}")
                        break

                # forward backward update, with gradient accumulation to simulate larger batch size
                # check if the gradients are accumulating or accumulated
                iter_t0 = time.perf_counter()
                is_accumulating = ((global_step+1) % self.gradient_accumulation_steps != 0 # +1 to prevent scale and backprop at gstep 0
                
                if is_accumulating:
                    
                    # This bloats the code with repeated code. should be fixed by looking into this (https://pytorch.org/docs/master/_modules/torch/nn/parallel/distributed.html#DistributedDataParallel.no_sync) in the future insha allah
                    if self.ddp:
                        #https://github.com/pytorch/pytorch/issues/43201#issue-680863643
                        with torch.nn.parallel.DistributedDataParallel.no_sync()
                            with self.ctx:
                                logits, loss = model(X, Y)
                                loss = loss / self.gradient_accumulation_steps # scale the loss to account for gradient accumulation
                            if scaler: # for GPU
                                # backward pass, with gradient scaling if training in fp16
                                scaler.scale(loss).backward()
                            else: # for TPU and CPU
                                loss.backward()
                    else:
                        with self.ctx:
                                logits, loss = model(X, Y)
                                loss = loss / self.gradient_accumulation_steps # scale the loss to account for gradient accumulation
                            if scaler: # for GPU
                                # backward pass, with gradient scaling if training in fp16
                                scaler.scale(loss).backward()
                            else: # for TPU and CPU
                                loss.backward()
                
                if not is_accumulating:                   
                    
                    # clip the gradient
                    if self.grad_clip != 0.0:
                        # step the optimizer and scaler 
                        if scaler:
                            scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
                    
                    # optimizer step

                    # for gpu with grad scaler
                    if self.device_type == "gpu" and scaler is not None: # for single gpu or multi gpus devices within ddp container
                        # take optimizer step and update the scaling factor if training in fp16
                        scaler.step(optimizer)
                        scaler.update()

                    # for single tpu host with pjrt
                    elif self.device_type == "tpu" and self.pjrt_dist:
                        xm.optimizer_step()

                    # for single tpu core
                    elif self.device_type == "tpu" and (not self.tpu_ddp and not self.pjrt_dist):
                        xm.optimizer_step(optimizer, barrier=True)
                        

                    # for cpu device or single tpu with multi core or multi tpu devices with multicore wrapped within ddp container
                    elif (self.device_type == "tpu" and self.tpu_ddp) or self.device_type == "cpu":
                        optimizer.step()
                    
                    # flush the gradients as soon as we can, no need for this memory anymore
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1 # update (no. of model parameters adjustment) steps

                # update local and global iteration nums
                global_iter += 1
                local_iter += 1
            
                # timing and logging
                t1 = time.perf_counter()
                iter_time = t1 - iter_t0

                if global_iter % self.log_interval == 0:
                    # get loss as float. note: this is a CPU-GPU sync point
                    # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
                    lossf = loss.item() * self.gradient_accumulation_steps
                    if local_iter >= 6: # let training settle a bit!
                        self.logger.info(f"Current Iteration: {local_iter}, Overall Iteration: {global_iter}, Total Steps: {global_step}, Loss: {lossf}, Time: {(iter_time*1000):.2f}ms, Estimated Remaining Hours: {(((t1 - total_t0) / (global_iter - initial_iter)) * (max_iters - global_iter) / 3600):.2f} hours, Estimated Remaining Days: {(((t1 - total_t0) / (global_iter - initial_iter)) * (max_iters - global_iter) / 3600 / 24):.2f} days")
                
            
        return train_losses
