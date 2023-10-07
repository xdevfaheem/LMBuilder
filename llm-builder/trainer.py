
import torch_xla.core.xla_model as xm
import torch
import matplotlib.pyplot as plt
import wandb
import time
import math
import os
                                                                
class Trainer:
    
    def __init__(self, trainer_config, train_loader, val_loader):
        self.config = trainer_config
        self.tbatch_genarator = train_loader
        self.vbatch_generator = val_loader
        self.epochs = epochs
        self.plot_dir =  os.path.join(self.config.log_dir, "plots")
        os.makedirs(self.plot_dir, exist_ok=True)
            
        # logging
        if self.config.wandb_log and self.config.mastr_proc:
            wandb.init(project=self.config.wandb_project, name=self.config.wandb_run_name, config=self.config.logging_config)
        self.logger = self.config.logger
    
    # helps estimate an arbitrarily accurate loss over either split using many batches
    @torch.no_grad()
    def validate_model(self, model, global_step, epoch):
        
        # dict for saving the loss values
        out = {}
        model.eval()
        
        for split in ['train', 'val']:
            
            losses = torch.zeros(self.config.eval_iters)
            generator = iter(self.tbatch_genarator if split=="train" else self.vbatch_generator)
            
            for k, (X, Y) in enumerate(generator):
                if k > self.config.eval_iters:
                    break
                with self.config.ctx:
                    _, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses
        
        # save loss curves to visualize later
        self.save_eval_loss_curves(out, global_step, epoch)
        
        out["train"] = out["train"].mean()
        out["val"] = out["val"].mean()
        
        del model #free up memory
        return out
    
    # learning rate decay scheduler (cosine with warmup)
    def get_lr(self, it):
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
    
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
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
    
    def save_checkpoint(self, out_dir, model, optimizer, scaler, global_step, best_val_loss, epoch):
        checkpoint_dir = os.path.join(out_dir, f"E{epoch}_iter{global_step}_ckpt.pt")
        checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'grad_scaler': scaler.state_dict(),
                    'model_args': self.config.model_args,
                    'global_step': global_step,
                    'best_val_loss': best_val_loss,
                    'config': self.config,
        }
        self.logger.info(f"saving checkpoint to {checkpoint_dir}")
        torch.save(checkpoint, checkpoint_dir)
    
    def train(self, model):
        optimizer = self.configure_optimizers(self.config.weight_decay, self.config.learning_rate, self.config.betas, self.config.device_type)
        # local and global steps 
        global_step=config.global_step
        local_step=0
        
        t0 = time.perf_counter()
        if config.eval_only:
            train_losses = torch.zeros(self.config.eval_interval)
        else:
            train_losses = torch.zeros(self.config.max_iters)
        scaler = config.scaler # scaler only for GPU
        # fetch the batch
        train_loader = iter(self.tbatch_genarator)
        model.train()
        
        for epoch in epochs:
            self.logger.info(f"{'-'*25} Epoch:{epoch} {'-'*25}")
            
            for X, Y in train_loader:
            
                # termination conditions
                if self.config.max_iters is not None and global_step >= self.config.max_iters:
                    self.logger.info("Training Completed!")
                    break

                # determine and set the learning rate for this iteration
                lr = self.get_lr(global_step) if self.config.decay_lr else self.config.learning_rate
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

                # evaluate the loss on train/val sets and write checkpoints
                if global_step % self.config.eval_interval == 0 and self.config.mastr_proc:
                    losses_dict = self.validate_model(model, global_step, epoch)
                    self.logger.info(f"Epoch: {epoch} - Step: {global_step} - Train loss: {losses_dict['train']:.4f} - Val loss: {losses_dict['val']:.4f}")
                    if self.config.wandb_log:
                        wandb.log({
                            "step": global_step,
                            "train_loss": losses_dict['train'],
                            "val_loss": losses_dict['val'],
                            "lr": lr,
                        })
                    if losses_dict['val'] < self.config.best_val_loss or self.config.always_save_checkpoint:
                        best_val_loss = losses_dict['val']
                        if global_step > 0:
                            self.save_checkpoint(self.config.out_dir, model, optimizer, scaler, global_step, best_val_loss, epoch)
                    if self.config.eval_only and global_step == 0:
                        logger.info(f"Training Stopped at {global_step}")
                        break

                # forward backward update, with gradient accumulation to simulate larger batch size
                # check if the gradients are accumulating or accumulated
                is_accumulating = ((global_step+1) % self.config.gradient_accumulation_steps != 0 # +1 to prevent scale and backprop at gstep 0
                
                if is_accumulating:
                    
                    # This bloats the code with repeated code. should be fixed by looking into this (https://pytorch.org/docs/master/_modules/torch/nn/parallel/distributed.html#DistributedDataParallel.no_sync) in the future insha allah
                    if self.config.ddp:
                        #https://github.com/pytorch/pytorch/issues/43201#issue-680863643
                        with torch.nn.parallel.DistributedDataParallel.no_sync()
                            with self.config.ctx:
                                logits, loss = model(X, Y)
                                loss = loss / self.config.gradient_accumulation_steps # scale the loss to account for gradient accumulation
                            if scaler: # for GPU
                                # backward pass, with gradient scaling if training in fp16
                                scaler.scale(loss).backward()
                            else: # for TPU and CPU
                                loss.backward()
                    else:
                        with self.config.ctx:
                                logits, loss = model(X, Y)
                                loss = loss / self.config.gradient_accumulation_steps # scale the loss to account for gradient accumulation
                            if scaler: # for GPU
                                # backward pass, with gradient scaling if training in fp16
                                scaler.scale(loss).backward()
                            else: # for TPU and CPU
                                loss.backward()
                
                if not is_accumulating:                   
                    # clip the gradient
                    if self.config.grad_clip != 0.0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_clip)
                    # step the optimizer and scaler if training in fp16
                    if scaler:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    # flush the gradients as soon as we can, no need for this memory anymore
                    optimizer.zero_grad(set_to_none=True)

                # timing and logging
                t1 = time.perf_counter()
                iter_time = t1 - t0
                t0 = t1

                if global_step % self.config.log_interval == 0:
                    # get loss as float. note: this is a CPU-GPU sync point
                    # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
                    lossf = loss.item() * self.config.gradient_accumulation_steps
                    if local_step >= 4: # let training settle a bit!
                        self.logger.info(f"Step: {global_step} - Loss: {lossf} - Time: {iter_time*1000:.2f}ms")
                        train_losses[global_step] = lossf
                
                global_step += 1
                local_iter_num += 1
            
        return train_losses
