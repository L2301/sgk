"""
Trainer with Training Curve Logging

Records metrics at every eval step for curve comparison:
- Train loss
- Val loss  
- Val perplexity
"""

import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass, field

sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class TrainingCurve:
    """Container for training curve data."""
    steps: List[int] = field(default_factory=list)
    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    val_perplexities: List[float] = field(default_factory=list)
    
    def add_point(self, step: int, train_loss: float, val_loss: float, val_ppl: float):
        self.steps.append(step)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.val_perplexities.append(val_ppl)
    
    def to_dict(self) -> dict:
        return {
            "steps": self.steps,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "val_perplexities": self.val_perplexities,
        }


class Trainer:
    """
    Trainer with identical settings for fair comparison.
    
    Records full training curves for analysis.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config,
        name: str,
        device: torch.device,
    ):
        self.model = model.to(device)
        self.config = config
        self.name = name
        self.device = device
        
        # Only optimize trainable parameters
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        
        self.optimizer = optim.AdamW(
            trainable_params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=config.betas,
        )
        
        # Linear warmup + cosine decay
        def lr_lambda(step):
            if step < config.warmup_steps:
                return step / max(1, config.warmup_steps)
            progress = (step - config.warmup_steps) / max(1, config.max_steps - config.warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * progress))
        
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        
        # Training state
        self.global_step = 0
        self.curve = TrainingCurve()
    
    def train_step(self, batch: torch.Tensor) -> float:
        """Single training step."""
        self.model.train()
        batch = batch.to(self.device)
        
        self.optimizer.zero_grad()
        
        loss, _ = self.model.compute_loss(batch)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.max_grad_norm
        )
        
        self.optimizer.step()
        self.scheduler.step()
        
        self.global_step += 1
        
        return loss.item()
    
    @torch.no_grad()
    def evaluate(self, batches: List[torch.Tensor]) -> Dict[str, float]:
        """Evaluate on a set of batches."""
        self.model.eval()
        
        total_loss = 0.0
        total_tokens = 0
        
        for batch in batches:
            batch = batch.to(self.device)
            loss, _ = self.model.compute_loss(batch)
            # Count tokens (batch_size * (seq_len - 1) for next token prediction)
            n_tokens = batch.numel() - batch.shape[0]  # Exclude first token per sequence
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens
        
        avg_loss = total_loss / max(1, total_tokens)
        perplexity = math.exp(min(avg_loss, 100))  # Cap to avoid overflow
        
        return {
            "loss": avg_loss,
            "perplexity": perplexity,
        }
    
    def train(
        self,
        train_batches: List[torch.Tensor],
        val_batches: List[torch.Tensor],
        verbose: bool = True,
    ) -> TrainingCurve:
        """
        Full training loop with curve logging.
        
        Returns training curve for analysis.
        """
        batch_idx = 0
        running_loss = 0.0
        loss_count = 0
        
        for step in range(self.config.max_steps):
            # Get batch (cycle through)
            batch = train_batches[batch_idx % len(train_batches)]
            batch_idx += 1
            
            # Train step
            loss = self.train_step(batch)
            running_loss += loss
            loss_count += 1
            
            # Logging
            if (step + 1) % self.config.log_interval == 0 and verbose:
                avg_train_loss = running_loss / loss_count
                lr = self.scheduler.get_last_lr()[0]
                print(f"  [{self.name}] Step {step + 1}: loss={avg_train_loss:.4f}, lr={lr:.2e}")
            
            # Evaluation + curve point
            if (step + 1) % self.config.eval_interval == 0:
                avg_train_loss = running_loss / loss_count
                val_metrics = self.evaluate(val_batches)
                
                self.curve.add_point(
                    step=step + 1,
                    train_loss=avg_train_loss,
                    val_loss=val_metrics["loss"],
                    val_ppl=val_metrics["perplexity"],
                )
                
                running_loss = 0.0
                loss_count = 0
                
                if verbose:
                    print(f"  [{self.name}] Val: loss={val_metrics['loss']:.4f}, ppl={val_metrics['perplexity']:.2f}")
        
        return self.curve
    
    def get_final_metrics(
        self,
        train_batches: List[torch.Tensor],
        val_batches: List[torch.Tensor],
        test_batches: List[torch.Tensor],
    ) -> Dict[str, Dict[str, float]]:
        """Get final metrics on all splits."""
        return {
            "train": self.evaluate(train_batches),
            "val": self.evaluate(val_batches),
            "test": self.evaluate(test_batches),
        }


if __name__ == "__main__":
    print("Trainer module - use from run_experiment.py")

