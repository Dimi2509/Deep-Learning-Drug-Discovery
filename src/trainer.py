from functools import partial
from copy import deepcopy

import numpy as np
import torch
from tqdm import tqdm

from meanTeacher_utils import GraphAugmentor, augment_batch


class SemiSupervisedEnsemble:
    def __init__(
        self,
        supervised_criterion,
        optimizer,
        scheduler,
        device,
        models,
        logger,
        datamodule,
    ):
        self.device = device
        self.models = models

        # Optim related things
        self.supervised_criterion = supervised_criterion
        all_params = [p for m in self.models for p in m.parameters()]
        self.optimizer = optimizer(params=all_params)
        self.scheduler = scheduler(optimizer=self.optimizer)

        # Dataloader setup
        self.train_dataloader = datamodule.train_dataloader()
        self.val_dataloader = datamodule.val_dataloader()
        self.test_dataloader = datamodule.test_dataloader()

        # Logging
        self.logger = logger

    def validate(self):
        for model in self.models:
            model.eval()

        val_losses = []
        
        with torch.no_grad():
            for x, targets in self.val_dataloader:
                x, targets = x.to(self.device), targets.to(self.device)
                
                # Ensemble prediction
                preds = [model(x) for model in self.models]
                avg_preds = torch.stack(preds).mean(0)
                
                val_loss = torch.nn.functional.mse_loss(avg_preds, targets)
                val_losses.append(val_loss.item())
        val_loss = np.mean(val_losses)
        return {"val_MSE": val_loss}

    def train(self, total_epochs, validation_interval):
        #self.logger.log_dict()
        for epoch in (pbar := tqdm(range(1, total_epochs + 1))):
            for model in self.models:
                model.train()
            supervised_losses_logged = []
            for x, targets in self.train_dataloader:
                x, targets = x.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                # Supervised loss
                supervised_losses = [self.supervised_criterion(model(x), targets) for model in self.models]
                supervised_loss = sum(supervised_losses)
                supervised_losses_logged.append(supervised_loss.detach().item() / len(self.models))  # type: ignore
                loss = supervised_loss
                loss.backward()  # type: ignore
                self.optimizer.step()
            self.scheduler.step()
            supervised_losses_logged = np.mean(supervised_losses_logged)

            summary_dict = {
                "supervised_loss": supervised_losses_logged,
            }
            if epoch % validation_interval == 0 or epoch == total_epochs:
                val_metrics = self.validate()
                summary_dict.update(val_metrics) # Appends to the summary dict
                pbar.set_postfix(summary_dict)
            self.logger.log_dict(summary_dict, step=epoch)

        return np.array(0)


class MeanTeacher:
    def __init__(
        self,
        supervised_criterion,
        optimizer,
        scheduler,
        device,
        models,  # List with single model (student)
        logger,
        datamodule,
        # Mean Teacher specific parameters
        ema_decay=0.999,
        consistency_weight=100.0,
        consistency_type="mse",
        consistency_rampup=5,
        augmentation=None,
        late_start_epochs=20,
        slowdown=1000,
    ):
        """
        Args:
            supervised_criterion: Loss function for labeled data (e.g., MSELoss)
            optimizer: Optimizer factory (partial function)
            scheduler: LR scheduler factory (partial function)
            device: Device to run on (cuda/cpu)
            models: List containing the student model
            logger: WandB logger instance
            datamodule: QM9DataModule instance
            ema_decay: EMA coefficient for teacher update (α in paper)
            consistency_weight: Maximum weight for consistency loss
            consistency_type: "mse" for consistency loss
            consistency_rampup: Epochs to ramp up consistency weight
            augmentation: Dict with augmentation parameters
        """
        self.device = device
        self.student_model = models[0]
        
        # Create teacher model as a copy of student
        # Key decision: Teacher parameters are detached (no gradients)
        # self.teacher_model = self._create_teacher_model(self.student_model)

        # Late start of teacher model
        self.teacher_model = None
        self.teacher_initialized = False
        self.late_start_epochs = late_start_epochs
        self.late_start_step = 0  # Global step when teacher was initialized
        
        # Loss functions
        self.supervised_criterion = supervised_criterion
        self.consistency_type = consistency_type
        
        # Optimizer and scheduler (only for student)
        self.optimizer = optimizer(params=self.student_model.parameters())
        self.scheduler = scheduler(optimizer=self.optimizer)
        
        # Mean Teacher hyperparameters
        self.ema_decay = ema_decay
        self.consistency_weight = consistency_weight
        self.consistency_rampup = consistency_rampup
        self.alpha = 0.0  # Initial alpha for EMA
        self.slowdown = slowdown
        
        # Graph augmentation
        if augmentation is None:
            augmentation = {}
        self.augmentor = GraphAugmentor(
            use_feature_noise=augmentation.get('use_feature_noise', True), # Default to True
            feature_noise_std=augmentation.get('feature_noise_std', 0.1),
            use_edge_dropout=augmentation.get('use_edge_dropout', False),
            edge_dropout_rate=augmentation.get('edge_dropout_rate', 0.0),
        )
        
        # Data loaders
        self.train_labeled_loader = datamodule.train_dataloader()
        self.train_unlabeled_loader = datamodule.unsupervised_train_dataloader()
        self.val_dataloader = datamodule.val_dataloader()
        self.test_dataloader = datamodule.test_dataloader()
        
        # Logging
        self.logger = logger
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
    
    def _create_teacher_model(self, student_model):
        """
        Create teacher as a copy of student with detached parameters.
        
        Key decision: Teacher parameters don't require gradients because
        they're updated via EMA, not backpropagation.
        """
        teacher = deepcopy(student_model)
        for param in teacher.parameters():
            param.detach_()  # Detach from computation graph
            param.requires_grad = False  # No gradients needed
        return teacher
    
    def _update_teacher(self):
        """
        Update teacher weights using exponential moving average of student weights.
        
        Formula: θ_teacher = α * θ_teacher + (1 - α) * θ_student
        """

        # Slowdown factor for alpha ramp-up
        # Make sure to start counting after late start
        # self.alpha = min(steps_since_init / self.slowdown, self.ema_decay)
        #self.alpha = min(1 - 1 / ((self.global_step) / self.slowdown + 1), self.ema_decay)

        # Adaptive alpha: 0.99 during rampup, then 0.999
        if (self.current_epoch - self.late_start_epochs) < self.consistency_rampup:
            self.alpha = 0.99  # Faster adaptation during rampup
        else:
            self.alpha = self.ema_decay  # Slower adaptation after rampup (0.999)

        for teacher_param, student_param in zip(self.teacher_model.parameters(), self.student_model.parameters()):
            teacher_param.data.mul_(self.alpha).add_(student_param.data, alpha=1 - self.alpha)
    
    def _get_consistency_weight(self, epoch):
        """
        Get current consistency weight with sigmoid ramp-up.
        
        Key decision: Ramp up consistency loss over first few epochs because
        early teacher predictions are poor. Using sigmoid (not linear) gives
        smoother transition as recommended in Temporal Ensembling paper.
        """
        # Teacher late start
        if not self.teacher_initialized:
            return 0.0  # No consistency before teacher exists
        
        if self.consistency_rampup == 0:
            return self.consistency_weight
        
        current = np.clip(epoch - self.late_start_epochs, 0.0, self.consistency_rampup)
        rampup_value = current / self.consistency_rampup
        
        # Sigmoid rampup (smoother than Gaussian)
        # Maps [0,1] -> [0,1] with smooth S-curve
        # More gradual in the middle where you're seeing spikes
        #rampup_factor = rampup_value ** 2 * (3.0 - 2.0 * rampup_value)  # Smoothstep BEST ONE SO FAR
        
        # Alternative: Gaussian rampup (original paper, more aggressive)
        #rampup_factor = np.exp(-5.0 * (1.0 - rampup_value) ** 2)
        
        # Alternative: Linear rampup (most gradual)
        rampup_factor = rampup_value

        # x3 rampup for more gradual start
        #rampup_factor = rampup_value ** 3
        
        return self.consistency_weight * float(rampup_factor)
    
    
    def _consistency_loss(self, student_output, teacher_output):
        """
        Compute consistency loss between student and teacher predictions.
        """
        if self.consistency_type == "mse":
            return torch.nn.functional.mse_loss(student_output, teacher_output)
        else:
            raise ValueError(f"Unknown consistency type: {self.consistency_type}")
    
    def _train_step_supervised_only(self, labeled_batch, epoch):
        """
        Training step before teacher is initialized.
        Only supervised loss, no consistency.
        """
        self.student_model.train()
        
        losses_dict = {}
        
        # Process labeled data
        labeled_data, labeled_targets = labeled_batch
        labeled_data = labeled_data.to(self.device)
        labeled_targets = labeled_targets.to(self.device)
        
        # Forward pass (student only)
        student_out = self.student_model(labeled_data)
        
        # Supervised loss
        supervised_loss = self.supervised_criterion(student_out, labeled_targets)
        losses_dict['supervised_loss'] = supervised_loss.item()
        
        # No consistency loss yet
        losses_dict['consistency_loss'] = 0.0
        losses_dict['consistency_weight'] = 0.0
        losses_dict['total_loss'] = supervised_loss.item()
        
        # Optimization
        self.optimizer.zero_grad()
        supervised_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        self.global_step += 1
    
        return losses_dict

    def _train_step(self, labeled_batch, unlabeled_batch, epoch):
        """
        Single training step with both labeled and unlabeled data.
        
        Key steps:
        1. Augment inputs differently for student and teacher
        2. Forward pass through both models
        3. Compute supervised loss (labeled only)
        4. Compute consistency loss (all data)
        5. Backward pass and optimize student
        6. Update teacher with EMA
        """
        self.student_model.train()
        self.teacher_model.train()
        
        losses_dict = {}
        
        # === Process labeled data ===
        labeled_data, labeled_targets = labeled_batch
        labeled_data = labeled_data.to(self.device)
        labeled_targets = labeled_targets.to(self.device)
        
        # Create two augmented views
        student_labeled = augment_batch(labeled_data, self.augmentor, training=True)
        teacher_labeled = augment_batch(labeled_data, self.augmentor, training=True) # labeled_data # 
        
        # Forward pass
        with torch.no_grad():
            # Teacher predictions (no grad)
            teacher_out = self.teacher_model(teacher_labeled)
        
        # Student predictions (with grad)
        student_out = self.student_model(student_labeled)
        
        # Supervised loss (only on labeled data)
        supervised_loss = self.supervised_criterion(student_out, labeled_targets)
        
        losses_dict['supervised_loss'] = supervised_loss.item()
        
        # === Process unlabeled data ===
        unlabeled_data, _ = unlabeled_batch
        unlabeled_data = unlabeled_data.to(self.device)
        
        student_unlabeled = augment_batch(unlabeled_data, self.augmentor, training=True)
        teacher_unlabeled = augment_batch(unlabeled_data, self.augmentor, training=True) # unlabeled_data #
        
        with torch.no_grad():
            teacher_unlabeled_out = self.teacher_model(teacher_unlabeled)
        
        student_unlabeled_out = self.student_model(student_unlabeled)
        
        # Combine labeled and unlabeled for consistency loss
        student_cons_all = torch.cat([student_out, student_unlabeled_out], dim=0)
        teacher_cons_all = torch.cat([teacher_out, teacher_unlabeled_out], dim=0)

        # print(f"Batch size for consistency: {student_cons_all.shape[0]}")
        # print(f"Student predictions sample: {student_cons_all[:5].flatten()}")
        # print(f"Teacher predictions sample: {teacher_cons_all[:5].flatten()}")
        
        # Consistency loss with ramp-up weight
        consistency_weight = self._get_consistency_weight(epoch)
        consistency_loss = self._consistency_loss(student_cons_all, teacher_cons_all)
        weighted_consistency_loss = consistency_weight * consistency_loss

        # print(f"\n=== DEBUG: Predictions ===")
        # print(f"Student: min={student_cons_all.min():.3f}, max={student_cons_all.max():.3f}, mean={student_cons_all.mean():.3f}, std={student_cons_all.std():.3f}")
        # print(f"Teacher: min={teacher_cons_all.min():.3f}, max={teacher_cons_all.max():.3f}, mean={teacher_cons_all.mean():.3f}, std={teacher_cons_all.std():.3f}")
        # print(f"Labeled targets: min={labeled_targets.min():.3f}, max={labeled_targets.max():.3f}, mean={labeled_targets.mean():.3f}")
        # print(f"Prediction difference: {(student_cons_all - teacher_cons_all).abs().mean():.3f}")
        # print(f"=========================\n")
        
        losses_dict['consistency_loss'] = consistency_loss.item()
        losses_dict['consistency_weight'] = consistency_weight
        
        # Total loss
        total_loss = supervised_loss + weighted_consistency_loss        
        losses_dict['total_loss'] = total_loss.item()

        # print(f"Epoch {epoch}, Step {self.global_step}:")
        # print(f"  Supervised:  {supervised_loss.item():.4f} ({100*supervised_loss.item()/total_loss.item():.1f}%)")
        # print(f"  Consistency: {consistency_loss.item():.4f} ({100*weighted_consistency_loss.item()/total_loss.item():.1f}%)")
        # print(f"  Total:       {total_loss.item():.4f}")
        # print(f"  Consistency weight: {consistency_weight:.2f}")
        # print(f"  Raw consistency loss: {consistency_loss.item():.4f}")
        
        # Optimization step
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        # Update teacher model with EMA
        self._update_teacher()
        # print(f"Original data x: {labeled_data.x[0, :50]}")
        # print(f"Student data x:  {student_labeled.x[0, :5]}")
        # print(f"Teacher data x:  {teacher_labeled.x[0, :5]}")
        
        self.global_step += 1
        self.current_epoch = epoch
        
        return losses_dict
    
    def validate(self, use_teacher=True):
        """
        Validate on the validation set.
        
        Key decision: Evaluate teacher model (not student) as it provides
        more stable predictions due to weight averaging.
        """
        model = self.teacher_model if use_teacher else self.student_model
        model.eval()
        
        val_losses = []
        
        with torch.no_grad():
            for batch, targets in self.val_dataloader:
                batch = batch.to(self.device)
                targets = targets.to(self.device)
                
                # No augmentation during validation
                preds = model(batch)
                
                val_loss = torch.nn.functional.mse_loss(preds, targets)
                val_losses.append(val_loss.item())
        
        return {"val_MSE": np.mean(val_losses)}
    
    def train(self, total_epochs, validation_interval):
        """
        Main training loop with Mean Teacher.
        
        Key decisions:
        1. Iterate through both labeled and unlabeled data simultaneously
        2. Log all relevant metrics to WandB
        3. Validate periodically using teacher model
        """
        for epoch in (pbar := tqdm(range(1, total_epochs + 1))):
            epoch_losses = {
                'supervised_loss': [],
                'consistency_loss': [],
                'consistency_weight': [],
                'total_loss': [],
            }

             # === LATE START: Initialize teacher after N epochs ===
            #print(f"Epoch {epoch} late start epochs: {self.late_start_epochs}, initialized: {self.teacher_initialized}")
            if epoch == self.late_start_epochs and not self.teacher_initialized:
                print(f"\n{'='*60}")
                print(f"INITIALIZING TEACHER MODEL (Late Start at Epoch {epoch})")
                print(f"{'='*60}\n")
                
                # Create teacher from current trained student
                self.teacher_model = self._create_teacher_model(self.student_model)
                self.teacher_initialized = True
                
                # Reset global_step for consistency rampup
                # (so rampup starts from this epoch)
                self.late_start_step = self.global_step

            
                
            # Create iterators for both dataloaders
            labeled_iter = iter(self.train_labeled_loader)
            unlabeled_iter = iter(self.train_unlabeled_loader) if self.train_unlabeled_loader else None
            
            # Train for one epoch
            # Key decision: Iterate through labeled data, cycling through unlabeled
            for labeled_batch in self.train_labeled_loader:
                try:
                    unlabeled_batch = next(unlabeled_iter)
                except StopIteration:
                    unlabeled_iter = iter(self.train_unlabeled_loader)
                    unlabeled_batch = next(unlabeled_iter)
                
                # Training step
                # losses = self._train_step(labeled_batch, unlabeled_batch, epoch)

                # Teacher Late start implementation
                            # === Use teacher only if initialized ===
                if self.teacher_initialized:
                    # Full Mean Teacher training
                    losses = self._train_step(labeled_batch, unlabeled_batch, epoch)
                else:
                    # Supervised-only training (pre-initialization)
                    losses = self._train_step_supervised_only(labeled_batch, epoch)
                    
                # Accumulate losses
                for key in epoch_losses:
                    if key in losses:
                        epoch_losses[key].append(losses[key])
            
            # Update learning rate
            self.scheduler.step()
            
            # Compute average losses
            summary_dict = {
                key: np.mean(values) for key, values in epoch_losses.items()
            }

            # Add alpha to summary
            if self.teacher_initialized:
                summary_dict['ema_alpha'] = self.alpha
            #summary_dict['learning_rate'] = self.optimizer.param_groups[0]['lr']
            #summary_dict['ema_decay_used'] = min(1 - 1 / (self.global_step + 1), self.ema_decay)
            
            # Validation
            if epoch % validation_interval == 0 or epoch == total_epochs:
                
                if self.teacher_initialized:
                    val_metrics_teacher = self.validate(use_teacher=True)
                    summary_dict['val_MSE_teacher'] = val_metrics_teacher['val_MSE']

                val_metrics_student = self.validate(use_teacher=False)            
                summary_dict['val_MSE_student'] = val_metrics_student['val_MSE']
                
                pbar.set_postfix({
                    'sup_loss': f"{summary_dict['supervised_loss']:.4f}",
                    'cons_loss': f"{summary_dict['consistency_loss']:.4f}",
                    #'val_MSE': f"{val_metrics_teacher['val_MSE']:.4f}"
                })
            
            # Log to WandB
            self.logger.log_dict(summary_dict, step=epoch)
        
        # Final test set evaluation (using teacher)
        print("\nEvaluating on test set...")
        test_metrics = self.validate_test()
        self.logger.log_dict(test_metrics, step=total_epochs)
        print(f"Test MSE (teacher): {test_metrics['test_MSE']:.6f}")
        
        return [test_metrics['test_MSE']]
    
    def validate_test(self):
        """Evaluate on test set using teacher model."""
        self.teacher_model.eval()
        
        test_losses = []
        
        with torch.no_grad():
            for batch, targets in self.test_dataloader:
                batch = batch.to(self.device)
                targets = targets.to(self.device)
                
                preds = self.teacher_model(batch)
                
                test_loss = torch.nn.functional.mse_loss(preds, targets)
                test_losses.append(test_loss.item())
        
        return {"test_MSE": np.mean(test_losses)}