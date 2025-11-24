from functools import partial

import numpy as np
import torch
from tqdm import tqdm
from graph_augmentation import GraphAugmentor

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
                summary_dict.update(val_metrics)
                pbar.set_postfix(summary_dict)
            self.logger.log_dict(summary_dict, step=epoch)





class FixMatchEnsemble:
    """
    FixMatch algorithm adapted for regression with ensemble models.
    
    Key differences from classification FixMatch:
    - Uses ensemble variance as confidence measure instead of softmax probability
    - Uses MSE for both supervised and unsupervised losses
    - Threshold is based on prediction uncertainty (ensemble std)
    """
    
    def __init__(
        self,
        supervised_criterion,
        optimizer,
        scheduler,
        device,
        models,
        logger,
        datamodule,
        # FixMatch specific parameters
        unsupervised_weight=1.0,
        confidence_threshold=0.5,  # Max std allowed for pseudo-labels
        use_ensemble_pseudolabels=True,
        augmentation_config=None,
    ):
        self.device = device
        self.models = models
        
        # FixMatch hyperparameters
        self.unsupervised_weight = unsupervised_weight
        self.confidence_threshold = confidence_threshold
        self.use_ensemble_pseudolabels = use_ensemble_pseudolabels
        
        # Augmentation setup
        if augmentation_config is None:
            augmentation_config = {}
        self.augmentor = GraphAugmentor(
            weak_config=augmentation_config.get('weak', None),
            strong_config=augmentation_config.get('strong', None)
        )
        
        # Optim related things
        self.supervised_criterion = supervised_criterion
        all_params = [p for m in self.models for p in m.parameters()]
        self.optimizer = optimizer(params=all_params)
        self.scheduler = scheduler(optimizer=self.optimizer)
        
        # Dataloader setup
        self.train_dataloader = datamodule.train_dataloader()
        self.unsupervised_dataloader = datamodule.unsupervised_train_dataloader()
        self.val_dataloader = datamodule.val_dataloader()
        self.test_dataloader = datamodule.test_dataloader()
        
        # Logging
        self.logger = logger
        
        # Statistics tracking
        self.pseudolabel_stats = {
            'total': 0,
            'used': 0,
            'avg_confidence': 0.0
        }
    
    def generate_pseudolabels(self, x_unlabeled):
        """
        Generate pseudo-labels using weak augmentation and ensemble predictions.
        
        Returns:
            pseudo_labels: Predicted values (ensemble mean)
            confidence_mask: Boolean mask indicating which predictions are confident
            ensemble_std: Standard deviation across ensemble (uncertainty measure)
        """
        # Apply weak augmentation
        x_weak = self.augmentor.weak(x_unlabeled)
        x_weak = x_weak.to(self.device)
        
        # Get predictions from all models
        with torch.no_grad():
            predictions = torch.stack([model(x_weak) for model in self.models])
            
            # Ensemble statistics
            pseudo_labels = predictions.mean(dim=0)  # [batch_size, 1]
            ensemble_std = predictions.std(dim=0)    # [batch_size, 1]
        
        # Confidence mask: low uncertainty = high confidence
        confidence_mask = ensemble_std < self.confidence_threshold
        
        return pseudo_labels, confidence_mask, ensemble_std
    
    def compute_unsupervised_loss(self, x_unlabeled):
        """
        Compute FixMatch unsupervised loss:
        1. Generate pseudo-labels with weak augmentation
        2. Apply strong augmentation
        3. Train to match pseudo-labels (for confident predictions only)
        """
        # Generate pseudo-labels with weak augmentation
        pseudo_labels, confidence_mask, ensemble_std = self.generate_pseudolabels(x_unlabeled)
        
        # Update statistics
        batch_size = confidence_mask.size(0)
        num_confident = confidence_mask.sum().item()
        self.pseudolabel_stats['total'] += batch_size
        self.pseudolabel_stats['used'] += num_confident
        self.pseudolabel_stats['avg_confidence'] += ensemble_std.mean().item()
        
        # If no confident predictions, return zero loss
        if num_confident == 0:
            return torch.tensor(0.0, device=self.device)
        
        # Apply strong augmentation
        x_strong = self.augmentor.strong(x_unlabeled)
        x_strong = x_strong.to(self.device)
        
        # Get predictions with strong augmentation from all models
        predictions_strong = [model(x_strong) for model in self.models]
        
        # Compute consistency loss only for confident pseudo-labels
        unsupervised_losses = []
        for pred in predictions_strong:
            # Apply confidence mask
            masked_pred = pred[confidence_mask]
            masked_target = pseudo_labels[confidence_mask]
            
            if masked_pred.numel() > 0:
                loss = torch.nn.functional.mse_loss(masked_pred, masked_target)
                unsupervised_losses.append(loss)
        
        if len(unsupervised_losses) == 0:
            return torch.tensor(0.0, device=self.device)
        
        unsupervised_loss = sum(unsupervised_losses) / len(unsupervised_losses)
        return unsupervised_loss
    
    def validate(self):
        """Validation loop - same as before."""
        for model in self.models:
            model.eval()
        
        val_losses = []
        ensemble_stds = []
        
        with torch.no_grad():
            for x, targets in self.val_dataloader:
                x, targets = x.to(self.device), targets.to(self.device)
                
                # Ensemble prediction
                preds = torch.stack([model(x) for model in self.models])
                avg_preds = preds.mean(0)
                std_preds = preds.std(0)
                
                val_loss = torch.nn.functional.mse_loss(avg_preds, targets)
                val_losses.append(val_loss.item())
                ensemble_stds.append(std_preds.mean().item())
        
        val_loss = np.mean(val_losses)
        val_std = np.mean(ensemble_stds)
        
        return {
            "val_MSE": val_loss,
            "val_ensemble_std": val_std
        }
    
    def train(self, total_epochs, validation_interval, warmup_epochs=10):
        """
        Training loop with FixMatch.
        
        Args:
            total_epochs: Total number of training epochs
            validation_interval: How often to validate
            warmup_epochs: Number of epochs before using unsupervised loss
        """
        for epoch in (pbar := tqdm(range(1, total_epochs + 1))):
            for model in self.models:
                model.train()
            
            # Reset statistics
            self.pseudolabel_stats = {
                'total': 0,
                'used': 0,
                'avg_confidence': 0.0
            }
            
            supervised_losses_logged = []
            unsupervised_losses_logged = []
            
            # Create iterators
            labeled_iter = iter(self.train_dataloader)
            unlabeled_iter = iter(self.unsupervised_dataloader)
            
            # Determine number of iterations (use the smaller of the two)
            num_iters = min(len(self.train_dataloader), len(self.unsupervised_dataloader))
            
            for i in range(num_iters):
                # Get labeled batch
                try:
                    x_labeled, targets = next(labeled_iter)
                except StopIteration:
                    labeled_iter = iter(self.train_dataloader)
                    x_labeled, targets = next(labeled_iter)
                
                x_labeled, targets = x_labeled.to(self.device), targets.to(self.device)
                
                # Get unlabeled batch
                try:
                    x_unlabeled, _ = next(unlabeled_iter)
                except StopIteration:
                    unlabeled_iter = iter(self.unsupervised_dataloader)
                    x_unlabeled, _ = next(unlabeled_iter)
                
                self.optimizer.zero_grad()
                
                # Supervised loss
                supervised_losses = [
                    self.supervised_criterion(model(x_labeled), targets) 
                    for model in self.models
                ]
                supervised_loss = sum(supervised_losses) / len(self.models)
                
                # Unsupervised loss (FixMatch)
                unsupervised_loss = torch.tensor(0.0, device=self.device)
                if epoch > warmup_epochs:
                    unsupervised_loss = self.compute_unsupervised_loss(x_unlabeled)
                
                # Total loss
                total_loss = supervised_loss + self.unsupervised_weight * unsupervised_loss
                
                # Backward pass
                total_loss.backward()
                self.optimizer.step()
                
                # Logging
                supervised_losses_logged.append(supervised_loss.detach().item())
                unsupervised_losses_logged.append(unsupervised_loss.detach().item())
            
            # Step scheduler
            self.scheduler.step()
            
            # Compute epoch statistics
            supervised_loss_avg = np.mean(supervised_losses_logged)
            unsupervised_loss_avg = np.mean(unsupervised_losses_logged)
            
            # Compute pseudo-label usage rate
            if self.pseudolabel_stats['total'] > 0:
                usage_rate = self.pseudolabel_stats['used'] / self.pseudolabel_stats['total']
                avg_conf = self.pseudolabel_stats['avg_confidence'] / num_iters
            else:
                usage_rate = 0.0
                avg_conf = 0.0
            
            summary_dict = {
                "supervised_loss": supervised_loss_avg,
                "unsupervised_loss": unsupervised_loss_avg,
                "total_loss": supervised_loss_avg + self.unsupervised_weight * unsupervised_loss_avg,
                "pseudolabel_usage_rate": usage_rate,
                "avg_ensemble_std": avg_conf,
            }
            
            # Validation
            if epoch % validation_interval == 0 or epoch == total_epochs:
                val_metrics = self.validate()
                summary_dict.update(val_metrics)
                pbar.set_postfix(summary_dict)
            
            self.logger.log_dict(summary_dict, step=epoch)
        
        return summary_dict
    

class VATTrainer:
    """Virtual Adversarial Training (VAT) trainer for regression using a GNN predictor.

    Notes:
    - Expects the datamodule to provide `train_dataloader()` for labeled data and
      `unsupervised_train_dataloader()` for unlabeled data.
    - Uses MSE as the divergence for VAT (suitable for regression outputs).
    - Accepts models as a list, iterates over all but designed for single model use.
    """
    def __init__(
        self,
        supervised_criterion,
        optimizer,
        scheduler,
        device,
        models,
        logger,
        datamodule,
        vat_xi: float = 1e-4,
        vat_eps: float = 0.5,
        vat_ip: int = 1,
        unsup_weight: float = 1.0,
        unsup_ratio: int = 5
    ):
        self.device = device
        self.models = models

        # Optim related things
        from hydra.utils import instantiate as _hydra_instantiate

        # supervised criterion: accept an instantiated loss, callable, or config
        if isinstance(supervised_criterion, torch.nn.modules.loss._Loss):
            self.supervised_criterion = supervised_criterion
        elif callable(supervised_criterion):
            self.supervised_criterion = supervised_criterion()
        else:
            try:
                self.supervised_criterion = _hydra_instantiate(supervised_criterion)
            except Exception:
                self.supervised_criterion = torch.nn.MSELoss()

        all_params = [p for m in self.models for p in m.parameters()]

        # optimizer may be a DictConfig, callable (partial), or already-instantiated
        if isinstance(optimizer, torch.optim.Optimizer):
            self.optimizer = optimizer
        elif isinstance(optimizer, partial):
            try:
                self.optimizer = optimizer(params=all_params)
            except TypeError:
                self.optimizer = optimizer(all_params)
        elif callable(optimizer):
            self.optimizer = optimizer(params=all_params)
        else:
            try:
                opt = _hydra_instantiate(optimizer, params=all_params)
                if isinstance(opt, partial):
                    try:
                        self.optimizer = opt(params=all_params)
                    except TypeError:
                        self.optimizer = opt(all_params)
                else:
                    self.optimizer = opt
            except Exception as e:
                raise RuntimeError(f"Could not instantiate optimizer from config: {e}")

        # scheduler may also be a config/callable/instantiated
        if isinstance(scheduler, torch.optim.lr_scheduler._LRScheduler):
            self.scheduler = scheduler
        elif isinstance(scheduler, partial):
            try:
                self.scheduler = scheduler(optimizer=self.optimizer)
            except TypeError:
                self.scheduler = scheduler(self.optimizer)
        elif callable(scheduler):
            self.scheduler = scheduler(optimizer=self.optimizer)
        else:
            try:
                sch = _hydra_instantiate(scheduler, optimizer=self.optimizer)
                if isinstance(sch, partial):
                    try:
                        self.scheduler = sch(optimizer=self.optimizer)
                    except TypeError:
                        self.scheduler = sch(self.optimizer)
                else:
                    self.scheduler = sch
            except Exception as e:
                raise RuntimeError(f"Could not instantiate scheduler from config: {e}")

        # VAT hyperparams
        self.vat_xi = vat_xi
        self.vat_eps = vat_eps
        self.vat_ip = vat_ip
        self.unsup_weight = unsup_weight
        self.unsup_ratio = unsup_ratio

        # Dataloader setup
        self.train_dataloader = datamodule.train_dataloader()
        self.unsup_dataloader = datamodule.unsupervised_train_dataloader()
        self.val_dataloader = datamodule.val_dataloader()
        self.test_dataloader = datamodule.test_dataloader()

        # Logging
        self.logger = logger

    def _normalize(self, d: torch.Tensor) -> torch.Tensor:
        d_flat = d.view(-1)
        norm = torch.norm(d_flat) + 1e-8
        return d / norm

    def _virtual_adversarial_loss(self, data, model):
        """Compute VAT loss for a single model."""
        model.eval()
        with torch.no_grad():
            pred = model(data).detach()

        d = torch.randn_like(data.x, device=data.x.device)

        for i in range(self.vat_ip):
            d.requires_grad_()
            data_pert = data.clone()
            data_pert.x = data.x + self.vat_xi * d
            pred_hat = model(data_pert)
            adv_distance = torch.nn.functional.mse_loss(pred_hat, pred)
            
            adv_distance.backward()
            d_grad = d.grad
            
            if d_grad is None:
                break
            d = self._normalize(d_grad.detach())
            model.zero_grad()

        r_adv = d * self.vat_eps
        data_r = copy.copy(data)
        data_r.x = data.x + r_adv
        pred_r = model(data_r)
        vat_loss = torch.nn.functional.mse_loss(pred_r, pred)
        model.train()
        return vat_loss

    def validate(self):
        for model in self.models:
            model.eval()

        val_losses = []
        with torch.no_grad():
            for x, targets in self.val_dataloader:
                x, targets = x.to(self.device), targets.to(self.device)
                
                preds = [model(x) for model in self.models]
                avg_preds = torch.stack(preds).mean(0)
                
                val_loss = torch.nn.functional.mse_loss(avg_preds, targets)
                val_losses.append(val_loss.item())
        return {"val_MSE": np.mean(val_losses)}

    # def train(self, total_epochs, validation_interval):
    #     unsup_iter = iter(self.unsup_dataloader)

    #     for epoch in (pbar := tqdm(range(1, total_epochs + 1))):
    #         for model in self.models:
    #             model.train()

    #         supervised_losses_logged = []
    #         vat_losses_logged = []

    #         for x, targets in self.train_dataloader:
    #             x, targets = x.to(self.device), targets.to(self.device)

    #             try:
    #                 xu = next(unsup_iter)
    #             except StopIteration:
    #                 unsup_iter = iter(self.unsup_dataloader)
    #                 xu = next(unsup_iter)

    #             xu, _ = xu
    #             xu = xu.to(self.device)

    #             self.optimizer.zero_grad()

    #             # VAT loss FIRST (has internal backward calls)
    #             vat_losses = [
    #                 self._virtual_adversarial_loss(xu, model) 
    #                 for model in self.models
    #             ]
    #             vat_loss = sum(vat_losses)

    #             # Supervised loss AFTER VAT
    #             supervised_losses = [
    #                 self.supervised_criterion(model(x), targets) 
    #                 for model in self.models
    #             ]
    #             supervised_loss = sum(supervised_losses)

    #             # Total loss
    #             loss = supervised_loss + self.unsup_weight * vat_loss

    #             supervised_losses_logged.append(supervised_loss.detach().item() / len(self.models))
    #             vat_losses_logged.append(vat_loss.detach().item() / len(self.models))

    #             loss.backward()
    #             self.optimizer.step()

    #         self.scheduler.step()

    #         summary_dict = {
    #             "supervised_loss": np.mean(supervised_losses_logged),
    #             "vat_loss": np.mean(vat_losses_logged),
    #         }
    #         if epoch % validation_interval == 0 or epoch == total_epochs:
    #             val_metrics = self.validate()
    #             summary_dict.update(val_metrics)
    #             pbar.set_postfix(summary_dict)
            # self.logger.log_dict(summary_dict, step=epoch)
    def train(self, total_epochs, validation_interval):
        unsup_iter = iter(self.unsup_dataloader)

        for epoch in (pbar := tqdm(range(1, total_epochs + 1))):
            for model in self.models:
                model.train()

            supervised_losses_logged = []
            vat_losses_logged = []

            for x, targets in self.train_dataloader:
                x, targets = x.to(self.device), targets.to(self.device)

                self.optimizer.zero_grad()

                # Multiple VAT batches
                vat_loss_total = 0
                for _ in range(self.unsup_ratio):
                    try:
                        xu = next(unsup_iter)
                    except StopIteration:
                        unsup_iter = iter(self.unsup_dataloader)
                        xu = next(unsup_iter)

                    xu, _ = xu
                    xu = xu.to(self.device)

                    vat_losses = [
                        self._virtual_adversarial_loss(xu, model) 
                        for model in self.models
                    ]
                    vat_loss_total += sum(vat_losses)

                vat_loss = vat_loss_total / self.unsup_ratio  # Average

                # Supervised loss
                supervised_losses = [
                    self.supervised_criterion(model(x), targets) 
                    for model in self.models
                ]
                supervised_loss = sum(supervised_losses)

                loss = supervised_loss + self.unsup_weight * vat_loss

                supervised_losses_logged.append(supervised_loss.detach().item() / len(self.models))
                vat_losses_logged.append(vat_loss.detach().item() / len(self.models))

                loss.backward()
                self.optimizer.step()
            self.scheduler.step()

            summary_dict = {
                "supervised_loss": np.mean(supervised_losses_logged),
                "vat_loss": np.mean(vat_losses_logged),
            }
            if epoch % validation_interval == 0 or epoch == total_epochs:
                val_metrics = self.validate()
                summary_dict.update(val_metrics)
                pbar.set_postfix(summary_dict)
            self.logger.log_dict(summary_dict, step=epoch)
