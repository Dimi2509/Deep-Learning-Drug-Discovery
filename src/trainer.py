from functools import partial
from itertools import cycle

import numpy as np
import torch
from tqdm import tqdm

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
        lambda_cps: float = 1.0
    ):
        self.device = device
        self.models = models
        self.lambda_cps = lambda_cps

        # Optim related things
        self.supervised_criterion = supervised_criterion
        all_params = [p for m in self.models for p in m.parameters()]
        self.optimizer = optimizer(params=all_params)
        self.scheduler = scheduler(optimizer=self.optimizer)

        # Dataloader setup
        self.train_dataloader = datamodule.train_dataloader()
        self.val_dataloader = datamodule.val_dataloader()
        self.test_dataloader = datamodule.test_dataloader()

        self.unsupervised_train_dataloader = datamodule.unsupervised_train_dataloader()


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
            cps_losses_logged = [] 
            for (x_l, y_l), (x_u, _) in zip(cycle(self.train_dataloader), self.unsupervised_train_dataloader):
                x_l, y_l = x_l.to(self.device), y_l.to(self.device)
                x_u = x_u.to(self.device)
                self.optimizer.zero_grad()

                # Forward pass on labeled + unlabeled
                preds_l = []  # labeled predictions
                preds_u = []  # unlabeled predictions
                for model in self.models:
                    preds_l.append(model(x_l))
                    preds_u.append(model(x_u))

                # Supervised loss (only labeled data)
                supervised_losses = [
                    self.supervised_criterion(pred_l, y_l) for pred_l in preds_l
                ]
                supervised_loss = sum(supervised_losses)

                # CPS: consistency loss between the two models 
                lambda_cps = self.lambda_cps

                if len(self.models) == 2:
                    # CPS: consistency loss ONLY on unlabeled data
                    preds1_u = preds_u[0]
                    preds2_u = preds_u[1]

                    # model 1 matches model 2, and vice versa (regression CPS)
                    loss_cps1 = torch.nn.functional.mse_loss(preds1_u, preds2_u.detach())
                    loss_cps2 = torch.nn.functional.mse_loss(preds2_u, preds1_u.detach())
                    cps_loss = loss_cps1 + loss_cps2

                    loss = supervised_loss + lambda_cps * cps_loss
                else:
                    # fallback: just supervised if not exactly 2 models
                    loss = supervised_loss

                # logging + backward as before
                supervised_losses_logged.append(supervised_loss.detach().item() / len(self.models))
                if len(self.models) == 2:
                    cps_losses_logged.append(cps_loss.detach().item())

                loss.backward()
                self.optimizer.step()

            self.scheduler.step()
            supervised_losses_logged = np.mean(supervised_losses_logged)
            
            if len(cps_losses_logged) > 0:
                cps_loss_epoch = np.mean(cps_losses_logged)
            else:
                cps_loss_epoch = 0.0

            summary_dict = {
                "supervised_loss": supervised_losses_logged,
                "cps_loss": cps_loss_epoch,
            }
            if epoch % validation_interval == 0 or epoch == total_epochs:
                val_metrics = self.validate()
                summary_dict.update(val_metrics)
                pbar.set_postfix(summary_dict)
            self.logger.log_dict(summary_dict, step=epoch)
            
        return np.array(0)
    
    