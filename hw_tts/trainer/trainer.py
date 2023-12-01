import random
from pathlib import Path
from random import shuffle
from typing import Optional
import PIL
import torch
from hw_tts.base import BaseTrainer
from hw_tts.logger.utils import plot_spectrogram_to_buf
from hw_tts.utils import ROOT_PATH, MetricTracker, inf_loop
from torch.nn.utils import clip_grad_norm_
from torchvision.transforms import ToTensor
from tqdm import tqdm
from hw_tts.preproc import MelSpectrogram
from hw_tts.utils.util import get_data


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(
            self,
            model,
            gen_criterion,
            desc_criterion,
            gen_optimizer,
            desc_optimizer,
            config,
            device,
            dataloaders,
            gen_lr_scheduler=None,
            desc_lr_scheduler=None,
            len_epoch=None,
            skip_oom=True,
    ):
        print(device)
        super().__init__(model=model, 
                         gen_criterion=gen_criterion,
                         gen_optimizer = gen_optimizer,
                         gen_lr_shceduler=gen_lr_scheduler,
                         desc_criterion=desc_criterion,
                         desc_optimizer=desc_optimizer,
                         desc_lr_shceduler=desc_lr_scheduler,
                         config=config,
                         device=device)
        
        self.skip_oom = skip_oom
        self.config = config
        self.train_dataloader = dataloaders["train"]
        self.device = device
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.train_dataloader)
        else:
            # iteration-based training
            self.train_dataloader = inf_loop(self.train_dataloader)
            self.len_epoch = len_epoch
        self.evaluation_dataloaders = {k: v for k, v in dataloaders.items() if k != "train"}
        self.gen_lr_scheduler = gen_lr_scheduler
        self.desc_lr_scheduler = desc_lr_scheduler
        self.log_step = self.config["trainer"].get("log_step", 50)
        self.batch_accum_steps = self.config["trainer"].get("batch_accum_steps", 1)
        self.loss_keys = ["GenLoss", "DescLoss", "AdversarialLoss", "FeatureMatchingLoss", "MelLoss"]
        

        self.train_metrics = MetricTracker(
            "GenLoss", "DescLoss", "AdversarialLoss", "FeatureMatchingLoss",
            "MelLoss", "Gen_grad_norm", "Desc_grad_norm", writer=self.writer
        )
        self.mels___ = MelSpectrogram().to(device)

    @staticmethod
    def move_batch_to_device(batch, device: torch.device):
        """
        Move all necessary tensors to the GPU
        """
        names = ["spectrogram", "real_audio"]
        for tensor_for_gpu in names:
            batch[tensor_for_gpu] = batch[tensor_for_gpu].to(device)
        return batch

    def _clip_grad_norm(self):
        if self.config["trainer"].get("grad_norm_clip", None) is not None:
            clip_grad_norm_(
                self.model.parameters(), self.config["trainer"]["grad_norm_clip"]
            )

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        self.writer.add_scalar("epoch", epoch)

        progress_bar = tqdm(range(self.len_epoch), desc='train')

        for batch_idx, batch in enumerate(self.train_dataloader):
            stop = False
            progress_bar.update(1)
            try:
                batch = self.process_batch(
                        batch,
                        is_train=True,
                        metrics=self.train_metrics
                )
            except RuntimeError as e:
                if "out of memory" in str(e) and self.skip_oom:
                    self.logger.warning("OOM on batch. Skipping batch.")
                    for p in self.model.parameters():
                        if p.grad is not None:
                            del p.grad  # free some memory
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
            if batch_idx % self.log_step == 0:
                self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
                self.logger.debug(
                    f"Train Epoch: {epoch} {self._progress(batch_idx)} \
                        GenLoss: {batch['GenLoss'].item()}, DescLoss: {batch['DescLoss'].item()}"
                )
                self.writer.add_scalar(
                    "learning rate Gen", self.gen_lr_scheduler.get_last_lr()[0]
                )
                self.writer.add_scalar(
                    "learning rate Desc", self.desc_lr_scheduler.get_last_lr()[0]
                )
                
                self._log_scalars(self.train_metrics)
                last_train_metrics = self.train_metrics.result()
                self.train_metrics.reset()
            if batch_idx + 1 >= self.len_epoch:
                stop = True
                break
            if stop:
                break
        log = last_train_metrics

        if self.gen_lr_scheduler is not None:
            self.gen_lr_scheduler.step()
        if self.desc_scheduler is not None:
            self.desc_scheduler.step()

        self._log_audio(batch['pred_audio'][0], 22050, 'AudioSyntTrain_0.wav')
        if batch['pred_audio'].shape[0] > 1:
            self._log_audio(batch['pred_audio'][1], 22050, 'AudioSyntTrain_1.wav')

        self.synt(self.model, 22050)
        return log
    
    def process_batch(self, batch, is_train: bool, metrics: MetricTracker):
        batch = self.move_batch_to_device(batch, self.device)
        generated = self.model(**batch)

        batch.update(generated)
        desc = self.model._descriminator(generated=batch["pred_audio"].detach(), real=batch["real_audio"])
        batch.update(desc)

        if is_train:
            self.desc_optimizer.zero_grad()
            desc_loss = self.desc_loss(**batch)
            desc_loss["DescLoss"].backward()
            
            self.train_metrics.update("Desc_grad_norm", self.get_grad_norm("Desc"))
            self.desc_optimizer.step()

            self.gen_optimizer.zero_grad()
            d_outputs = self.model._descriminator(generated=batch["pred_audio"], real=batch["real_audio"])
            batch.update(d_outputs)

            gen_loss = self.gen_loss(**batch)
            gen_loss["GenLoss"].backward()

            self.train_metrics.update("Gen_grad_norm", self.get_grad_norm('Gen'))
            self.gen_optimizer.step()

            batch.update(gen_loss)
            batch.update(desc_loss)

            for key in self.loss_keys:
                metrics.update(key, batch[key].item())
        return batch
    
    def synt(self):
        self.model.eval()
        data = get_data(22050)
        for i, mel in enumerate(data):
            generated = self.model(mel.to(self.device))["pred_audio"].squeeze(0)
            generated = generated.detach().cpu().numpy()
            self._log_audio(generated, 22050, f"test_{i}")
           
    def _log_predictions(
            self,
            pred_audio,
            *args,
            **kwargs):
        if self.writer is None:
            return

        for i in range(pred_audio.shape[0]):
            self._log_audio(pred_audio[i, ...], 22050, f"Train Synt_{i}")

    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.train_dataloader, "n_samples"):
            current = batch_idx * self.train_dataloader.batch_size
            total = self.train_dataloader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    def _log_spectrogram(self, spectrogram_batch):
        spectrogram = random.choice(spectrogram_batch.cpu())
        image = PIL.Image.open(plot_spectrogram_to_buf(spectrogram))
        self.writer.add_image("spectrogram", ToTensor()(image))

    def _log_audio(self, audio, sr, name):
        self.writer.add_audio(f"Audio_{name}", audio, sample_rate=sr)

    @torch.no_grad()
    def get_grad_norm(self, part='Gen', norm_type=2):
        if part=='Gen':
            model = self.model.generator
        else:
            model = self.model.descriminator
        parameters = model.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).cpu() for p in parameters]
            ),
            norm_type,
        )
        return total_norm.item()

    def _log_scalars(self, metric_tracker: MetricTracker):
        if self.writer is None:
            return
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(f"{metric_name}", metric_tracker.avg(metric_name))