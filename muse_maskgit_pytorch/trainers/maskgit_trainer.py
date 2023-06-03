from typing import List

import torch  # noqa: F401
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers.optimization import SchedulerType
from ema_pytorch import EMA
from PIL import Image
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from muse_maskgit_pytorch.muse_maskgit_pytorch import MaskGit
from muse_maskgit_pytorch.t5 import t5_encode_text_from_encoded
from muse_maskgit_pytorch.trainers.base_accelerated_trainer import BaseAcceleratedTrainer

try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
except ImportError:
    torch_xla = None
    xm = None
    met = None


class MaskGitTrainer(BaseAcceleratedTrainer):
    def __init__(
        self,
        maskgit: MaskGit,
        dataloader: DataLoader,
        valid_dataloader: DataLoader,
        accelerator: Accelerator,
        optimizer: Optimizer,
        scheduler: SchedulerType,
        *,
        current_step: int,
        num_train_steps: int,
        batch_size: int,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = None,
        save_results_every: int = 100,
        save_model_every: int = 1000,
        log_metrics_every: int = 10,
        results_dir="./results",
        logging_dir="./results/logs",
        apply_grad_penalty_every=4,
        use_ema=True,
        ema_update_after_step=0,
        ema_update_every=1,
        validation_prompts=["a photo of a dog"],
        clear_previous_experiments=False,
        validation_image_scale: float = 1.0,
        only_save_last_checkpoint=False,
    ):
        super().__init__(
            dataloader=dataloader,
            valid_dataloader=valid_dataloader,
            accelerator=accelerator,
            current_step=current_step,
            num_train_steps=num_train_steps,
            gradient_accumulation_steps=gradient_accumulation_steps,
            max_grad_norm=max_grad_norm,
            save_results_every=save_results_every,
            save_model_every=save_model_every,
            results_dir=results_dir,
            logging_dir=logging_dir,
            apply_grad_penalty_every=apply_grad_penalty_every,
            clear_previous_experiments=clear_previous_experiments,
            validation_image_scale=validation_image_scale,
            only_save_last_checkpoint=only_save_last_checkpoint,
        )
        self.save_results_every = save_results_every
        self.log_metrics_every = log_metrics_every
        self.batch_size = batch_size
        # maskgit
        maskgit.vae.requires_grad_(False)
        maskgit.transformer.t5.requires_grad_(False)
        self.model: MaskGit = maskgit

        self.optim: Optimizer = optimizer
        self.lr_scheduler: SchedulerType = scheduler

        self.use_ema = use_ema
        self.validation_prompts: List[str] = validation_prompts
        if use_ema:
            ema_model = EMA(
                self.model,
                update_after_step=ema_update_after_step,
                update_every=ema_update_every,
            )
            self.ema_model = ema_model
        else:
            self.ema_model = None

    def save_validation_images(
        self, validation_prompts, step: int, cond_image=None, cond_scale=3, temperature=1
    ):
        images = self.model.generate(
            validation_prompts,
            cond_images=cond_image,
            cond_scale=cond_scale,
            temperature=temperature,
        ).to("cpu")

        save_dir = self.results_dir.joinpath("MaskGit")
        save_dir.mkdir(exist_ok=True, parents=True)
        save_file = save_dir.joinpath(f"maskgit_S{step:04d}.png")

        if self.accelerator.is_main_process:
            save_image(images, save_file, "png")
            self.log_validation_images([Image.open(save_file)], step, ["\n---\n".join(validation_prompts)])
        return save_file

    def train(self):
        self.steps = self.steps + 1
        self.model.train()

        if self.accelerator.is_main_process:
            proc_label = f"[P{self.accelerator.process_index:03d}][Master]"
        else:
            proc_label = f"[P{self.accelerator.process_index:03d}][Worker]"

        # logs
        while self.steps < self.num_train_steps:
            for imgs, input_ids, attn_mask in self.dl:
                train_loss = 0.0
                steps = int(self.steps.item())

                with torch.no_grad():
                    text_embeds = t5_encode_text_from_encoded(
                        input_ids, attn_mask, self.model.transformer.t5, self.accelerator.device
                    )

                with self.accelerator.accumulate(self.model), self.accelerator.autocast():
                    loss = self.model(imgs, text_embeds=text_embeds)
                    self.accelerator.backward(loss)
                    if self.max_grad_norm is not None and self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optim.step()
                    self.lr_scheduler.step()
                    self.optim.zero_grad()

                    if self.use_ema:
                        self.ema_model.update()

                    gathered_loss = self.accelerator.gather_for_metrics(loss)
                    train_loss = gathered_loss.mean() / self.gradient_accumulation_steps

                    logs = {"loss": train_loss, "lr": self.lr_scheduler.get_last_lr()[0]}
                    self.print(f"[S{steps:05d}]{proc_label}: maskgit loss: {logs['loss']} - lr: {logs['lr']}")
                    self.accelerator.log(logs, step=steps)

                if not (steps % self.save_model_every):
                    self.accelerator.print(f"[S{steps:05d}]{proc_label}: saving model to {self.results_dir}")

                    state_dict = self.accelerator.unwrap_model(self.model).state_dict()
                    maskgit_save_name = "maskgit_superres" if self.model.cond_image_size else "maskgit"
                    file_name = (
                        f"{maskgit_save_name}.{steps}.pt"
                        if not self.only_save_last_checkpoint
                        else f"{maskgit_save_name}.pt"
                    )

                    model_path = self.results_dir.joinpath(file_name)
                    self.accelerator.wait_for_everyone()
                    self.accelerator.save(state_dict, model_path)

                    if self.use_ema:
                        self.accelerator.print(
                            f"[S{steps:05d}]{proc_label}: saving EMA model to {self.results_dir}"
                        )
                        ema_state_dict = self.accelerator.unwrap_model(self.ema_model).state_dict()
                        file_name = (
                            f"{maskgit_save_name}.{steps}.ema.pt"
                            if not self.only_save_last_checkpoint
                            else f"{maskgit_save_name}.ema.pt"
                        )
                        model_path = str(self.results_dir / file_name)
                        self.accelerator.wait_for_everyone()
                        self.accelerator.save(ema_state_dict, model_path)

                if not (steps % self.save_results_every):
                    cond_image = None
                    if self.model.cond_image_size:
                        self.accelerator.print(
                            "With conditional image training, we recommend keeping the validation prompts to empty strings"
                        )
                        cond_image = F.interpolate(imgs[0], 256)

                    self.accelerator.print(f"[S{steps:05d}]{proc_label}: Logging validation images")
                    saved_image = self.save_validation_images(
                        self.validation_prompts, steps, cond_image=cond_image
                    )
                    self.accelerator.print(f"[S{steps:05d}]{proc_label}: saved to {saved_image}")

                if met is not None and not (steps % self.log_metrics_every):
                    self.accelerator.print(f"[S{steps:05d}]{proc_label}: metrics:")

                self.steps += 1

        # loop complete, save final model
        self.accelerator.print(f"[S{steps:05d}]{proc_label}[FINAL]: saving model to {self.results_dir}")
        state_dict = self.accelerator.unwrap_model(self.model).state_dict()
        maskgit_save_name = "maskgit_superres" if self.model.cond_image_size else "maskgit"
        file_name = (
            f"{maskgit_save_name}.{steps}.pt"
            if not self.only_save_last_checkpoint
            else f"{maskgit_save_name}.pt"
        )

        model_path = self.results_dir.joinpath(file_name)
        self.accelerator.wait_for_everyone()
        self.accelerator.save(state_dict, model_path)

        if self.use_ema:
            self.accelerator.print(
                f"[S{steps:05d}]{proc_label}[FINAL]: saving EMA model to {self.results_dir}"
            )
            ema_state_dict = self.accelerator.unwrap_model(self.ema_model).state_dict()
            file_name = (
                f"{maskgit_save_name}.{steps}.ema.pt"
                if not self.only_save_last_checkpoint
                else f"{maskgit_save_name}.ema.pt"
            )
            model_path = str(self.results_dir / file_name)
            self.accelerator.wait_for_everyone()
            self.accelerator.save(ema_state_dict, model_path)

        cond_image = None
        if self.model.cond_image_size:
            self.accelerator.print(
                "With conditional image training, we recommend keeping the validation prompts to empty strings"
            )
            cond_image = F.interpolate(imgs[0], 256)
        steps = int(self.steps.item()) + 1  # get the final step count, plus one
        self.accelerator.print(f"[S{steps:05d}]{proc_label}: Logging validation images")
        saved_image = self.save_validation_images(self.validation_prompts, steps, cond_image=cond_image)
        self.accelerator.print(f"[S{steps:05d}]{proc_label}: saved to {saved_image}")

        if met is not None and not (steps % self.log_metrics_every):
            self.accelerator.print(f"[S{steps:05d}]{proc_label}: metrics:")
