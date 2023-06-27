import argparse
import copy
import os

import wandb
from accelerate import Accelerator
from torch.optim import AdamW
from torch.utils.data import DataLoader

from core.image_datasets import PairDataset
from core.nn_util import EMA
from core.resample import create_named_schedule_sampler
from core.script_util import (add_dict_to_argparser, args_to_dict,
                              LLDE_create_model_and_diffusion,
                              LLDE_model_and_diffusion_defaults)


def main(args):
    """Setup"""
    save_dir = os.path.join(args.checkpoints_dir, args.experiment_name)
    accelerator = Accelerator(
        log_with="wandb",
        logging_dir=save_dir,
    )
    accelerator.init_trackers(
        project_name=args.project_name, 
        config=vars(args), 
        init_kwargs={
            "wandb": {
                "name": args.experiment_name, 
                "settings": wandb.Settings(start_method="fork")
            }
        }
    )
    device = accelerator.device

    train_dataset = PairDataset(args)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_threads, 
        pin_memory=True,
    )

    model, diffusion = LLDE_create_model_and_diffusion(
        **args_to_dict(args, LLDE_model_and_diffusion_defaults().keys())
    )
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ema = EMA(args.ema_rate)
    model_ema = copy.deepcopy(model)
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion) 

    model = accelerator.prepare(model)
    model_ema = accelerator.prepare(model_ema)
    train_loader = accelerator.prepare(train_loader)
    optimizer = accelerator.prepare(optimizer)

    """Train"""
    step = 0
    while(step < args.max_iterations):
        batch, cond = next(iter(train_loader))

        optimizer.zero_grad()
        t, weights = schedule_sampler.sample(batch.shape[0], device)
        losses = diffusion.training_losses(model, batch, t, model_kwargs=cond)
        loss = (losses["loss"] * weights).mean()
        accelerator.backward(loss)
        optimizer.step()
        ema.update_model_average(model_ema, model)

        if (step + 1) % args.log_interval == 0:
            accelerator.log({"loss": loss.detach().item()}, step=step)
                  
        # save model
        if (step + 1) % args.save_interval == 0:
            model_save_path = os.path.join(save_dir, f'{args.experiment_name}_iter_{step}.pt')
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            accelerator.save(unwrapped_model.state_dict(), model_save_path)

        step += 1

    accelerator.end_training()


def create_argparser():
    defaults = LLDE_model_and_diffusion_defaults()
    train_defaults = dict(
        project_name="LLDE",
        experiment_name="LLDE_01",
        dataset_dir="../Datasets/LOL",
        checkpoints_dir="checkpoints",
        schedule_sampler="uniform",
        num_threads=4,
        batch_size=32,
        image_size=128,
        lr=1e-4,
        weight_decay=0.0,
        dropout=0.3,
        ema_rate=0.9999,
        log_interval=50,
        save_interval=5000,
        max_iterations=150000,
    )
    defaults.update(train_defaults)
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser  


if __name__ == "__main__":
    args = create_argparser().parse_args()
    main(args)