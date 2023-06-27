import argparse
import os

import torch
from accelerate import Accelerator
from PIL import Image
from torch.utils.data import DataLoader

from core.image_datasets import ImageFolder, postprocess_img
from core.script_util import (add_dict_to_argparser, args_to_dict,
                              LLDE_create_model_and_diffusion,
                              LLDE_model_and_diffusion_defaults)

def main(args):
    """Setup"""
    accelerator = Accelerator()
    device = accelerator.device

    test_dataset = ImageFolder(args.dataset_dir)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=args.num_threads, 
        pin_memory=True
    )

    model, diffusion = LLDE_create_model_and_diffusion(
        **args_to_dict(args, LLDE_model_and_diffusion_defaults().keys())
    )
    model_path = os.path.join(args.checkpoints_dir, f'{args.model_name}.pt')
    model.load_state_dict(torch.load(model_path, map_location=device))

    os.makedirs(args.saved_images_dir, exist_ok = True)
    model = accelerator.prepare(model)
    test_loader = accelerator.prepare(test_loader)

    """Test"""
    model.eval()
    for i, input in enumerate(test_loader):
        output = diffusion.p_sample_loop(
            model, 
            input.shape,
            model_kwargs={"low_light": input},
        )
        output = postprocess_img(output)
        output_name = f'img_{i+1000}.png'
        Image.fromarray(output[0]).save(os.path.join(args.saved_images_dir, output_name))

def create_argparser():
    defaults = LLDE_model_and_diffusion_defaults()
    test_defaults = dict(
        model_name="LLDE",
        checkpoints_dir="checkpoints",
        dataset_dir="../Datasets/LSRW/low",
        saved_images_dir="saved_images",
        timestep_respacing="25",
        num_threads=2,
    )
    defaults.update(test_defaults)
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser  


if __name__ == "__main__":
    args = create_argparser().parse_args()
    main(args)