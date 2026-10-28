import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP, DataParallel
from torch.utils.data import DataLoader, Dataset, distributed
from torchvision import transforms
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline
from typing import Dict
import torchvision.transforms.functional as TF
from pathlib import Path
import numpy as np
import random
from torchvision.transforms.functional import to_pil_image
import time
from random_prior import get_jigsaw, curriculum_strategy_jigsaw
    

class ImageTextPairDataset(Dataset):
    def __init__(self, image_dir, text_dir, transform=None, max_samples=None):
        self.image_dir = image_dir
        self.text_dir = text_dir
        self.transform = transform
        self.keys = sorted([
            os.path.splitext(f)[0]
            for f in os.listdir(image_dir)
            if f.endswith(".jpg")
        ])
        if max_samples:
            self.keys = self.keys[:max_samples]
    def __len__(self):
        return len(self.keys)
    def __getitem__(self, idx):
        key = self.keys[idx]
        img_path = os.path.join(self.image_dir, key + ".jpg")
        txt_path = os.path.join(self.text_dir, key + ".txt")
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        with open(txt_path, "r", encoding="utf-8") as f:
            prompt = f.read().strip().replace("<|endoftext|>", "").strip()
        if "|" in prompt:
            prompt = prompt.split("|")[0].strip()
        elif "." in prompt:
            prompt = prompt.split(".")[0].strip()
            
        return {"image": img, "prompt": prompt}

def register_cross_attention_hooks(unet, container: dict, device="cuda"):
    def hook_fn(module, input, output):
        name = module._layer_name if hasattr(module, "_layer_name") else str(id(module))
        key = f"{name}-{module.__class__.__name__}"
        if isinstance(output, tuple):
            output = output[0]
        container[key] = output.to(device)
    handles = []
    for name, module in unet.named_modules():
        if name.endswith("attn2"):
            module._layer_name = name
            handles.append(module.register_forward_hook(hook_fn))
            # print(f"Hooking {name} - {module.__class__.__name__}")
    return handles

def remove_hooks(hooks):
    for h in hooks:
        h.remove()

def load_pipeline(model_id="runwayml/stable-diffusion-v1-5", device="cuda"):
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_id, torch_dtype=torch.float16
    ).to(device)
    pipe.unet.eval(), pipe.vae.eval(), pipe.text_encoder.eval()
    return pipe

def check(x):
    return x
                        
def MMAP(pipe, target_img, target_prompt, dataloader, rank, device, batch_size=8, num_steps=20, alpha=1/255., epsilon=16/255., uap=None):
    img_size = 512
    delta = torch.zeros(1, 3, img_size, img_size, device=device, requires_grad=True, dtype=torch.float16)
    if uap is not None:
        delta = torch.load(uap).to(device)
        delta.requires_grad_()
        print("UAP is loaded")

    # set num_inference_steps for diffusion process
    pipe.scheduler.set_timesteps(num_inference_steps=50)
    
    # set timestep set for attack
    timestep_indices = [5, 10, 15, 20, 25]
    timestep_set = [pipe.scheduler.timesteps[i].item() for i in timestep_indices]
    
    target_attn_per_t = {}
    preprocess = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        lambda x: x.half()
    ])
    normalize = transforms.Compose([transforms.Normalize([0.5]*3, [0.5]*3)])
    #dataloader = None
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.time()
    if dataloader is not None:
        for step in range(num_steps):
            for batch_idx, batch in enumerate(dataloader):
                
                # for loss ablation and reproducibility
                seed = step * 10 + batch_idx  
                torch.manual_seed(seed)
                with torch.no_grad():
                    for t in timestep_set:
                        t_img = preprocess(target_img).unsqueeze(0).to(device)
                        t_img = normalize(t_img)
                        t_embed, t_embed_uncon = pipe.encode_prompt(target_prompt, device=device, num_images_per_prompt=1, do_classifier_free_guidance=True)
                        t_tensor = torch.tensor([t], device=device, dtype=torch.long)
                        attn_dict = {}
                        hooks = register_cross_attention_hooks(pipe.unet, attn_dict, device=device)
                        t_latents = pipe.prepare_latents(t_img, t_tensor, t_img.size(0), 1, t_img.dtype, device)
                        _ = pipe.unet(t_latents, timestep=t_tensor, encoder_hidden_states=t_embed)
                        remove_hooks(hooks)
                        target_attn_per_t[int(t)] = attn_dict
                
                
                
                x = batch["image"].to(device)
                prompts = batch["prompt"]
                
                mask = [target_prompt.lower() not in p.lower() for p in prompts]
                if not any(mask): continue
                x = x[mask]
                prompts = [p for i, p in enumerate(prompts) if mask[i]]
                #print(prompts)
                prompt_embed, uncon_prompt_embed = pipe.encode_prompt(prompts, device=device, num_images_per_prompt=1, 
                                                                    do_classifier_free_guidance=True)
                x_normed = normalize(x)
                grad_accum = torch.zeros_like(delta)
                total_L = 0
                
                for t in timestep_set:
                    t_tensor = torch.tensor([t], device=device, dtype=torch.long)
                    t_scalar = int(t_tensor.item())
                    batch_size = x_normed.size(0)
                    t_batch = torch.full((batch_size,), t_scalar, dtype=torch.long, device=device)

                    with torch.no_grad():
                        x_latent_orig = pipe.prepare_latents(x_normed, t_tensor, batch_size, 1, x_normed.dtype, device)
                    
                    delta = delta.detach().clone().requires_grad_()
                    x_adv = normalize(torch.clamp(x + delta, 0, 1))
                    x_latent_adv = pipe.prepare_latents(x_adv, t_tensor, batch_size, 1, x_adv.dtype, device)
                    
                    attn_adv_dict_tar, attn_adv_dict_non, attn_orig_dict = {}, {}, {}
                    
                    hooks = register_cross_attention_hooks(pipe.unet, attn_adv_dict_tar, device)
                    
                    # targeted loss
                    t_embed_new = t_embed.expand(x_latent_adv.shape[0], -1, -1).contiguous()
                    if x_latent_adv.shape[0] != t_embed_new.shape[0]:
                        print(x_latent_adv.shape[0], t_embed_new.shape[0])
                    _ = pipe.unet(x_latent_adv, timestep=t_batch, encoder_hidden_states=t_embed_new)
                    remove_hooks(hooks)
                    
                    ## S #####
                    hooks = register_cross_attention_hooks(pipe.unet, attn_adv_dict_non, device)
                    # suppression loss
                    _ = pipe.unet(x_latent_adv, timestep=t_batch, encoder_hidden_states=prompt_embed)
                    remove_hooks(hooks)
                    
                    ### O #####
                    with torch.no_grad():
                        hooks = register_cross_attention_hooks(pipe.unet, attn_orig_dict, device)
                        _ = pipe.unet(x_latent_orig, timestep=t_batch, encoder_hidden_states=prompt_embed)
                        remove_hooks(hooks)
                    
                    target_attn_dict = target_attn_per_t[t_scalar]
                    L_target, L_sup = 0, 0
                    total_L_sup, total_L_tar = 0, 0,
                    # A: x_non+delta, c_tar  S: x_non+delta, c_non  O: x_non, c_non,  T:  x_tar+delta, c_tar
                    for key in target_attn_dict:
                        A, S, O, T = attn_adv_dict_tar[key], attn_adv_dict_non[key], attn_orig_dict[key], target_attn_dict[key]
                        A, S, O, T = A[0] if isinstance(A, tuple) else A, S[0] if isinstance(O, tuple) else S, O[0] if isinstance(O, tuple) else O, T[0] if isinstance(T, tuple) else T
                        if A.shape[1:] != T.shape[1:]: continue
                        A, S, O, T = A.to(delta.device), S.to(delta.device), O.to(delta.device), T.to(delta.device)    
                        A, S, O, T = check(A), check(S), check(O), check(T.repeat(A.size(0), 1, 1))


                        
                        # targeted loss
                        L_target -= torch.norm(A - T, p=2)
                        # suppression loss
                        L_sup += torch.norm(S - O, p=2)
                        
                    loss = L_target+L_sup
                    grad_t = torch.autograd.grad(loss, delta, retain_graph=False)[0]
                    grad_accum += grad_t
                    total_L_tar += L_target.detach()

                    
                grad_accum /= len(timestep_set)
                
                with torch.no_grad():
                    dist.all_reduce(grad_accum, op=dist.ReduceOp.SUM)
                    grad_accum /= dist.get_world_size()
                    delta += alpha * grad_accum.sign()
                    delta.clamp_(-epsilon, epsilon)
                    
                print(f"[Step {step}, {batch_idx}] Target: {total_L_tar.item():.4f}")
                
            torch.save(delta, f"uap_{step}.pt")
    # data-free
    else:
        print("Data-free start")
        bs, fre = 1, 1 # batch size and freqency of jigsaw
        with torch.no_grad():
            for t in timestep_set:
                t_img = preprocess(target_img).unsqueeze(0).to(device)
                t_img = normalize(t_img)
                t_embed, t_embed_uncon = pipe.encode_prompt(target_prompt, device=device, num_images_per_prompt=1, do_classifier_free_guidance=True)
                t_tensor = torch.tensor([t], device=device, dtype=torch.long)
                attn_dict = {}
                hooks = register_cross_attention_hooks(pipe.unet, attn_dict, device=device)
                t_latents = pipe.prepare_latents(t_img, t_tensor, t_img.size(0), 1, t_img.dtype, device)
                _ = pipe.unet(t_latents, timestep=t_tensor, encoder_hidden_states=t_embed)
                remove_hooks(hooks)
                target_attn_per_t[int(t)] = attn_dict
                
        for batch_idx in range(5000):  
            bs, fre = curriculum_strategy_jigsaw(batch_idx, bs=bs, fre=fre)
            target_img_ = preprocess(target_img).unsqueeze(0).to(device)
            x = get_jigsaw(target_img_, bs=bs, fre=1, min=0, max=256, filter=True).to(device)
            prompts = ["Jigsaw Noise"] * x.size(0)
            x_normed = normalize(x)

            grad_accum = torch.zeros_like(delta)
            total_L_tar = 0

            for t in timestep_set:
                t_tensor = torch.tensor([t], device=device, dtype=torch.long)
                t_scalar = int(t_tensor.item())
                batch_size = x_normed.size(0)
                t_batch = torch.full((batch_size,), t_scalar, dtype=torch.long, device=device)

                delta = delta.detach().clone().requires_grad_()
                x_adv = normalize(torch.clamp(x.half() + delta, 0, 1))
                x_latent_adv = pipe.prepare_latents(x_adv, t_tensor, batch_size, 1, x_adv.dtype, device)

                attn_adv_dict_tar = {}
                hooks = register_cross_attention_hooks(pipe.unet, attn_adv_dict_tar, device)
                t_embed_new = t_embed.expand(x_latent_adv.shape[0], -1, -1).contiguous()
                _ = pipe.unet(x_latent_adv, timestep=t_batch, encoder_hidden_states=t_embed_new)
                remove_hooks(hooks)

                target_attn_dict = target_attn_per_t[t_scalar]
                L_target = 0

                for key in target_attn_dict:
                    A, T = attn_adv_dict_tar[key], target_attn_dict[key]
                    A = A[0] if isinstance(A, tuple) else A
                    T = T[0] if isinstance(T, tuple) else T
                    if A.shape[1:] != T.shape[1:]: continue
                    A, T = A.to(delta.device), T.to(delta.device)
                    A, T = A.to(delta.device), T.to(delta.device)    
                    A, T = check(A), check(T.repeat(A.size(0), 1, 1))
                    L_target -= torch.norm(A - T, p=2)

                grad_t = torch.autograd.grad(L_target, delta, retain_graph=False)[0]
                grad_accum += grad_t
                total_L_tar += L_target.detach()

            grad_accum /= len(timestep_set)

            with torch.no_grad():
                dist.all_reduce(grad_accum, op=dist.ReduceOp.SUM)
                grad_accum /= dist.get_world_size()
                delta += alpha * grad_accum.sign()
                delta.clamp_(-epsilon, epsilon)

            print(f"[Target: {total_L_tar.item():.4f}")
        
    torch.cuda.synchronize()
    end = time.time()
    latency_ms = (end - start) * 1000  # ms 
    used_memory = torch.cuda.max_memory_allocated() / 1024**3  # bytes → GB

    print(f"Latency: {latency_ms:.2f} ms")
    print(f"Peak GPU Usage: {used_memory:.2f} GB")
    return delta.detach()

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def main_ddp(rank, world_size):
    setup(rank, world_size)
    device = f"cuda:{rank}"
    img_size = 512
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        lambda x: x.half()
    ])
    dataset = ImageTextPairDataset(
        "/workspace/data/laion_aesthetic_samples/save_image",
        "/workspace/data/laion_aesthetic_samples/save_txt",
        transform=transform,
        max_samples=10000
    )
    sampler = distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    
    # 8 batch size with 8 GPUs, total 64 batch size
    dataloader = DataLoader(dataset, batch_size=8, sampler=sampler)
    
    # load surrogate model SD V1.5
    pipe = load_pipeline(device=device)
    pipe.unet = DDP(pipe.unet, device_ids=[rank])
    
    # set your target image and prompt
    target_img = Image.open("../target/tiger.png").convert("RGB")
    target_prompt = "tiger"

    # if you want 'data-free setting'
    dataloader = None
    
    # generate UAP
    delta = MMAP(pipe, target_img, target_prompt, dataloader, rank, device, uap=None)
    if rank == 0:
        torch.save(delta, "uap.pt")
        print("Saved UAP to uap.pt")
    cleanup()


        
if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(main_ddp, args=(world_size,), nprocs=world_size, join=True)