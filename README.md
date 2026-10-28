# Universal Image Immunization against Diffusion-based Image Editing via Semantic Injection

Official PyTorch implementation of **Universal Image Immunization against Diffusion-based Image Editing via Semantic Injection**, accepted to **ECCV 2026**.

[[Paper]](https://arxiv.org/abs/2602.14679)

We introduce a universal image immunization method that learns a single, image-agnostic perturbation for diffusion-based image editing. The perturbation injects target semantics while suppressing the source-image semantics to disrupt unauthorized edits.

## Environment

The code was tested with:

- Ubuntu 20.04
- Python 3.10.16
- PyTorch 2.6.0
- Torchvision 0.21.0
- diffusers 0.32.2
- CUDA 11.6

Install the required Python packages:

```bash
pip install torch==2.6.0 torchvision==0.21.0
pip install diffusers==0.32.2 transformers accelerate pillow numpy scikit-image
```

The Stable Diffusion v1.5 surrogate model is downloaded automatically through `diffusers` on the first run.

## Usage

In `main.py`, set `target_img` to your target-image path and update `target_prompt` accordingly.

The released configuration uses data-free training by setting `dataloader = None`. To use LAION-Aesthetics samples instead, set the image and text directories in `ImageTextPairDataset` and remove the `dataloader = None` assignment.

Run UAP training with:

```bash
python main.py
```

The script uses all GPUs exposed through `CUDA_VISIBLE_DEVICES` and saves the final perturbation as `uap.pt`.

## Citation

```bibtex
@inproceedings{lee2026universal,
  title={Universal Image Immunization against Diffusion-based Image Editing via Semantic Injection},
  author={Lee, Chanhui and Choi, Donggyu and Shin, Seunghyun and Jeon, Hae-gon and Son, Jeany},
  booktitle={European Conference on Computer Vision},
  organization={Springer},
  year={2026}
}
```
