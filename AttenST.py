import os
import copy
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from pytorch_lightning import seed_everything
from transformers import AutoProcessor, Blip2ForConditionalGeneration, CLIPVisionModelWithProjection
from diffusers_utils.stable_diffusion import decode_latent, attention_op, get_sdxlunet_layers

from src.eunms import Model_Type, Scheduler_Type
from src.utils.enums_utils import get_pipes
from src.config import RunConfig

from inversion import run as SP_inversion
from diffusers_utils.config import get_args


def generate_caption(
        image: Image.Image,
        text: str = None,
        decoding_method: str = "Nucleus sampling",
        temperature: float = 1.0,
        length_penalty: float = 1.0,
        repetition_penalty: float = 1.5,
        max_length: int = 50,
        min_length: int = 1,
        num_beams: int = 5,
        top_p: float = 0.9,
) -> str:
    """
    Generate a caption for an image using the BLIP2 model.

    Args:
        image (Image.Image): Input image.
        text (str, optional): Optional text input. 
        decoding_method (str, optional): Decoding method. Defaults to "Nucleus sampling".
        temperature (float, optional): Temperature parameter. 
        length_penalty (float, optional): Length penalty. 
        repetition_penalty (float, optional): Repetition penalty. 
        max_length (int, optional): Maximum caption length. 
        min_length (int, optional): Minimum caption length. 
        num_beams (int, optional): Number of beams for beam search.
        top_p (float, optional): Top-p parameter for nucleus sampling.

    Returns:
        str: Generated caption for the image.
    """
    if text is not None:
        inputs = processor(images=image, text=text, return_tensors="pt").to("cuda", torch.float16)
        generated_ids = model.generate(**inputs)
    else:
        inputs = processor(images=image, return_tensors="pt").to("cuda", torch.float16)
        generated_ids = model.generate(
            pixel_values=inputs.pixel_values,
            do_sample=decoding_method == "Nucleus sampling",
            temperature=temperature,
            length_penalty=length_penalty,
            repetition_penalty=repetition_penalty,
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            top_p=top_p,
        )
    result = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    return result


class AttenST:
    """
    Attention-based Style Transfer class for handling style transfer tasks.
    """

    def __init__(self, unet, vae, text_encoder, tokenizer, scheduler, style_guided_layers=None):
        """
        Initialize the AttenST class.

        Args:
            unet: SD UNet model.
            vae: VAE model.
            text_encoder: Text encoder.
            tokenizer: Tokenizer.
            scheduler: Scheduler.
            style_guided_layers (dict, optional): Style transfer parameters. Defaults to None.
        """
        style_guided_layers_default = {
            'transformer_blocks_num': [4, 5]  # Layers where style-guided attention is applied
        }
        if style_guided_layers is not None:
            style_guided_layers_default.update(style_guided_layers)
        self.style_guided_layers = style_guided_layers_default

        self.unet = unet  # SD UNet
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.scheduler = scheduler

        self.SA_deatures = {}  # Dictionary to store attention features
        self.SA_deatures_modify = {}  # Dictionary to store modified attention features

        self.cur_t = None  # Current timestep

        _, attn = get_sdxlunet_layers(unet)

        # Register hooks only for specified layers
        for i, transformer_model in enumerate(attn):
            if i not in self.style_guided_layers['transformer_blocks_num']:
                continue

            if not hasattr(transformer_model, "transformer_blocks") or not isinstance(
                    transformer_model.transformer_blocks, nn.ModuleList):
                raise AttributeError(f"Layer {i} is not a valid Transformer2DModel with transformer_blocks.")

            for j, transformer_block in enumerate(transformer_model.transformer_blocks):
                layer_name = f"layer{i}_block{j}_attn"
                self.SA_deatures[layer_name] = {}

                if hasattr(transformer_block, "attn1") and isinstance(transformer_block.attn1, nn.Module):
                    transformer_block.attn1.register_forward_hook(self.__get_query_key_value(layer_name))
                    transformer_block.attn1.register_forward_hook(self.__modify_self_attn_qkv(layer_name))
                else:
                    raise AttributeError(f"Transformer block {j} in layer {i} does not have an 'attn1' module.")

        self.save_SA_features = False  # Whether to save attention features
        self.modify_attention_features = False  # Whether to modify attention features

    def update_cur_t(self, t):
        """
        Callback function to update the current timestep.

        Args:
            t: Current timestep.
        """
        self.cur_t = t

    def SPI_inversion(self, input_image, prompt, config, pipe_inversion, pipe_inference, vae=None):
        """
        Perform Style preserving inversion.

        Args:
            input_image: Input image.
            prompt: Prompt text.
            config: Configuration object.
            pipe_inversion: Inversion pipeline.
            pipe_inference: Inference pipeline.
            vae: VAE model.

        Returns:
            tuple: Inverted latent, all latents, and predicted images.
        """
        pred_images = []

        _, inv_latent, _, all_latents = SP_inversion(input_image,
                                               prompt,
                                               config,
                                               pipe_inversion=pipe_inversion,
                                               pipe_inference=pipe_inference,
                                               do_reconstruction=False,
                                               callback_t=self.update_cur_t,
                                               )
        for img in all_latents:
            pred_images.append(decode_latent(img, vae))

        return inv_latent, all_latents, pred_images

    def __get_query_key_value(self, name):
        """
        Hook function to get query, key, and value from attention layers.

        Args:
            name: Layer name.

        Returns:
            function: Hook function.
        """
        def hook(model, input, output):
            if self.save_SA_features:
                _, query, key, value, _ = attention_op(model, input[0])
                self.SA_deatures[name][int(self.cur_t)] = (query.detach(), key.detach(), value.detach())

        return hook

    def __modify_self_attn_qkv(self, name):
        """
        Hook function to modify query, key, and value in self-attention layers.

        Args:
            name: Layer name.

        Returns:
            function: Hook function.
        """
        def hook(model, input, output):
            if self.modify_attention_features:
                _, q_cs, k_cs, v_cs, _ = attention_op(model, input[0])
                q_c, k_s, v_s = self.SA_deatures_modify[name][int(self.cur_t)]

                if q_c.shape[0] != q_cs.shape[0]:
                    q_c = torch.cat([q_c] * 2)

                q_hat_cs = q_c * 0.75 + q_cs * (1 - 0.75)

                if k_cs.shape[0] != k_s.shape[0]:
                    k_s = torch.cat([k_s] * 2)
                    v_s = torch.cat([v_s] * 2)

                k_cs, v_cs = k_s, v_s

                _, _, _, _, modified_output = attention_op(model, input[0], key=k_cs, value=v_cs, query=q_hat_cs,
                                                           temperature=1.5)

                return modified_output

        return hook


if __name__ == "__main__":
    cfg = get_args()

    # Directory to save results
    save_dir = cfg.save_dir
    style_path = cfg.sty_path
    cnt_path = cfg.cnt_path

    os.makedirs(save_dir, exist_ok=True)

    # Set random seed and device
    seed_everything(22)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    guidance_scale = 1.  # No text guidance

    # Load BLIP2 model for prompt generation
    MODEL_ID = "Salesforce/blip2-opt-2.7b"
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = Blip2ForConditionalGeneration.from_pretrained(MODEL_ID, device_map="cuda", load_in_8bit=False,
                                                          torch_dtype=torch.float16,
                                                          revision="51572668da0eb669e01a189dc22abe6088589a24")
    model.eval()

    # Generate caption for content image
    cnt = Image.open(cnt_path).convert("RGB").resize((512, 512))
    content_image_prompt = generate_caption(cnt)
    print(content_image_prompt)

    # Generate caption for style image
    sty = Image.open(style_path).convert("RGB").resize((512, 512))
    style_image_prompt = generate_caption(sty)
    print(style_image_prompt)

    # Free up memory by deleting the model
    del model
    del processor
    torch.cuda.empty_cache()

    # Set style transfer parameters
    style_guided_layers = {
        'transformer_blocks_num': [4, 5],  # Layers where style-guided attention is applied
    }

    # Load image encoder
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        "/model_checkpoints/IP-Adapter", subfolder="models/image_encoder",
        torch_dtype=torch.float16
    ).to(device)

    # Set up model type and scheduler type for SPI inversion
    model_type = Model_Type.SDXL
    scheduler_type = Scheduler_Type.DDIM
    pipe_inversion, pipe_inference = get_pipes(model_type=model_type, scheduler_type=scheduler_type, device=device,
                                               controlnet=None,
                                               image_encoder=image_encoder)
    scheduler = pipe_inference.scheduler

    # Configure run parameters
    config = RunConfig(model_type=model_type,
                       num_inference_steps=20,
                       num_inversion_steps=20,
                       resample_steps=5,
                       scheduler_type=scheduler_type,
                       guidance_scale=guidance_scale,
                       seed=22)
    scheduler.set_timesteps(config.num_inversion_steps)
    timesteps = scheduler.timesteps

    # Initialize AttenST class
    attenst = AttenST(pipe_inference.unet, pipe_inference.vae, pipe_inference.text_encoder,
                                         pipe_inference.tokenizer, pipe_inference.scheduler,
                                         style_guided_layers=style_guided_layers)

    # Get attention features for style image
    attenst.save_SA_features = True
    attenst.modify_attention_features = False

    inv_style_latent, all_style_latents, pred_images = attenst.SPI_inversion(input_image=sty,
                                                                                   prompt=style_image_prompt,
                                                                                   config=config,
                                                                                   pipe_inversion=pipe_inversion,
                                                                                   pipe_inference=pipe_inference,
                                                                                   vae=attenst.vae)

    # Save attention features for style image
    style_features = copy.deepcopy(attenst.SA_deatures)

    # Get attention features for content image
    attenst.save_SA_features = True
    attenst.modify_attention_features = False

    inv_cnt_latent, all_cnt_latents, pred_cnt_images = attenst.SPI_inversion(input_image=cnt,
                                                                                   prompt=content_image_prompt,
                                                                                   config=config,
                                                                                   pipe_inversion=pipe_inversion,
                                                                                   pipe_inference=pipe_inference,
                                                                                   vae=attenst.vae)

    # Save attention features for content image
    content_features = copy.deepcopy(attenst.SA_deatures)

    # Modify attention features for style transfer
    count = 0
    for t in scheduler.timesteps:
        t = t.item()
        for layer_name in style_features.keys():
            if not count:
                attenst.SA_deatures_modify[layer_name] = {}

            attenst.SA_deatures_modify[layer_name][t] = (
                content_features[layer_name][t][0], style_features[layer_name][t][1],
                style_features[layer_name][t][2])  # Use content as query and style as key/value
        count = 1

    # Enable modification of attention features
    attenst.save_SA_features = False
    attenst.modify_attention_features = True

    # Set content and style weights
    content_weight = 0.4  # Higher weight enhances content preservation
    style_weight = 0.6  # Higher weight enhances style incorporation

    print(f"content_weight:{content_weight},style_weight:{style_weight}")

    # Decide whether to use ADAIN based on configuration
    if cfg.without_init_adain:
        print("without adain....")
        latent_cs = inv_cnt_latent
    else:
        print("content-aware adain....")
        latent_cs = (inv_cnt_latent - inv_cnt_latent.mean(dim=(2, 3), keepdim=True)) / (
                inv_cnt_latent.std(dim=(2, 3), keepdim=True) + 1e-4) * \
                    (style_weight * inv_style_latent.std(dim=(2, 3), keepdim=True) + content_weight * inv_cnt_latent.std(
                        dim=(2, 3), keepdim=True)) + \
                    (style_weight * inv_style_latent.mean(dim=(2, 3),
                                                          keepdim=True) + content_weight * inv_cnt_latent.mean(dim=(2, 3), keepdim=True))

    # Free up memory by deleting the inversion pipeline
    del pipe_inversion
    torch.cuda.empty_cache()
    print("Style transfer...")

    # Load IP-Adapter
    pipe_inference.load_ip_adapter(
        ["/model_checkpoints/IP-Adapter",
         "/model_checkpoints/IP-Adapter",
         ],
        subfolder=["sdxl_models", "sdxl_models"],
        weight_name=[
            "ip-adapter_sdxl_vit-h.safetensors",
            "ip-adapter_sdxl_vit-h.safetensors",
        ],
        image_encoder_folder=None,
    )

    # Set content and style scaling
    scale_cnt = {
        "down": {"block_2": [0, 1.2]},
    }
    scale_style = {
        "up": {"block_0": [0.0, 1.2, 1.2]},
    }
    pipe_inference.set_ip_adapter_scale([scale_cnt, scale_style])

    # Generate the final stylized image
    cnt_pompt2 = content_image_prompt + " best quality, high quality,intricate details"

    stylized_image = \
        pipe_inference(
            prompt=cnt_pompt2,
            negative_prompt="lowres, low quality, worst quality, deformed, noisy, blurry",
            num_inference_steps=config.num_inference_steps,
            ip_adapter_image=[cnt, sty],
            image=latent_cs,
            strength=1.0,
            denoising_start=0.0,
            guidance_scale=guidance_scale,
            callback_t=attenst.update_cur_t,
        ).images[0]

    # Save the stylized image
    content_name = os.path.splitext(os.path.basename(cnt_path))[0]
    style_name = os.path.splitext(os.path.basename(style_path))[0]
    save_path = os.path.join(save_dir, f"{content_name}_{style_name}.png")
    stylized_image.save(save_path)
    torch.cuda.empty_cache()
