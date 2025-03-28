import torch
from PIL import Image

from src.eunms import Model_Type, Scheduler_Type
from src.utils.enums_utils import get_pipes
from src.config import RunConfig

from main import run as invert



device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_type = Model_Type.SDXL
scheduler_type = Scheduler_Type.DDIM
pipe_inversion, pipe_inference = get_pipes(model_type, scheduler_type, device=device)

input_image = Image.open("/home/hqz/disk3/hb/styleID_server/data/cnt/1.png").convert("RGB").resize((1024, 1024))
prompt = "the golden gate bridge in san francisco, california"

config = RunConfig(model_type = model_type,
                    num_inference_steps = 10,
                    num_inversion_steps = 10,
                    num_renoise_steps = 1,
                    scheduler_type = scheduler_type,
                    perform_noise_correction = False,
                    seed = 7865)

_, inv_latent, _, all_latents = invert(input_image,
                                       prompt,
                                       config,
                                       pipe_inversion=pipe_inversion,
                                       pipe_inference=pipe_inference,
                                       do_reconstruction=False)

rec_image = pipe_inference(image = inv_latent,
                           prompt = prompt,
                           denoising_start=0.0,
                           num_inference_steps = config.num_inference_steps,
                           guidance_scale = 1.0).images[0]

rec_image.save("lion_reconstructed.jpg")
