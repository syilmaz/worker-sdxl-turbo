""" Example handler file. """

import runpod
from diffusers import AutoPipelineForImage2Image
import torch
import base64
import io
import time
from PIL import Image
from io import BytesIO
import requests
import math


# If your handler runs inference on a model, load the model here.
# You will want models to be loaded into memory before starting serverless.

try:
    pipe = AutoPipelineForImage2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
    pipe.to("cuda")
except RuntimeError:
    quit()


def resize_crop(image, size=512):
    image = image.convert("RGB")
    w, h = image.size
    image = image.resize((size, int(size * (h / w))), Image.BICUBIC)
    return image

def handler(job):
    """ Handler function that will be used to process jobs. """
    job_input = job['input']
    prompt = job_input['prompt']
    img = job_input['image']
    # img = Image.open(BytesIO(base64.b64decode(image))).convert("RGB")
    # img.thumbnail((768, 768))

    # time_start = time.time()
    # image = pipe(prompt=prompt, num_inference_steps=2, guidance_scale=0.0, image=img).images[0]
    # print(f"Time taken: {time.time() - time_start}")

    # url = "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0f/1665_Girl_with_a_Pearl_Earring.jpg/800px-1665_Girl_with_a_Pearl_Earring.jpg"

    # response = requests.get(url)
    image = Image.open(BytesIO(base64.b64decode(image))).convert("RGB")

    strength = 0.7
    steps = 2
    seed = 1231231
    init_image = resize_crop(image)
    generator = torch.manual_seed(seed)
    last_time = time.time()

    if int(steps * strength) < 1:
        steps = math.ceil(1 / max(0.10, strength))
        
    results = pipe(
        prompt=prompt,
        image=init_image,
        generator=generator,
        num_inference_steps=steps,
        guidance_scale=0.0,
        strength=strength,
        width=512,
        height=512,
        output_type="pil",
    )

    buffer = io.BytesIO()
    results.images[0].save(buffer, format="PNG")
    image_bytes = buffer.getvalue()

    return base64.b64encode(image_bytes).decode('utf-8')


runpod.serverless.start({"handler": handler})
