""" Example handler file. """

import runpod
from diffusers import AutoPipelineForImage2Image
import torch
import base64
import io
import time
from PIL import Image
from io import BytesIO


# If your handler runs inference on a model, load the model here.
# You will want models to be loaded into memory before starting serverless.

try:
    pipe = AutoPipelineForImage2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
    pipe.to("cuda")
except RuntimeError:
    quit()

def handler(job):
    """ Handler function that will be used to process jobs. """
    job_input = job['input']
    prompt = job_input['prompt']
    image = job_input['image']
    img = Image.open(BytesIO(base64.b64decode(image))).convert("RGB")
    img.thumbnail((768, 768))

    time_start = time.time()
    image = pipe(prompt=prompt, num_inference_steps=2, guidance_scale=0.0, image=img).images[0]
    print(f"Time taken: {time.time() - time_start}")

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    image_bytes = buffer.getvalue()

    return base64.b64encode(image_bytes).decode('utf-8')


runpod.serverless.start({"handler": handler})
