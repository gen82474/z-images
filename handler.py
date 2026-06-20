import os
import io
import base64
import torch
from diffusers import ZImagePipeline
from PIL import Image
import runpod

# ------------------------------------------------------------------
# Load model once at worker startup
# ------------------------------------------------------------------
MODEL_ID = os.environ.get("MODEL_ID", "Tongyi-MAI/Z-Image-Turbo")

print("Loading Z-Image-Turbo model...")
pipe = ZImagePipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    use_safetensors=True,
    low_cpu_mem_usage=False,
).to("cuda")
print("Model loaded successfully.")


def encode_image(image: Image.Image) -> str:
    """Convert a PIL image to a base64 PNG data URI."""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode("utf-8")


def validate_resolution(width: int, height: int) -> tuple[int, int]:
    """Clamp and round resolution to valid VAE dimensions."""
    width = (int(width) // 16) * 16
    height = (int(height) // 16) * 16
    width = max(512, min(2048, width))
    height = max(512, min(2048, height))
    return width, height


def handler(job):
    job_input = job.get("input", {})

    # ------------------------------------------------------------------
    # API key check (optional, set API_KEY env var to enable)
    # ------------------------------------------------------------------
    expected_api_key = os.environ.get("API_KEY")
    if expected_api_key:
        input_api_key = job_input.get("api_key")
        if input_api_key != expected_api_key:
            return {"error": "Unauthorized: Invalid or missing 'api_key'"}

    # ------------------------------------------------------------------
    # Parse prompts
    # ------------------------------------------------------------------
    prompts_raw = job_input.get("prompt", "")
    if isinstance(prompts_raw, str):
        prompts = [prompts_raw]
    elif isinstance(prompts_raw, list):
        prompts = prompts_raw
    else:
        return {"error": "prompt must be a string or list of strings"}

    if not prompts or any(not isinstance(p, str) or not p.strip() for p in prompts):
        return {"error": "all prompts must be non-empty strings"}

    # ------------------------------------------------------------------
    # Parse resolutions (defaults to 1024x1024)
    # ------------------------------------------------------------------
    resolutions = job_input.get("resolutions", [{"width": 1024, "height": 1024}])
    if not isinstance(resolutions, list) or not resolutions:
        return {"error": "resolutions must be a non-empty list"}

    # ------------------------------------------------------------------
    # Turbo parameters
    # ------------------------------------------------------------------
    num_inference_steps = int(job_input.get("num_inference_steps", 9))
    guidance_scale = float(job_input.get("guidance_scale", 0.0))
    num_images_per_prompt = int(job_input.get("num_images_per_prompt", 1))

    seed = job_input.get("seed")
    generator = None
    if seed is not None:
        generator = torch.Generator(device="cuda").manual_seed(int(seed))

    # ------------------------------------------------------------------
    # Generate images for each resolution
    # ------------------------------------------------------------------
    results = []

    for res in resolutions:
        width, height = validate_resolution(res.get("width", 1024), res.get("height", 1024))

        with torch.inference_mode():
            output = pipe(
                prompt=prompts,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                num_images_per_prompt=num_images_per_prompt,
                generator=generator,
            )

        images_b64 = [encode_image(img) for img in output.images]

        results.append({
            "resolution": {"width": width, "height": height},
            "images": images_b64,
        })

    return {
        "success": True,
        "results": results,
        "metadata": {
            "prompts": prompts,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "num_images_per_prompt": num_images_per_prompt,
            "seed": seed,
        },
    }


runpod.serverless.start({"handler": handler})
