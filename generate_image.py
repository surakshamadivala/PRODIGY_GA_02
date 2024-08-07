from diffusers import StableDiffusionPipeline

# Correct model ID
model_id = "runwayml/stable-diffusion-v1-5"

# Load the Stable Diffusion pipeline
pipeline = StableDiffusionPipeline.from_pretrained(model_id)

# Use the pipeline to generate an image
prompt = "A beautiful landscape with mountains and a river"
image = pipeline(prompt).images[0]

# Save the generated image
image.save("generated_image.png")
