from diffusers import StableDiffusionPipeline
import os
import torch
pipe_diffusion = StableDiffusionPipeline.from_pretrained('CompVis/stable-diffusion-v1-4',torch_dtype=torch.float16).to('cuda')

def image_gen(prompt_list,output_dir="generated_images"):
    #prompt list contains image for a sinagle post
    os.makedirs(output_dir, exist_ok=True)
    image_path_list=[]

    for prompt in prompt_list:
        image = pipe_diffusion(prompt).images[0]

    # Create a safe filename using the first 20 characters of the prompt
        safe_prompt = "".join([c if c.isalnum() or c in (' ', '_') else '' for c in prompt])[:10].replace(" ", "_")
        image_path = os.path.join(output_dir, f"{safe_prompt}.png")
        image_path_list.append(image_path)
        image.save(image_path)
    return image_path_list

