import os
import sys
import torch
import folder_paths
from PIL import Image
import numpy as np


# Add the parent directory to the Python path so we can import from easycontrol
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from InstantCharacter.pipeline import InstantCharacterFluxPipeline
from huggingface_hub import login


if "ipadapter" not in folder_paths.folder_names_and_paths:
    current_paths = [os.path.join(folder_paths.models_dir, "ipadapter")]
else:
    current_paths, _ = folder_paths.folder_names_and_paths["ipadapter"]
folder_paths.folder_names_and_paths["ipadapter"] = (current_paths, folder_paths.supported_pt_extensions)


class InstantCharacterLoadModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "hf_token": ("STRING", {"default": "", "multiline": True}),
                "ip_adapter_name": (folder_paths.get_filename_list("ipadapter"), ),
                "cpu_offload": ("BOOLEAN", {"default": False})
            }
        }
    
    RETURN_TYPES = ("INSTANTCHAR_PIPE",)
    FUNCTION = "load_model"
    CATEGORY = "InstantCharacter"

    def load_model(self, hf_token, ip_adapter_name, cpu_offload):
        login(token=hf_token)
        base_model = "black-forest-labs/FLUX.1-dev"
        image_encoder_path = "google/siglip-so400m-patch14-384"
        image_encoder_2_path = "facebook/dinov2-giant"
        cache_dir = folder_paths.get_folder_paths("diffusers")[0]
        image_encoder_cache_dir = folder_paths.get_folder_paths("clip_vision")[0]
        image_encoder_2_cache_dir = folder_paths.get_folder_paths("clip_vision")[0]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        ip_adapter_path = folder_paths.get_full_path("ipadapter", ip_adapter_name)
        
        pipe = InstantCharacterFluxPipeline.from_pretrained(
            base_model, 
            torch_dtype=torch.bfloat16,
            cache_dir=cache_dir,    
        )
        if cpu_offload:
            pipe.enable_sequential_cpu_offload()
        else:
            pipe.to(device)
        
        pipe.init_adapter(
            image_encoder_path=image_encoder_path,
            cache_dir=image_encoder_cache_dir,
            image_encoder_2_path=image_encoder_2_path,
            cache_dir_2=image_encoder_2_cache_dir,
            subject_ipadapter_cfg=dict(
                subject_ip_adapter_path=ip_adapter_path,
                nb_token=1024
            ),
        )
        
        return (pipe,)


class InstantCharacterGenerate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipe": ("INSTANTCHAR_PIPE",),
                "prompt": ("STRING", {"multiline": True}),
                "height": ("INT", {"default": 768, "min": 256, "max": 2048, "step": 64}),
                "width": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 64}),
                "guidance_scale": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 10.0, "step": 0.1}),
                "num_inference_steps": ("INT", {"default": 28, "min": 1, "max": 100, "step": 1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "subject_scale": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 2.0, "step": 0.1}),
            },
            "optional": {
                "subject_image": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "InstantCharacter"

    def generate(self, pipe, prompt, height, width, guidance_scale, 
                num_inference_steps, seed, subject_scale, subject_image=None):
        
        # Convert subject image from tensor to PIL if provided
        subject_image_pil = None
        if subject_image is not None:
            if isinstance(subject_image, torch.Tensor):
                if subject_image.dim() == 4:  # [batch, height, width, channels]
                    img = subject_image[0].cpu().numpy()
                else:  # [height, width, channels]
                    img = subject_image.cpu().numpy()
                subject_image_pil = Image.fromarray((img * 255).astype(np.uint8))
            elif isinstance(subject_image, np.ndarray):
                subject_image_pil = Image.fromarray((subject_image * 255).astype(np.uint8))
        
        # Generate image
        output = pipe(
            prompt=prompt,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=torch.Generator("cpu").manual_seed(seed),
            subject_image=subject_image_pil,
            subject_scale=subject_scale,
        )
        
        # Convert PIL image to tensor format
        image = np.array(output.images[0]) / 255.0
        image = torch.from_numpy(image).float()
        
        # Add batch dimension if needed
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        return (image,)

# --- New Node for Guff Integration (Prepare Sampler Inputs) ---
class InstantCharacterGuffSamplerInputs:
    """
    Prepares inputs for a standard ComfyUI sampler node like KSampler or Guff.
    This node allows using the InstantCharacter prompts/latent setup with
    an external sampler node for more control, specifically designed to
    facilitate using the Guff node for Flux diffusion manually in a workflow.
    It takes a standard MODEL input, allowing connection from a model loader.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",), # Connect the MODEL output from your Guff-compatible model loader here
                "positive_prompt": ("STRING", {"multiline": True}),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}), # Default to empty string as requested
                "latent_image": ("LATENT",), # Connect your Empty Latent Image or VAE Encode output here
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                # Sampler/Scheduler inputs are omitted here, as they are typically
                # selected directly on the KSampler/Guff node itself.
                # Note: Subject image handling is specific to the InstantCharacter pipeline
                # and is not included here as it's not a standard sampler input.
                # You would need separate IP-Adapter or conditioning nodes if needed.
            }
        }

    # Define the output types and names. These match standard sampler inputs.
    RETURN_TYPES = ("MODEL", "LATENT", "CONDITIONING", "CONDITIONING", "INT", "FLOAT", "INT")
    RETURN_NAMES = ("model", "latent", "positive", "negative", "seed", "steps", "cfg")
    FUNCTION = "prepare_inputs"
    CATEGORY = "InstantCharacter/Guff" # Put it in a subcategory for organization

    def prepare_inputs(self, model, positive_prompt, negative_prompt, latent_image, seed, steps, cfg):
        """
        Generates conditioning tensors from prompts using the provided MODEL
        and passes through other sampler inputs.
        """
        print("Executing InstantCharacter: Prepare for Guff/Sampler node")

        # Get conditioning from prompts using the model's CLIP/tokenizer
        # This requires the MODEL to have a clip component (like SDXL or Flux models)
        try:
            # comfy.sd.get_conditioning returns a tuple: (tensor, pooled_output)
            positive_cond, positive_pooled = comfy.sd.get_conditioning(model, positive_prompt, [])
            negative_cond, negative_pooled = comfy.sd.get_conditioning(model, negative_prompt, [])

            # Standard format for conditioning output in ComfyUI
            positive_conditioning = [[positive_cond, {"pooled_output": positive_pooled}]]
            negative_conditioning = [[negative_cond, {"pooled_output": negative_pooled}]]

        except Exception as e:
             print(f"Error generating conditioning in Guff prep node: {e}")
             # Raise an error if conditioning fails, as the sampler won't work without it.
             raise ValueError(f"Could not generate conditioning. Ensure the loaded MODEL is compatible (e.g., SDXL or Flux model with CLIP). Error: {e}")

        # Return the prepared inputs. These can be connected directly to a KSampler or Guff node.
        return (model, latent_image, positive_conditioning, negative_conditioning, seed, steps, cfg)
