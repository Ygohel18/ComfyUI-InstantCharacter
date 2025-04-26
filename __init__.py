# 注册节点
from .nodes.comfy_nodes import InstantCharacterLoadModel, InstantCharacterGenerate, InstantCharacterGuffSamplerInputs

NODE_CLASS_MAPPINGS = {
    "InstantCharacterLoadModel": InstantCharacterLoadModel, # Keep existing node
    "InstantCharacterGenerate": InstantCharacterGenerate, # Keep existing node
    "InstantCharacter_Guff_Sampler_Inputs": InstantCharacter_Guff_Sampler_Inputs, # Add new node
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "InstantCharacterLoadModel": "Instant Character: Load Pipeline", # Display name for existing node
    "InstantCharacterGenerate": "Instant Character: Generate (Pipeline)", # Display name for existing node
    "InstantCharacter_Guff_Sampler_Inputs": "Instant Character: Prepare for Guff/Sampler", # Display name for new node
}

WEB_DIRECTORY = "./web"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', "WEB_DIRECTORY"]