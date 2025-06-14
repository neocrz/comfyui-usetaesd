import os
import torch
import comfy.sd
import comfy.utils
import folder_paths
import comfy.model_management
from nodes import MAX_RESOLUTION

# Global cache for TAESD VAE instances
_TAESD_VAE_CACHE = {}
TAESD_MODEL_NAMES = ["taesd", "taesdxl", "taesd3", "taef1"]

def get_taesd_vae_instance(model_name="taesd"):
    """
    Loads and caches a TAESD model as a comfy.sd.VAE instance.
    'model_name' can be "taesd", "taesdxl", "taesd3", "taef1", corresponding to
    files like "taesd_encoder.safetensors" and "taesd_decoder.safetensors"
    in the "models/vae_approx/" directory.
    """
    if model_name not in _TAESD_VAE_CACHE:
        loaded_vae = None
        try:
            sd = {}
            # Determine filenames based on model_name
            # Common extensions to try if .safetensors is not found
            extensions = [".safetensors", ".pt", ".bin", ".pth"]
            
            encoder_basename = f"{model_name}_encoder"
            decoder_basename = f"{model_name}_decoder"
            
            encoder_path = None
            decoder_path = None

            for ext in extensions:
                if encoder_path is None:
                    current_encoder_path = folder_paths.get_full_path("vae_approx", encoder_basename + ext)
                    if current_encoder_path and os.path.exists(current_encoder_path):
                        encoder_path = current_encoder_path
                
                if decoder_path is None:
                    current_decoder_path = folder_paths.get_full_path("vae_approx", decoder_basename + ext)
                    if current_decoder_path and os.path.exists(current_decoder_path):
                        decoder_path = current_decoder_path
                
                if encoder_path and decoder_path:
                    break
            
            if not encoder_path:
                raise FileNotFoundError(f"TAESD encoder file not found for '{model_name}' (tried extensions: {', '.join(extensions)}) in vae_approx directory. Expected name: {encoder_basename}[extension]")
            if not decoder_path:
                raise FileNotFoundError(f"TAESD decoder file not found for '{model_name}' (tried extensions: {', '.join(extensions)}) in vae_approx directory. Expected name: {decoder_basename}[extension]")

            print(f"Loading TAESD encoder from: {encoder_path}")
            print(f"Loading TAESD decoder from: {decoder_path}")

            # Load state dicts
            enc_sd = comfy.utils.load_torch_file(encoder_path, safe_load=True)
            for k_enc in enc_sd:
                sd[f"taesd_encoder.{k_enc}"] = enc_sd[k_enc]

            dec_sd = comfy.utils.load_torch_file(decoder_path, safe_load=True)
            for k_dec in dec_sd:
                sd[f"taesd_decoder.{k_dec}"] = dec_sd[k_dec]

            # Set vae_scale and vae_shift parameters, crucial for TAESD's own scaling
            if model_name == "taesd":
                sd["vae_scale"] = torch.tensor(0.18215)
                sd["vae_shift"] = torch.tensor(0.0)
            elif model_name == "taesdxl":
                sd["vae_scale"] = torch.tensor(0.13025)
                sd["vae_shift"] = torch.tensor(0.0)
            elif model_name == "taesd3":
                sd["vae_scale"] = torch.tensor(1.5305)
                sd["vae_shift"] = torch.tensor(0.0609)
            elif model_name == "taef1":
                sd["vae_scale"] = torch.tensor(0.3611)
                sd["vae_shift"] = torch.tensor(0.1159)
            else:
                print(f"Warning: Unknown TAESD model_name '{model_name}' for specific vae_scale/shift. Using defaults for 'taesd'.")
                sd["vae_scale"] = torch.tensor(0.18215)
                sd["vae_shift"] = torch.tensor(0.0)
            
            loaded_vae = comfy.sd.VAE(sd=sd)
            loaded_vae.throw_exception_if_invalid()
            
            _TAESD_VAE_CACHE[model_name] = loaded_vae
            print(f"Successfully loaded and cached TAESD VAE: {model_name}")

        except Exception as e:
            print(f"Error loading TAESD VAE '{model_name}': {e}")
            raise RuntimeError(f"Failed to load TAESD VAE '{model_name}'. Check file paths ({encoder_basename}, {decoder_basename} with extensions {extensions} in models/vae_approx/) and integrity.") from e
            
    return _TAESD_VAE_CACHE[model_name]

class EncodeTAESD:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pixels": ("IMAGE",),
                "taesd_model_name": (TAESD_MODEL_NAMES, {"default": "taesd"}),
            }
        }
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "encode"
    CATEGORY = "latent/TAESD"
    DESCRIPTION = "Encodes an image into TAESD's latent space."

    def encode(self, pixels, taesd_model_name):
        vae = get_taesd_vae_instance(taesd_model_name)
        latent_samples = vae.encode(pixels[:,:,:,:3]) 
        return ({"samples": latent_samples},)

class DecodeTAESD:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "samples": ("LATENT",),
                "taesd_model_name": (TAESD_MODEL_NAMES, {"default": "taesd"}),
            }
        }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode"
    CATEGORY = "latent/TAESD"
    DESCRIPTION = "Decodes latents from TAESD's latent space back to an image."

    def decode(self, samples, taesd_model_name):
        vae = get_taesd_vae_instance(taesd_model_name)
        pixels = vae.decode(samples["samples"])
        return (pixels,)

class EncodeTAESDTiled:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pixels": ("IMAGE",),
                "taesd_model_name": (TAESD_MODEL_NAMES, {"default": "taesd"}),
                "tile_size": ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 64, "tooltip": "Tile size for encoding (in image pixels)"}),
                "overlap": ("INT", {"default": 64, "min": 0, "max": MAX_RESOLUTION, "step": 32, "tooltip": "Overlap between tiles (in image pixels)"}),
            }
        }
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "encode_tiled"
    CATEGORY = "latent/TAESD"
    DESCRIPTION = "Encodes an image into TAESD's latent space using tiled processing."

    def encode_tiled(self, pixels, taesd_model_name, tile_size, overlap):
        vae = get_taesd_vae_instance(taesd_model_name)
        latent_samples = vae.encode_tiled(pixels[:,:,:,:3], tile_x=tile_size, tile_y=tile_size, overlap=overlap)
        return ({"samples": latent_samples},)

class DecodeTAESDTiled:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "samples": ("LATENT",),
                "taesd_model_name": (TAESD_MODEL_NAMES, {"default": "taesd"}),
                "tile_size": ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 64, "tooltip": "Tile size for decoding (image pixels, converted to latent space for VAE)"}),
                "overlap": ("INT", {"default": 64, "min": 0, "max": MAX_RESOLUTION, "step": 32, "tooltip": "Overlap between tiles (image pixels, converted to latent space for VAE)"}),
            }
        }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode_tiled"
    CATEGORY = "latent/TAESD"
    DESCRIPTION = "Decodes latents from TAESD's latent space back to an image using tiled processing."

    def decode_tiled(self, samples, taesd_model_name, tile_size, overlap): 
        vae = get_taesd_vae_instance(taesd_model_name)
        
        if tile_size < overlap * 4: 
            overlap = tile_size // 4
        
        compression_factor = vae.spacial_compression_decode() 

        latent_tile_x = max(1, tile_size // compression_factor)
        latent_tile_y = max(1, tile_size // compression_factor) 
        latent_overlap = max(0, overlap // compression_factor) 
        
        pixels = vae.decode_tiled(samples["samples"], tile_x=latent_tile_x, tile_y=latent_tile_y, overlap=latent_overlap)
        return (pixels,)

NODE_CLASS_MAPPINGS = {
    "EncodeTAESD": EncodeTAESD,
    "DecodeTAESD": DecodeTAESD,
    "EncodeTAESDTiled": EncodeTAESDTiled,
    "DecodeTAESDTiled": DecodeTAESDTiled,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EncodeTAESD": "TAESD Encode",
    "DecodeTAESD": "TAESD Decode",
    "EncodeTAESDTiled": "TAESD Encode (Tiled)",
    "DecodeTAESDTiled": "TAESD Decode (Tiled)",
}
