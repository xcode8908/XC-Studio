"""X-Studio - Core Video Generation Engine & Hardware Auto-Configuration.

Developed for X-Studio
Dev: @SILENTXOP
YouTuber: @silentx_nomore
YouTube: https://youtube.com/@silentx_nomore

This engine interfaces directly with OpenMontage local video generation models
(Wan 2.x, CogVideoX, LTX-Video, HunyuanVideo), provides GPU and VRAM discovery,
automatic model recommendation, and zero-cloud local rendering inside Google Colab.
"""

from __future__ import annotations

import gc
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Supported genuine local models and specifications
SUPPORTED_LOCAL_MODELS: dict[str, dict[str, Any]] = {
    "wan2.1-1.3b": {
        "name": "Wan 2.1 T2V (1.3B)",
        "provider": "wan",
        "tool_name": "wan_video",
        "min_vram_gb": 8.0,
        "recommended_vram_gb": 12.0,
        "hf_id": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        "description": "High visual quality with rapid diffusion speed. Fits standard Google Colab T4 GPUs.",
        "operations": ["text_to_video", "video_to_video", "text_to_image"],
        "supports_image_input": False,
        "default_width": 832,
        "default_height": 480,
        "default_steps": 30,
        "fps": 16,
        "resolutions": ["832x480", "704x480", "480x480"],
        "default_duration_sec": 3.0,
    },
    "cogvideo-2b": {
        "name": "CogVideoX (2B)",
        "provider": "cogvideo",
        "tool_name": "cogvideo_video",
        "min_vram_gb": 6.0,
        "recommended_vram_gb": 8.0,
        "hf_id": "THUDM/CogVideoX-2b",
        "description": "Ultra-lightweight text-to-video model. Fastest generation and lowest memory footprint.",
        "operations": ["text_to_video"],
        "supports_image_input": False,
        "default_width": 720,
        "default_height": 480,
        "default_steps": 25,
        "fps": 8,
        "resolutions": ["720x480", "480x480"],
        "default_duration_sec": 4.0,
    },
    "ltx2-local": {
        "name": "LTX-2 (Local)",
        "provider": "ltx",
        "tool_name": "ltx_video_local",
        "min_vram_gb": 12.0,
        "recommended_vram_gb": 16.0,
        "hf_id": "Lightricks/LTX-2",
        "description": "Real-time transformer-based architecture with fluid 30 FPS motion synthesis.",
        "operations": ["text_to_video", "image_to_video"],
        "supports_image_input": True,
        "default_width": 768,
        "default_height": 512,
        "default_steps": 25,
        "fps": 30,
        "resolutions": ["768x512", "512x512"],
        "default_duration_sec": 3.0,
    },
    "cogvideo-5b": {
        "name": "CogVideoX 1.5 (5B)",
        "provider": "cogvideo",
        "tool_name": "cogvideo_video",
        "min_vram_gb": 12.0,
        "recommended_vram_gb": 16.0,
        "hf_id": "THUDM/CogVideoX-5b",
        "description": "High-fidelity generation with balanced motion dynamics. Supports image-to-video.",
        "operations": ["text_to_video", "image_to_video"],
        "supports_image_input": True,
        "default_width": 720,
        "default_height": 480,
        "default_steps": 30,
        "fps": 8,
        "resolutions": ["720x480", "480x480"],
        "default_duration_sec": 4.0,
    },
    "wan2.2-ti2v-5b": {
        "name": "Wan 2.2 TI2V (5B)",
        "provider": "wan",
        "tool_name": "wan_video",
        "min_vram_gb": 12.0,
        "recommended_vram_gb": 16.0,
        "hf_id": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        "description": "Cinematic 720p generation with high coherence. Supports text and image inputs with offloading.",
        "operations": ["text_to_video", "image_to_video", "video_to_video", "first_last_frame"],
        "supports_image_input": True,
        "default_width": 1280,
        "default_height": 704,
        "default_steps": 40,
        "fps": 24,
        "resolutions": ["1280x704", "704x480"],
        "default_duration_sec": 3.0,
    },
}


def detect_gpu_hardware() -> dict[str, Any]:
    """Inspect system GPU, CUDA runtime, and available VRAM.

    Returns detailed hardware statistics and adaptive recommendations.
    """
    try:
        import torch
    except ImportError:
        return {
            "has_cuda": False,
            "error": "PyTorch is not installed. Run 'pip install torch torchvision'.",
            "gpu_name": "None",
            "vram_total_gb": 0.0,
            "vram_free_gb": 0.0,
            "recommended_model": None,
            "available_models": [],
        }

    if not torch.cuda.is_available():
        return {
            "has_cuda": False,
            "error": (
                "No NVIDIA GPU detected in current runtime!\n"
                "In Google Colab, go to: Runtime -> Change runtime type -> Hardware accelerator -> "
                "Select 'T4 GPU' (free) or 'L4'/'A100' (Colab Pro) and reconnect."
            ),
            "gpu_name": "CPU Only",
            "vram_total_gb": 0.0,
            "vram_free_gb": 0.0,
            "recommended_model": None,
            "available_models": [],
        }

    device_id = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(device_id)
    props = torch.cuda.get_device_properties(device_id)
    total_vram_gb = round(props.total_memory / (1024**3), 2)

    # Free memory estimate
    try:
        torch.cuda.empty_cache()
        allocated_bytes = torch.cuda.memory_allocated(device_id)
        reserved_bytes = torch.cuda.memory_reserved(device_id)
        free_vram_gb = round((props.total_memory - max(allocated_bytes, reserved_bytes)) / (1024**3), 2)
    except Exception:
        free_vram_gb = total_vram_gb

    # Determine eligible and recommended models based on physical VRAM
    available_models: list[str] = []
    for model_key, meta in SUPPORTED_LOCAL_MODELS.items():
        if total_vram_gb >= (meta["min_vram_gb"] - 0.5):
            available_models.append(model_key)

    # If VRAM is tight, prioritize lightweight models
    if total_vram_gb >= 14.0:
        # Colab T4 (15-16GB) or L4 (24GB) or A100 (40GB)
        recommended_model = "wan2.1-1.3b"
    elif total_vram_gb >= 8.0:
        recommended_model = "wan2.1-1.3b" if "wan2.1-1.3b" in available_models else "cogvideo-2b"
    elif total_vram_gb >= 5.5:
        recommended_model = "cogvideo-2b"
    else:
        recommended_model = available_models[0] if available_models else None

    return {
        "has_cuda": True,
        "error": None,
        "device_id": device_id,
        "gpu_name": gpu_name,
        "cuda_version": torch.version.cuda,
        "vram_total_gb": total_vram_gb,
        "vram_free_gb": free_vram_gb,
        "available_models": available_models if available_models else ["cogvideo-2b"],
        "recommended_model": recommended_model or "cogvideo-2b",
    }


def parse_resolution(res_str: str) -> tuple[int, int]:
    """Parse 'WIDTHxHEIGHT' string into integer tuple."""
    try:
        parts = res_str.lower().split("x")
        return int(parts[0]), int(parts[1])
    except Exception:
        return 720, 480


def generate_video(
    prompt: str,
    model_key: str = "wan2.1-1.3b",
    resolution: str = "832x480",
    duration_seconds: float = 3.0,
    steps: int | None = None,
    seed: int | None = None,
    negative_prompt: str | None = None,
    reference_image_path: str | None = None,
    progress_callback: Callable[[float, str], None] | None = None,
) -> dict[str, Any]:
    """Execute local video generation inside Colab GPU using OpenMontage engines.

    Parameters:
        prompt: Text description of the scene to generate.
        model_key: Key of the supported model (e.g. 'wan2.1-1.3b', 'cogvideo-2b').
        resolution: 'WIDTHxHEIGHT' string.
        duration_seconds: Output video length in seconds.
        steps: Number of diffusion inference steps.
        seed: Random seed for reproducibility (-1 or None for random).
        negative_prompt: Elements to avoid in the generation.
        reference_image_path: Path to reference image for image-to-video mode.
        progress_callback: Optional callable(fraction, message) for UI updates.

    Returns:
        dict with success status, output_path, duration, and metadata.
    """
    if not prompt or not prompt.strip():
        return {
            "success": False,
            "error": "Prompt cannot be empty. Please enter a descriptive scene prompt.",
            "output_path": None,
        }

    # Verify hardware
    hw = detect_gpu_hardware()
    if not hw["has_cuda"]:
        return {
            "success": False,
            "error": hw["error"] or "CUDA GPU is unavailable.",
            "output_path": None,
        }

    if model_key not in SUPPORTED_LOCAL_MODELS:
        model_key = hw["recommended_model"] or "wan2.1-1.3b"

    meta = SUPPORTED_LOCAL_MODELS[model_key]

    # Verify VRAM safety
    if hw["vram_total_gb"] < (meta["min_vram_gb"] - 1.0):
        return {
            "success": False,
            "error": (
                f"Insufficient GPU VRAM: {meta['name']} requires at least {meta['min_vram_gb']} GB VRAM, "
                f"but only {hw['vram_total_gb']} GB was detected.\n"
                f"Please switch to a lighter model like 'CogVideoX (2B)' or 'Wan 2.1 (1.3B)'."
            ),
            "output_path": None,
        }

    width, height = parse_resolution(resolution)
    if steps is None or steps <= 0:
        steps = meta["default_steps"]

    # Clamp steps for Colab stability
    steps = max(10, min(60, steps))

    # Calculate frames from duration and fps
    fps = meta["fps"]
    num_frames = max(8, round(duration_seconds * fps))

    # Output directory (temporary Colab workspace, NEVER Google Drive)
    output_dir = PROJECT_ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = int(time.time())
    output_file = output_dir / f"x_studio_{model_key}_{timestamp}.mp4"

    # Operation mode
    is_i2v = bool(reference_image_path and Path(reference_image_path).exists() and meta["supports_image_input"])
    operation = "image_to_video" if is_i2v else "text_to_video"

    if progress_callback:
        progress_callback(0.1, f"Loading model {meta['name']} into GPU memory...")

    # Enable local generation status flag expected by OpenMontage BaseTool
    os.environ["VIDEO_GEN_LOCAL_ENABLED"] = "true"
    os.environ["VIDEO_GEN_LOCAL_MODEL"] = model_key

    start_time = time.time()

    try:
        if meta["provider"] == "wan":
            from tools.video.wan_video import WanVideo
            tool = WanVideo()
            inputs: dict[str, Any] = {
                "prompt": prompt.strip(),
                "model_variant": model_key,
                "operation": operation,
                "width": width,
                "height": height,
                "fps": fps,
                "duration_seconds": duration_seconds,
                "num_frames": num_frames,
                "num_inference_steps": steps,
                "enable_model_offload": True,
                "output_path": str(output_file),
            }
            if seed is not None and seed >= 0:
                inputs["seed"] = seed
            if negative_prompt and negative_prompt.strip():
                inputs["negative_prompt"] = negative_prompt.strip()
            if is_i2v and reference_image_path:
                inputs["reference_image_path"] = reference_image_path

            if progress_callback:
                progress_callback(0.3, "Denoising video latent frames...")

            res = tool.execute(inputs)
            if not res.success:
                return {
                    "success": False,
                    "error": res.error or "Wan video generation failed.",
                    "output_path": None,
                }

        elif meta["provider"] == "cogvideo":
            from tools.video.cogvideo_video import CogVideoVideo
            tool = CogVideoVideo()
            inputs = {
                "prompt": prompt.strip(),
                "model_variant": model_key,
                "operation": operation,
                "width": width,
                "height": height,
                "num_frames": num_frames,
                "num_inference_steps": steps,
                "enable_model_offload": True,
                "output_path": str(output_file),
            }
            if seed is not None and seed >= 0:
                inputs["seed"] = seed
            if is_i2v and reference_image_path:
                inputs["reference_image_path"] = reference_image_path

            if progress_callback:
                progress_callback(0.3, f"Denoising frames with {meta['name']}...")

            res = tool.execute(inputs)
            if not res.success:
                return {
                    "success": False,
                    "error": res.error or f"{meta['name']} generation failed.",
                    "output_path": None,
                }

        elif meta["provider"] == "ltx":
            from tools.video.ltx_video_local import LTXVideoLocal
            tool = LTXVideoLocal()
            inputs = {
                "prompt": prompt.strip(),
                "model_variant": "ltx2-local",
                "operation": operation,
                "width": width,
                "height": height,
                "num_frames": num_frames,
                "num_inference_steps": steps,
                "enable_model_offload": True,
                "output_path": str(output_file),
            }
            if seed is not None and seed >= 0:
                inputs["seed"] = seed
            if is_i2v and reference_image_path:
                inputs["reference_image_path"] = reference_image_path

            if progress_callback:
                progress_callback(0.3, "Synthesizing frames with LTX-Video...")

            res = tool.execute(inputs)
            if not res.success:
                return {
                    "success": False,
                    "error": res.error or "LTX local generation failed.",
                    "output_path": None,
                }

        else:
            return {
                "success": False,
                "error": f"Unsupported model provider: {meta['provider']}",
                "output_path": None,
            }

        if progress_callback:
            progress_callback(0.95, "Finalizing MP4 container encoding...")

        elapsed = round(time.time() - start_time, 2)

        # Confirm output file existence
        if not output_file.exists() or output_file.stat().st_size == 0:
            return {
                "success": False,
                "error": "Video rendering completed, but output MP4 file was not written to disk.",
                "output_path": None,
            }

        file_size_mb = round(output_file.stat().st_size / (1024 * 1024), 2)

        if progress_callback:
            progress_callback(1.0, f"Ready! Generated {file_size_mb} MB video in {elapsed}s.")

        return {
            "success": True,
            "error": None,
            "output_path": str(output_file),
            "filename": output_file.name,
            "elapsed_seconds": elapsed,
            "file_size_mb": file_size_mb,
            "model_name": meta["name"],
            "resolution": f"{width}x{height}",
            "fps": fps,
            "duration": duration_seconds,
            "seed": seed,
        }

    except Exception as exc:
        # Clear CUDA memory cache on failure
        try:
            import torch
            torch.cuda.empty_cache()
            gc.collect()
        except Exception:
            pass
        return {
            "success": False,
            "error": f"Generation exception encountered: {exc}",
            "output_path": None,
        }
