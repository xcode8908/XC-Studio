"""Generate X_Studio_Colab.ipynb notebook."""

import json
from pathlib import Path

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 🎬 X-Studio: Colab-Accelerated AI Video Generator\n",
                "\n",
                "**Developer:** [@SILENTXOP](https://github.com/silentxop)  \n",
                "**YouTuber:** [@silentx_nomore](https://youtube.com/@silentx_nomore)  \n",
                "**YouTube Channel:** [https://youtube.com/@silentx_nomore](https://youtube.com/@silentx_nomore)  \n",
                "\n",
                "---\n",
                "\n",
                "### 📌 Target Architecture\n",
                "Windows Low-End PC ➔ Chrome Browser ➔ X-Studio Web UI ➔ Google Colab GPU ➔ Local Model Synthesis ➔ Direct MP4 Download to Windows PC.\n",
                "\n",
                "> **Important:** \n",
                "> 1. Make sure you are using a GPU runtime: **Runtime** -> **Change runtime type** -> **T4 GPU** (or L4 / A100).\n",
                "> 2. No Google Drive required — video files are saved in Colab temporary storage and downloaded directly via browser.\n",
                "> 3. Low-end PC friendly — zero local GPU required on your Windows computer."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 🔍 Step 1: Verify NVIDIA GPU & CUDA Environment\n",
                "Run this cell to detect your GPU model, available VRAM, and verify CUDA."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import torch\n",
                "import sys\n",
                "\n",
                "print('=' * 60)\n",
                "print('🔍 X-Studio Hardware Diagnostic')\n",
                "print('=' * 60)\n",
                "\n",
                "if not torch.cuda.is_available():\n",
                "    raise SystemExit(\n",
                "        '❌ CRITICAL: No GPU detected!\\n'\n",
                "        'Please switch to a GPU runtime in Colab:\\n'\n",
                "        '1. Click Runtime in the top menu.\\n'\n",
                "        '2. Select Change runtime type.\\n'\n",
                "        '3. Under Hardware accelerator, select T4 GPU (free) or L4/A100.\\n'\n",
                "        '4. Click Save and re-run this cell.'\n",
                "    )\n",
                "\n",
                "gpu_name = torch.cuda.get_device_name(0)\n",
                "total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)\n",
                "\n",
                "print(f'✅ GPU Detected:     {gpu_name}')\n",
                "print(f'✅ Total VRAM:       {total_vram_gb:.2f} GB')\n",
                "print(f'✅ PyTorch Version:  {torch.__version__}')\n",
                "print(f'✅ CUDA Version:     {torch.version.cuda}')\n",
                "\n",
                "if total_vram_gb >= 14.0:\n",
                "    print('\\n🚀 Recommended Models: Wan 2.1 (1.3B) [Fast, High Quality], CogVideoX (2B), LTX-2, Wan 2.2 TI2V (5B)')\n",
                "elif total_vram_gb >= 7.0:\n",
                "    print('\\n🚀 Recommended Models: Wan 2.1 (1.3B), CogVideoX (2B)')\n",
                "else:\n",
                "    print('\\n🚀 Recommended Models: CogVideoX (2B) [Ultra-low VRAM]')\n",
                "print('=' * 60)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 📦 Step 2: Install X-Studio Dependencies\n",
                "Clone or sync the repository and install required packages (`diffusers`, `transformers`, `accelerate`, `gradio`, etc.)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os, sys\n",
                "\n",
                "# Clone repo if not already present\n",
                "if not os.path.exists('XC-Studio') and not os.path.exists('x_studio_engine.py'):\n",
                "    print('📥 Cloning X-Studio repository...')\n",
                "    !git clone https://github.com/xcode8908/XC-Studio.git\n",
                "    %cd XC-Studio\n",
                "elif os.path.exists('XC-Studio'):\n",
                "    %cd XC-Studio\n",
                "    !git pull\n",
                "\n",
                "print('\\n📦 Installing required Python dependencies...')\n",
                "!pip install -q diffusers transformers accelerate sentencepiece gradio imageio imageio-ffmpeg pydantic pyyaml requests\n",
                "\n",
                "print('\\n✅ Environment ready for X-Studio generation!')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 🚀 Step 3: Launch X-Studio Web UI\n",
                "Run this cell to start the X-Studio server.  \n",
                "Look for the **`Running on public URL: https://xxxx.gradio.live`** in the output and click it to control X-Studio from **Google Chrome** on your Windows PC!"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from x_studio_server import start_server\n",
                "\n",
                "# Starts server and provides a public HTTPS link for your Chrome browser\n",
                "start_server(port=7860, share=True)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 🧪 Optional: Quick Test Generation via Python Script\n",
                "If you prefer to test video generation directly without launching the UI, run this cell."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from x_studio_engine import generate_video\n",
                "\n",
                "test_prompt = 'A cinematic drone shot flying over a majestic medieval castle on a misty green mountain at sunrise, golden sunlight rays, volumetric clouds, photorealistic, 4k movie trailer style'\n",
                "print(f\"🎬 Generating Wan 2.1 video: '{test_prompt}'...\")\n",
                "\n",
                "result = generate_video(\n",
                "    prompt=test_prompt,\n",
                "    model_key='wan2.1-1.3b',  # Best quality for Colab T4 GPU\n",
                "    resolution='832x480',\n",
                "    duration_seconds=3.0,\n",
                "    steps=25,\n",
                "    seed=42,\n",
                ")\n",
                "\n",
                "if result['success']:\n",
                "    print(f'\\n✅ Test video generated successfully!')\n",
                "    print(f\"Output path: {result['output_path']}\")\n",
                "    print(f\"Duration:    {result['duration']}s\")\n",
                "    print(f\"File Size:   {result['file_size_mb']} MB\")\n",
                "else:\n",
                "    print(f\"\\n❌ Generation failed: {result['error']}\")"
            ]
        }
    ],
    "metadata": {
        "accelerator": "GPU",
        "colab": {
            "provenance": []
        },
        "kernelspec": {
            "display_name": "Python 3",
            "name": "python3"
        },
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 0
}

out_path = Path("X_Studio_Colab.ipynb")
with out_path.open("w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print("Created X_Studio_Colab.ipynb successfully!")
