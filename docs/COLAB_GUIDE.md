# 🎬 X-Studio: Complete Google Colab & Chrome Guide

**X-Studio**  
**Dev:** [@SILENTXOP](https://github.com/silentxop)  
**YouTuber:** [@silentx_nomore](https://youtube.com/@silentx_nomore)  
**YouTube Channel:** [https://youtube.com/@silentx_nomore](https://youtube.com/@silentx_nomore)  

---

## 1. What X-Studio Is
**X-Studio** is an open-source, cloud-accelerated AI video production studio designed for creators using low-end or non-GPU Windows PCs. 

Instead of demanding an expensive local NVIDIA GPU, all heavy AI video synthesis (diffusion transformers, latents denoising, and MP4 encoding) is offloaded to a free or Pro **Google Colab GPU runtime**. You control the entire creation workflow seamlessly from **Google Chrome** on your local machine and download finished MP4 videos directly to your browser—**no Google Drive required**.

---

## 2. Google Colab Setup

### Step 1: Open or Upload Notebook
1. Navigate to [Google Colab](https://colab.research.google.com/).
2. Click **File** ➔ **Upload notebook**.
3. Select [`X_Studio_Colab.ipynb`](../X_Studio_Colab.ipynb) from this repository.

### Step 2: Enable GPU Hardware Accelerator (Critical)
1. In Colab's top menu, click **Runtime** ➔ **Change runtime type**.
2. Under **Hardware accelerator**, select **T4 GPU** (available on free accounts) or **L4 / A100** (Colab Pro).
3. Click **Save**.

---

## 3. Starting X-Studio
Run the cells in `X_Studio_Colab.ipynb` sequentially:

1. **Step 1 (Hardware Diagnostic):** Detects your GPU name, total VRAM, and confirms CUDA.
2. **Step 2 (Dependencies):** Automatically pulls the repository and installs `diffusers`, `transformers`, `accelerate`, and `gradio`.
3. **Step 3 (Launch Server):**
   ```python
   from x_studio_server import start_server
   start_server(port=7860, share=True)
   ```

---

## 4. Connecting Chrome
Once Step 3 executes, Gradio creates a secure public tunnel. In the cell output, locate the link:
```text
Running on public URL: https://xxxxxxxxxxxx.gradio.live
```
1. Click the `gradio.live` link or copy-paste it into **Google Chrome** on your Windows PC.
2. The **X-Studio Web UI** opens instantly in Chrome with complete control over the Colab GPU.

---

## 5. Selecting a Model
Choose from genuine, tested local models in the dropdown based on your GPU:

| Model Key | Model Name | Min VRAM | Best For | Default Resolution | Default FPS |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `cogvideo-2b` | **CogVideoX (2B)** | **6 GB** | Ultra-lightweight, fastest testing | `720x480` | 8 |
| `wan2.1-1.3b` | **Wan 2.1 T2V (1.3B)** | **8 GB** | Best quality-to-VRAM ratio (Colab T4 default) | `832x480` | 16 |
| `ltx2-local` | **LTX-2 (Local)** | **12 GB** | Fast 30 FPS fluid motion | `768x512` | 30 |
| `cogvideo-5b` | **CogVideoX 1.5 (5B)** | **12 GB** | Higher fidelity, image-to-video | `720x480` | 8 |
| `wan2.2-ti2v-5b` | **Wan 2.2 TI2V (5B)** | **12 GB** | Cinematic 720p with sequential offload | `1280x704` | 24 |

*X-Studio automatically inspects your available VRAM and highlights the optimal model.*

---

## 6. Entering a Prompt
In the **Prompt** text area, describe your scene in visual, cinematographic detail:
- **Style:** `Cinematic, photorealistic, 4K, octane render, unreal engine 5`
- **Lighting:** `Golden hour sunset, moody volumetric lighting, neon reflections`
- **Camera Movement:** `Slow drone dolly forward, smooth tracking shot, cinematic pan`

*Example:*
> *"A majestic golden eagle soaring gracefully over snow-capped mountain peaks during golden hour, cinematic lighting, ultra-detailed feathers, 4K documentary style."*

---

## 7. Generating a Video
1. Select your target **Resolution** (e.g. `832x480` or `720x480`).
2. Set **Duration** (start with `2.0s` or `3.0s` for quick generation).
3. Set **Inference Steps** (`25` to `30` steps provides an ideal speed-to-quality balance).
4. *(Optional)* Expand **Advanced Options** to provide a reference image or negative prompt.
5. Click **🚀 Generate Video (Colab GPU)**.

---

## 8. Monitoring Progress
- The Web UI displays real-time progress indicators:
  - `Loading model weights into GPU...`
  - `Denoising video latent frames...`
  - `Encoding MP4 container...`
- You can also view raw step logs inside the Google Colab terminal cell.

---

## 9. Downloading MP4 to Local PC (No Google Drive)
When generation finishes:
1. The video immediately loads in the **Video Preview** player in Chrome for instant playback.
2. Directly below the preview, click the **📥 Direct MP4 Download** button.
3. The video downloads directly through your Chrome browser into your Windows `Downloads` folder.
4. **Google Drive is never needed or touched.**

---

## 10. GPU / VRAM Requirements

- **Free Google Colab (Tesla T4 - ~15 GB VRAM):**
  - Fully supports `wan2.1-1.3b` and `cogvideo-2b`.
  - Supports `wan2.2-ti2v-5b` and `ltx2-local` with sequential offload.
- **Colab Pro / Compute Units (NVIDIA L4 - 24 GB / A100 - 40 GB):**
  - Supports all models at full precision and higher resolutions.

---

## 11. Common Errors & Fixes

| Error | Cause | Solution |
| :--- | :--- | :--- |
| `No NVIDIA GPU detected` | Colab is running in CPU mode | Go to `Runtime` ➔ `Change runtime type` ➔ Select `T4 GPU`. |
| `CUDA out of memory` | Model exceeded available VRAM | Switch to `cogvideo-2b` or lower resolution (`720x480`), or reduce duration to `2.0s`. |
| `Public URL failed to start` | Gradio tunnel timeout | Re-run the server cell. |
| `ModuleNotFoundError: diffusers` | Dependencies not installed | Ensure Step 2 in the notebook completed successfully. |

---

## 12. Colab Session Limitations & Tips
- **Session Timeout:** Free Colab sessions disconnect if left idle for ~30 minutes or after ~12 hours total. Keep the Colab browser tab open while generating.
- **Resource Reallocation:** If your GPU is reclaimed by Google, simply restart the session via `Runtime` ➔ `Disconnect and delete runtime`, then reconnect.

---

## 13. Stopping & Resetting the Runtime
- To stop the server: Click the **Stop (■)** button on the running Colab code cell.
- To clear VRAM without restarting:
  ```python
  import torch, gc
  torch.cuda.empty_cache()
  gc.collect()
  ```
- To reset everything: In Colab menu, select **Runtime** ➔ **Restart session**.

---

<div align="center">
  <sub><b>X-Studio</b> • Developed by <b>@SILENTXOP</b> • YouTuber: <a href="https://youtube.com/@silentx_nomore"><b>@silentx_nomore</b></a></sub>
</div>
