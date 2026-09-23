<h1 align="center">🎬 X-Studio</h1>

<p align="center"><strong>Cloud-Accelerated AI Video Production Studio Powered by Google Colab & Controlled via Chrome</strong></p>

<div align="center">

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xcode8908/XC-Studio/blob/main/X_Studio_Colab.ipynb)
[![YouTube Channel](https://img.shields.io/badge/YouTube-%40silentx__nomore-FF0000?style=for-the-badge&logo=youtube&logoColor=white)](https://youtube.com/@silentx_nomore)
[![Developer](https://img.shields.io/badge/Developer-%40SILENTXOP-111111?style=for-the-badge&logo=github&logoColor=white)](https://github.com/silentxop)
[![License](https://img.shields.io/badge/License-AGPLv3-blue.svg?style=for-the-badge)](LICENSE)

---

### **Dev:** [@SILENTXOP](https://github.com/silentxop) &nbsp;•&nbsp; **YouTuber:** [@silentx_nomore](https://youtube.com/@silentx_nomore)  
🎥 **Official YouTube Channel:** [https://youtube.com/@silentx_nomore](https://youtube.com/@silentx_nomore)

</div>

---

## ⚡ 1-Click Launch on Google Colab

Generate cinematic AI videos for free using Google Colab's cloud GPU right from your Google Chrome browser on any low-end PC:

👉 **[Launch X-Studio in Google Colab](https://colab.research.google.com/github/xcode8908/XC-Studio/blob/main/X_Studio_Colab.ipynb)**

---

## 🎯 Architecture: Low-End PC Friendly

No local graphics card or heavy machine learning setup required on your Windows computer:

```text
Low-End Windows PC
       ↓
Google Chrome Browser
       ↓
X-Studio Web UI (Interactive Gradio Interface)
       ↓
Google Colab Runtime (Free Cloud T4 / L4 / A100 GPU)
       ↓
Local AI Video Synthesis (Wan 2.1 / CogVideoX / LTX-Video)
       ↓
Final Rendered MP4 Video
       ↓
Direct Browser Download to Local PC (Zero Google Drive Required!)
```

---

## 🚀 3-Step Quick Start Guide

### Step 1: Open in Colab & Enable GPU
1. Click the **[Open In Colab](https://colab.research.google.com/github/xcode8908/XC-Studio/blob/main/X_Studio_Colab.ipynb)** badge.
2. In Colab's menu, click **Runtime** ➔ **Change runtime type**.
3. Under **Hardware accelerator**, select **T4 GPU** (free) and click **Save**.

### Step 2: Run All Cells
* Press `Ctrl + F9` or click **Runtime** ➔ **Run all**.
* The notebook will automatically set up PyTorch, install Diffusers, configure the video engines, and launch the Web UI server.

### Step 3: Open Web UI & Generate
* In the output of Step 4, click the public Gradio live link:
  ```text
  Running on public URL: https://xxxxxxxx.gradio.live
  ```
* The Web UI opens in your Chrome browser with **Wan 2.1 (1.3B)** and a sample cinematic prompt pre-filled!
* Click **🚀 Generate Video** and then **⬇️ Download Video (MP4)** to save directly to your local computer.

---

## 🧠 Supported Local Video Models (Zero Paid API Keys)

Every model runs **100% inside Google Colab's free GPU**:

| Model | Variant | Min VRAM | Best For | Default Resolution | Default FPS |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Wan 2.1 T2V (1.3B)** *(Default)* | `wan2.1-1.3b` | **8 GB** | Highest visual fidelity on Colab T4 | `832x480` | 16 FPS |
| **CogVideoX (2B)** | `cogvideo-2b` | **6 GB** | Ultra-lightweight & fastest generation | `720x480` | 8 FPS |
| **LTX-2 (Local)** | `ltx2-local` | **12 GB** | Real-time transformer synthesis | `768x512` | 30 FPS |
| **CogVideoX 1.5 (5B)** | `cogvideo-5b` | **12 GB** | High-motion fidelity & Image-to-Video | `720x480` | 8 FPS |
| **Wan 2.2 TI2V (5B)** | `wan2.2-ti2v-5b` | **12 GB** | Cinematic 720p with sequential offload | `1280x704` | 24 FPS |

---

## ✨ Key Features

* **Zero Local GPU Needed:** Works on any low-end Windows laptop or desktop.
* **100% Free & Open-Source:** No paid API keys, credit cards, or token subscriptions.
* **Direct Browser Download:** Final MP4 videos download directly via Chrome — no cluttered Google Drive storage.
* **Pre-configured Defaults:** Pre-loaded with Wan 2.1 (1.3B), optimal inference steps, and cinematic prompts.
* **Full In-Browser Video Player:** Instant HTML5 video preview right inside Chrome.

---

## 👤 Credits & Socials

* **Developer:** [@SILENTXOP](https://github.com/silentxop)
* **YouTuber:** [@silentx_nomore](https://youtube.com/@silentx_nomore)
* **Subscribe on YouTube:** [https://youtube.com/@silentx_nomore](https://youtube.com/@silentx_nomore)
* **Repository:** [https://github.com/xcode8908/XC-Studio](https://github.com/xcode8908/XC-Studio)
