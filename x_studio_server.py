"""X-Studio - Web Interface Server for Google Colab and Chrome Control.

Branding:
X-Studio
Dev - @SILENTXOP
YouTuber - @silentx_nomore
YouTube: https://youtube.com/@silentx_nomore

Provides a clean, lightweight Chrome Web UI powered by Gradio and FastAPI.
Runs heavy generation inside the Colab GPU runtime and streams the MP4
directly to your Windows browser for local download without Google Drive.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

# Ensure project root is on PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from x_studio_engine import (
    SUPPORTED_LOCAL_MODELS,
    detect_gpu_hardware,
    generate_video,
)

# Custom CSS for rich X-Studio dark aesthetic
X_STUDIO_CSS = """
body {
    background-color: #0b0f19;
    color: #e2e8f0;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
}
.gradio-container {
    max-width: 1200px !important;
    margin: 0 auto !important;
}
.x-header {
    background: linear-gradient(135deg, #1e1b4b 0%, #0f172a 100%);
    border: 1px solid #312e81;
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 20px;
    text-align: center;
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);
}
.x-header h1 {
    font-size: 2.4rem;
    font-weight: 800;
    margin: 0 0 8px 0;
    background: linear-gradient(90deg, #818cf8, #c084fc, #38bdf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.5px;
}
.x-header .subtitle {
    font-size: 1.05rem;
    color: #94a3b8;
    margin: 0 0 12px 0;
}
.x-branding-badges {
    display: flex;
    justify-content: center;
    gap: 16px;
    flex-wrap: wrap;
    margin-top: 10px;
}
.x-badge {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 0.9rem;
    color: #e2e8f0;
    display: inline-flex;
    align-items: center;
    gap: 6px;
    text-decoration: none;
    transition: all 0.2s ease;
}
.x-badge:hover {
    background: rgba(255, 255, 255, 0.12);
    border-color: #818cf8;
}
.x-badge.youtube {
    border-color: #ef4444;
    color: #fca5a5;
}
.x-badge.youtube:hover {
    background: rgba(239, 68, 68, 0.15);
}
.hw-banner {
    background: #111827;
    border: 1px solid #374151;
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 16px;
    font-size: 0.9rem;
}
.hw-banner.ok {
    border-left: 4px solid #10b981;
}
.hw-banner.warn {
    border-left: 4px solid #f59e0b;
}
.generate-btn {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: 700 !important;
    font-size: 1.1rem !important;
    padding: 14px 20px !important;
    border-radius: 8px !important;
    box-shadow: 0 4px 14px 0 rgba(79, 70, 229, 0.4) !important;
    transition: all 0.2s ease !important;
}
.generate-btn:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px 0 rgba(79, 70, 229, 0.6) !important;
}
.footer-text {
    text-align: center;
    margin-top: 30px;
    padding: 20px;
    color: #64748b;
    font-size: 0.85rem;
    border-top: 1px solid #1e293b;
}
"""


def build_x_studio_app():
    import gradio as gr

    hw = detect_gpu_hardware()

    model_choices = [
        (f"{meta['name']} ({meta['min_vram_gb']}GB VRAM)", key)
        for key, meta in SUPPORTED_LOCAL_MODELS.items()
    ]
    default_model_key = hw["recommended_model"] or "wan2.1-1.3b"

    with gr.Blocks(title="X-Studio | AI Video Generator", css=X_STUDIO_CSS) as app:
        # Header banner
        gr.HTML(
            """
            <div class="x-header">
                <h1>🎬 X-Studio</h1>
                <p class="subtitle">Cloud-Accelerated Local AI Video Generation for Any Browser</p>
                <div class="x-branding-badges">
                    <span class="x-badge">💻 <b>X-Studio</b></span>
                    <span class="x-badge">👨‍💻 Dev: <b>@SILENTXOP</b></span>
                    <a href="https://youtube.com/@silentx_nomore" target="_blank" class="x-badge youtube">
                        ▶️ YouTuber: <b>@silentx_nomore</b> (Subscribe)
                    </a>
                </div>
            </div>
            """
        )

        # Hardware diagnostics banner
        if hw["has_cuda"]:
            hw_html = f"""
            <div class="hw-banner ok">
                <b>🟢 Colab GPU Active:</b> {hw['gpu_name']} &nbsp;|&nbsp;
                <b>Total VRAM:</b> {hw['vram_total_gb']} GB &nbsp;|&nbsp;
                <b>Recommended Model:</b> {SUPPORTED_LOCAL_MODELS[default_model_key]['name']}
            </div>
            """
        else:
            hw_html = f"""
            <div class="hw-banner warn">
                <b>⚠️ Hardware Warning:</b> {hw['error']}<br>
                <i>Connect to a GPU runtime in Colab for full hardware-accelerated video synthesis.</i>
            </div>
            """
        gr.HTML(hw_html)

        with gr.Row():
            # Left Column: Configuration & Controls
            with gr.Column(scale=5):
                gr.Markdown("### 1. Scene Description & Model")
                default_prompt = (
                    "A cinematic drone shot flying over a majestic medieval castle on a misty green mountain "
                    "at sunrise, golden sunlight rays, volumetric clouds, photorealistic, 4k movie trailer style"
                )

                prompt_input = gr.Textbox(
                    label="Prompt",
                    value=default_prompt,
                    placeholder="Enter your scene prompt...",
                    lines=3,
                    max_lines=6,
                )

                with gr.Row():
                    model_selector = gr.Dropdown(
                        label="Video Model",
                        choices=model_choices,
                        value="cogvideo-2b",
                        info="CogVideoX (2B) — Ultra-lightweight (~4.5GB) & generates in ~45 seconds on T4 GPU.",
                    )
                    resolution_selector = gr.Dropdown(
                        label="Resolution",
                        choices=["720x480", "832x480", "768x512", "1280x704", "480x480"],
                        value="720x480",
                    )

                with gr.Row():
                    duration_slider = gr.Slider(
                        label="Duration (seconds)",
                        minimum=1.0,
                        maximum=10.0,
                        step=0.5,
                        value=3.0,
                        info="Higher duration takes more steps/frames.",
                    )
                    steps_slider = gr.Slider(
                        label="Inference Steps",
                        minimum=10,
                        maximum=50,
                        step=5,
                        value=25,
                        info="25-30 steps recommended for optimal speed.",
                    )

                with gr.Accordion("Advanced Options (Image-to-Video, Seed, Negative Prompt)", open=False):
                    ref_image = gr.Image(
                        label="Reference Image (Optional for Image-to-Video)",
                        type="filepath",
                    )
                    negative_prompt = gr.Textbox(
                        label="Negative Prompt",
                        placeholder="worst quality, blurry, distorted, watermark...",
                        lines=2,
                    )
                    seed_input = gr.Number(
                        label="Random Seed (-1 for random)",
                        value=-1,
                        precision=0,
                    )

                generate_btn = gr.Button(
                    "🚀 Generate Video (Colab GPU)",
                    variant="primary",
                    elem_classes=["generate-btn"],
                )

            # Right Column: Output, Preview & Direct Download
            with gr.Column(scale=5):
                gr.Markdown("### 2. Output & Direct Download")
                status_box = gr.Markdown("Ready to generate. Enter prompt and click Generate Video.")

                video_preview = gr.Video(
                    label="Video Preview (Plays directly in Chrome)",
                    interactive=False,
                    autoplay=True,
                )

                download_file = gr.File(
                    label="📥 Direct MP4 Download (Save to Local Windows PC)",
                    interactive=False,
                )

                meta_box = gr.JSON(label="Generation Details", visible=False)

        # Dynamic model update handler: adjust resolution choices & default info
        def on_model_change(selected_key: str):
            if selected_key in SUPPORTED_LOCAL_MODELS:
                meta = SUPPORTED_LOCAL_MODELS[selected_key]
                return (
                    gr.update(choices=meta["resolutions"], value=meta["resolutions"][0]),
                    gr.update(value=meta["default_duration_sec"]),
                    gr.update(value=meta["default_steps"]),
                )
            return gr.update(), gr.update(), gr.update()

        model_selector.change(
            fn=on_model_change,
            inputs=[model_selector],
            outputs=[resolution_selector, duration_slider, steps_slider],
        )

        # Video Generation Action
        def handle_generation(
            prompt,
            model_key,
            resolution,
            duration,
            steps,
            image_path,
            neg_prompt,
            seed_val,
            progress=gr.Progress(track_tqdm=True),
        ):
            if not prompt or not prompt.strip():
                return (
                    None,
                    None,
                    "⚠️ **Error:** Prompt cannot be empty. Please enter a prompt.",
                    gr.update(visible=False),
                )

            progress(0.05, desc="Initializing X-Studio runtime on Colab GPU...")

            def ui_callback(pct, msg):
                progress(pct, desc=msg)

            seed_arg = int(seed_val) if seed_val is not None and seed_val >= 0 else None

            res = generate_video(
                prompt=prompt,
                model_key=model_key,
                resolution=resolution,
                duration_seconds=float(duration),
                steps=int(steps),
                seed=seed_arg,
                negative_prompt=neg_prompt,
                reference_image_path=image_path,
                progress_callback=ui_callback,
            )

            if not res["success"]:
                err_msg = f"❌ **Generation Failed:**\n\n{res['error']}"
                return None, None, err_msg, gr.update(visible=False)

            video_path = res["output_path"]
            success_msg = (
                f"✅ **Success!** Video generated in **{res['elapsed_seconds']}s**\n\n"
                f"- **Model:** `{res['model_name']}`\n"
                f"- **Resolution:** `{res['resolution']}` @ `{res['fps']} FPS`\n"
                f"- **File Size:** `{res['file_size_mb']} MB`\n\n"
                f"Click the download button below to save the MP4 directly to your PC."
            )

            return (
                video_path,
                video_path,
                success_msg,
                gr.update(value=res, visible=True),
            )

        generate_btn.click(
            fn=handle_generation,
            inputs=[
                prompt_input,
                model_selector,
                resolution_selector,
                duration_slider,
                steps_slider,
                ref_image,
                negative_prompt,
                seed_input,
            ],
            outputs=[video_preview, download_file, status_box, meta_box],
        )

        # Footer
        gr.HTML(
            """
            <div class="footer-text">
                <p><b>X-Studio</b> • Developed by <b>@SILENTXOP</b> • YouTuber: <a href="https://youtube.com/@silentx_nomore" target="_blank" style="color: #ef4444; font-weight: 600;">@silentx_nomore</a></p>
                <p>Local GPU AI Video Synthesis • OpenMontage Engine Architecture • Direct Local Download</p>
            </div>
            """
        )

    return app


def start_server(
    port: int = 7860,
    share: bool = True,
    tunnel: str | None = None,
    host: str = "0.0.0.0",
):
    """Launch the X-Studio Web UI server."""
    app = build_x_studio_app()

    print("=" * 60)
    print("🎬 Starting X-Studio Video Generation Server")
    print("Dev: @SILENTXOP | YouTuber: @silentx_nomore")
    print("YouTube: https://youtube.com/@silentx_nomore")
    print("=" * 60)

    # Launch with public share link enabled by default for Colab -> Chrome access
    app.queue().launch(
        server_name=host,
        server_port=port,
        share=share,
        show_error=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start X-Studio Web UI")
    parser.add_argument("--port", type=int, default=7860, help="Server port (default: 7860)")
    parser.add_argument("--share", action="store_true", default=True, help="Create public Gradio tunnel for Chrome access")
    parser.add_argument("--no-share", dest="share", action="store_false", help="Disable public tunnel (localhost only)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host interface (default: 0.0.0.0)")

    args = parser.parse_args()
    start_server(port=args.port, share=args.share, host=args.host)
