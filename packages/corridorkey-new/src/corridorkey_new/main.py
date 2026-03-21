from pathlib import Path

from corridorkey_new import (
    detect_gpu,
    load,
    load_config,
    resolve_alpha,
    resolve_device,
    scan,
    setup_logging,
)
from corridorkey_new.inference import load_model
from corridorkey_new.infra.model_hub import ensure_model
from corridorkey_new.pipeline import PipelineConfig, PipelineRunner

CLIPS_DIR = Path(r"C:\Users\Rajes\Downloads\Samples\sample_inputs_mod")


def _generate_alpha_externally(manifest) -> Path:
    """Stub: simulate external alpha generation."""
    print(f"  Alpha required for '{manifest.clip_name}'.")
    print(f"  Run your alpha generator on: {manifest.frames_dir}")
    raw = input("  Enter path to generated alpha frames directory: ").strip()
    return Path(raw)


def main() -> None:
    config = load_config()
    setup_logging(config)

    gpu = detect_gpu()
    print(gpu.model_dump_json(indent=2))

    clips = scan(CLIPS_DIR)
    manifest = load(clips[0])
    print(manifest.model_dump_json(indent=2))

    if manifest.needs_alpha:
        alpha_dir = _generate_alpha_externally(manifest)
        manifest = resolve_alpha(manifest, alpha_dir)
        print(f"alpha resolved: {manifest.alpha_frames_dir}")

    device = resolve_device(config.device)
    inference_config = config.to_inference_config(device=device)

    # Download the model if it's not already on disk.
    ensure_model(dest_dir=inference_config.checkpoint_path.parent)

    print(f"\nLoading model from {inference_config.checkpoint_path} ...")
    model = load_model(inference_config)
    print("Model loaded.")

    pipeline_config = PipelineConfig(
        preprocess=config.to_preprocess_config(device=device),
        inference=inference_config,
        model=model,
    )

    print(f"\nRunning pipeline for '{manifest.clip_name}' ({manifest.frame_count} frames)...")
    PipelineRunner(manifest, pipeline_config).run()
    print(f"Done. Output written to: {manifest.output_dir}")
