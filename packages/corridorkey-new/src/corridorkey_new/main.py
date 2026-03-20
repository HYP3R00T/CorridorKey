from pathlib import Path

from corridorkey_new import (
    FrameReadError,
    PreprocessConfig,
    detect_gpu,
    load,
    load_config,
    preprocess_frame,
    resolve_alpha,
    resolve_device,
    scan,
    setup_logging,
)
from corridorkey_new.loader.validator import get_frame_files

CLIPS_DIR = Path(r"C:\Users\Rajes\Downloads\Samples\sample_inputs_mod")


def _generate_alpha_externally(manifest) -> Path:
    """Stub: simulate external alpha generation.

    In a real CLI/GUI this would invoke the alpha generator tool, wait for it
    to finish, and return the path to the generated alpha frames directory.

    For now, prompt the user to provide the path manually.
    """
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

    # Build file lists once — reused across all frames.
    image_files = get_frame_files(manifest.frames_dir)
    alpha_files = get_frame_files(manifest.alpha_frames_dir)

    device = resolve_device(config.device)
    preprocess_config = PreprocessConfig(img_size=2048, device=device)

    print(f"\nPreprocessing {manifest.frame_count} frame(s) for '{manifest.clip_name}'...")

    try:
        for i in range(*manifest.frame_range):
            result = preprocess_frame(
                manifest,
                i,
                preprocess_config,
                image_files=image_files,
                alpha_files=alpha_files,
            )
            print(
                f"  frame {i:04d} — tensor {tuple(result.tensor.shape)}"
                f" on {result.tensor.device}"
                f" | original ({result.meta.original_h}x{result.meta.original_w})"
            )
    except FrameReadError as e:
        print(f"  ERROR reading frame: {e}")

    print("Done.")
