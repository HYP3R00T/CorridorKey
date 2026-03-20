from corridorkey_new.entrypoint import scan

CLIPS_DIR = r"C:\Users\Rajes\Downloads\Samples"


def main() -> None:
    clips = scan(CLIPS_DIR)
    print(f"Found {len(clips)} clip(s)")
    for c in clips:
        print(c)
