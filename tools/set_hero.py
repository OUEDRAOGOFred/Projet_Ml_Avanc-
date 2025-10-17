"""
Set the hero image for the Streamlit app by copying a local image file to assets/hero_xray.png
Usage:
    python tools\\set_hero.py "C:\\path\\to\\your\\image.png"
If run without an argument, the script will print usage and look for common image files in the current directory.
"""
import sys
import os
import shutil

def main():
    if len(sys.argv) < 2:
        print("Usage: python tools\\set_hero.py <path-to-image>")
        # list candidates
        cwd = os.getcwd()
        imgs = [f for f in os.listdir(cwd) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if imgs:
            print('\nFound images in current directory:')
            for i, f in enumerate(imgs):
                print(f"  {i+1}. {f}")
            print('\nYou can run: python tools\\set_hero.py "<full-path>"')
        return

    source = sys.argv[1]
    if not os.path.exists(source):
        print(f"File not found: {source}")
        return

    assets_dir = os.path.join(os.getcwd(), 'assets')
    os.makedirs(assets_dir, exist_ok=True)
    dest = os.path.join(assets_dir, 'hero_xray.png')
    try:
        shutil.copy2(source, dest)
        print(f"Saved hero image to {dest}")
    except Exception as e:
        print(f"Error copying file: {e}")

if __name__ == '__main__':
    main()
