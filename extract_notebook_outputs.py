"""extract_notebook_outputs.py
Reads a .ipynb notebook, extracts base64-encoded images from cell outputs and
text outputs. Saves images to notebook_outputs/images/ and saves a metrics JSON
and a text dump of outputs to notebook_outputs/.

Usage:
    python extract_notebook_outputs.py [path_to_notebook.ipynb]

If no path is provided, defaults to 'Untitled10_(5).ipynb' in the current folder.
"""
import json
import os
import sys
import base64
import re
from pathlib import Path


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def extract_images_and_text(nb_path, out_dir):
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    imgs_dir = os.path.join(out_dir, 'images')
    ensure_dir(imgs_dir)

    text_out = []
    metrics = {}
    img_count = 0
    
    # NEW: Store structured text outputs
    text_outputs = []

    tables_dir = os.path.join(out_dir, 'tables')
    ensure_dir(tables_dir)

    image_map = []
    expected_outputs = []

    for ci, cell in enumerate(nb.get('cells', [])):
        cell_source = ''.join(cell.get('source', [])) if cell.get('source') else ''
        outputs = cell.get('outputs', [])
        for oi, out in enumerate(outputs):
            # collect textual output
            if 'text' in out:
                txt = out['text']
                if isinstance(txt, list):
                    txt = ''.join(txt)
                text_out.append(txt)
                
                # NEW: Store structured text with cell info
                text_outputs.append({
                    'cell_index': ci,
                    'output_index': oi,
                    'text': txt,
                    'cell_source': cell_source.strip()[:200]  # First 200 chars
                })

            data = out.get('data', {}) or {}

            # Extract images
            for mime in ('image/png', 'image/jpeg'):
                if mime in data:
                    b64 = data[mime]
                    if isinstance(b64, list):
                        b64 = ''.join(b64)
                    b64 = re.sub(r'^data:[^,]+,', '', b64)
                    try:
                        raw = base64.b64decode(b64)
                    except Exception:
                        continue
                    ext = 'png' if 'png' in mime else 'jpg'
                    out_name = f'nb_image_c{ci}_o{oi}_{img_count}.{ext}'
                    out_path = os.path.join(imgs_dir, out_name)
                    with open(out_path, 'wb') as imgf:
                        imgf.write(raw)
                    # record mapping
                    image_map.append({
                        'image': os.path.relpath(out_path, start=out_dir),
                        'cell_index': ci,
                        'output_index': oi,
                        'cell_source': cell_source.strip()
                    })
                    img_count += 1

            # Extract HTML tables (if present) and save as CSV
            if 'text/html' in data:
                html = data['text/html']
                if isinstance(html, list):
                    html = ''.join(html)
                try:
                    import pandas as pd
                    tables = pd.read_html(html)
                    for ti, tbl in enumerate(tables):
                        tbl_name = f'table_c{ci}_o{oi}_{ti}.csv'
                        tbl_path = os.path.join(tables_dir, tbl_name)
                        tbl.to_csv(tbl_path, index=False)
                except Exception:
                    # ignore table parse errors
                    pass

            # Scan the cell source code for explicit save calls (plt.savefig, fig.savefig, cv2.imwrite)
            try:
                src = cell_source or ''
                # common patterns: plt.savefig('name.png'), fig.savefig("name.png"), cv2.imwrite('name.png', ...)
                save_patterns = re.findall(r"(?:plt|fig|figure)\.savefig\(\s*['\"]([^'\"]+)['\"]", src)
                save_patterns += re.findall(r"cv2\.imwrite\(\s*['\"]([^'\"]+)['\"]", src)
                # also catch pathlib / open writes to 'notebook_outputs' etc
                save_patterns += re.findall(r"to_csv\(\s*['\"]([^'\"]+\.csv)['\"]", src)
                for ref in save_patterns:
                    ref_path = os.path.normpath(os.path.join(os.path.dirname(nb_path), ref))
                    expected_outputs.append({'cell_index': ci, 'ref': ref, 'resolved_path': ref_path, 'exists': os.path.exists(ref_path)})
                    # If the referenced file already exists, copy into our images/tables dirs
                    if os.path.exists(ref_path):
                        if ref.lower().endswith(('.png', '.jpg', '.jpeg')):
                            target = os.path.join(imgs_dir, os.path.basename(ref))
                            try:
                                with open(ref_path, 'rb') as rf, open(target, 'wb') as wf:
                                    wf.write(rf.read())
                                image_map.append({'image': os.path.relpath(target, start=out_dir), 'cell_index': ci, 'cell_source': cell_source.strip()})
                                img_count += 1
                            except Exception:
                                pass
                        elif ref.lower().endswith('.csv'):
                            target_tables_dir = tables_dir
                            try:
                                import shutil as _sh
                                _sh.copy(ref_path, os.path.join(target_tables_dir, os.path.basename(ref)))
                            except Exception:
                                pass
            except Exception:
                pass

    # Save textual outputs dump
    text_path = os.path.join(out_dir, 'notebook_text_outputs.txt')
    with open(text_path, 'w', encoding='utf-8') as tf:
        tf.write('\n\n'.join(text_out))
    
    # NEW: Save structured text outputs as JSON
    text_outputs_path = os.path.join(out_dir, 'text_outputs.json')
    with open(text_outputs_path, 'w', encoding='utf-8') as tof:
        json.dump(text_outputs, tof, ensure_ascii=False, indent=2)
    print(f'Structured text outputs saved to {text_outputs_path}')

    # Very small metric extraction heuristics: look for lines containing keywords
    keywords = ['accuracy', 'auc', 'precision', 'recall', 'f1', 'best model', 'meilleur', 'accuracy:']
    combined = '\n'.join(text_out)
    for line in combined.splitlines():
        low = line.lower()
        for kw in keywords:
            if kw in low:
                # try to extract a number from the line
                nums = re.findall(r"\d+\.\d+|\d+%|\d+", line)
                metrics.setdefault(line.strip(), nums)
                break

    metrics_path = os.path.join(out_dir, 'notebook_metrics.json')
    with open(metrics_path, 'w', encoding='utf-8') as mf:
        json.dump({'metrics_found': metrics}, mf, ensure_ascii=False, indent=2)

    print(f'Extracted {img_count} images to {imgs_dir}')
    print(f'Text outputs saved to {text_path}')
    print(f'Metrics heuristics saved to {metrics_path}')
    # Save image->cell mapping
    image_map_path = os.path.join(out_dir, 'image_map.json')
    with open(image_map_path, 'w', encoding='utf-8') as imf:
        json.dump(image_map, imf, ensure_ascii=False, indent=2)
    print(f'Image map saved to {image_map_path}')

    # Save expected outputs manifest (references found in code cells)
    expected_path = os.path.join(out_dir, 'expected_outputs.json')
    try:
        with open(expected_path, 'w', encoding='utf-8') as ef:
            json.dump({'expected': expected_outputs}, ef, ensure_ascii=False, indent=2)
        print(f'Expected outputs manifest saved to {expected_path}')
    except Exception:
        pass

    # List tables created
    created_tables = os.listdir(tables_dir) if os.path.exists(tables_dir) else []
    if created_tables:
        print('Tables saved:')
        for t in created_tables:
            print(' -', os.path.join(tables_dir, t))


def main():
    nb = sys.argv[1] if len(sys.argv) > 1 else 'Untitled10_(5).ipynb'
    nb_path = Path(nb)
    if not nb_path.exists():
        print(f'Notebook not found: {nb_path.resolve()}')
        return
    out_dir = os.path.join(Path(nb_path).parent, 'notebook_outputs')
    ensure_dir(out_dir)
    extract_images_and_text(str(nb_path), out_dir)


if __name__ == '__main__':
    main()
