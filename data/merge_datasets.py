import os
import shutil
import yaml
from pathlib import Path

# Unified classes
CLASSES = ['ball', 'player', 'court', 'paddle']
MERGED_DIR = Path('pickleball_merged_v2')

def remap_and_copy(dataset_path, class_map, split, prefix):
    """
    dataset_path: path to the roboflow dataset (e.g., 'ball-tracker-1')
    class_map: dict {old_idx: new_idx}
    split: 'train', 'valid', or 'test'
    prefix: string to avoid filename collisions
    """
    img_dir = dataset_path / split / 'images'
    lbl_dir = dataset_path / split / 'labels'
    
    target_img = MERGED_DIR / split / 'images'
    target_lbl = MERGED_DIR / split / 'labels'
    
    target_img.mkdir(parents=True, exist_ok=True)
    target_lbl.mkdir(parents=True, exist_ok=True)
    
    if not img_dir.exists(): return

    for img_file in img_dir.glob('*.*'):
        lbl_file = lbl_dir / (img_file.stem + '.txt')
        if not lbl_file.exists(): continue
        
        # Copy image with prefix
        new_name = f"{prefix}_{img_file.name}"
        shutil.copy(img_file, target_img / new_name)
        
        # Read and remap labels
        with open(lbl_file, 'r') as f:
            lines = f.readlines()
            
        new_lines = []
        for line in lines:
            parts = line.split()
            old_idx = int(parts[0])
            if old_idx in class_map:
                new_idx = class_map[old_idx]
                new_lines.append(f"{new_idx} {' '.join(parts[1:])}\n")
        
        with open(target_lbl / (f"{prefix}_{img_file.stem}.txt"), 'w') as f:
            f.writelines(new_lines)

def merge():
    if MERGED_DIR.exists():
        shutil.rmtree(MERGED_DIR)
    
    # Define maps based on the notebook
    # Dataset 1: ball-tracker-pczbl (class 0: ball)
    map1 = {0: 0} 
    # Dataset 2: pickleball-vision (0: court, 1: paddle, 2: player, 3: ball)
    map2 = {0: 2, 1: 3, 2: 1, 3: 0} # Fixed as per notebook cell 5
    # Dataset 3: pickleball-detection-1oqlw (0: ball, 1: player)
    map3 = {0: 0, 1: 1}

    datasets = [
        (Path('ball-tracker-pczbl-1'), map1, 'd1'),
        (Path('pickleball-vision-1'), map2, 'd2'),
        (Path('pickleball-detection-1oqlw-1'), map3, 'd3')
    ]

    for split in ['train', 'valid', 'test']:
        for path, cmap, pref in datasets:
            if path.exists():
                print(f"Merging {path} {split}...")
                remap_and_copy(path, cmap, split, pref)

    # Create data.yaml
    data_config = {
        'train': str((MERGED_DIR / 'train' / 'images').absolute()),
        'val': str((MERGED_DIR / 'valid' / 'images').absolute()),
        'test': str((MERGED_DIR / 'test' / 'images').absolute()),
        'nc': len(CLASSES),
        'names': CLASSES
    }
    
    with open(MERGED_DIR / 'data.yaml', 'w') as f:
        yaml.dump(data_config, f)
    
    print(f"✓ Merged dataset created at {MERGED_DIR}")

if __name__ == "__main__":
    merge()
