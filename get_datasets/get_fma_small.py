import kagglehub
import shutil
from pathlib import Path

# Download to cache (returns cache path)
cache_path = kagglehub.dataset_download("aaronyim/fma-small")
print(f"Downloaded to: {cache_path}")

# Your desired location
target_dir = Path("./datasets")
target_dir.mkdir(parents=True, exist_ok=True)

# Copy all files from cache to your folder
for item in Path(cache_path).iterdir():
    if item.is_file():
        shutil.copy2(item, target_dir / item.name)
    else:
        shutil.copytree(item, target_dir / item.name)

print(f"Files copied to: {target_dir.absolute()}")