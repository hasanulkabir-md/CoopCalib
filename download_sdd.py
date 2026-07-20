import gdown
import os

os.makedirs('data/raw/sdd', exist_ok=True)

print("Downloading SDD dataset from Google Drive...")
gdown.download_folder(
    'https://drive.google.com/drive/folders/1zAa9SnPBRxPSBNfNHhXVCrOGCrLPz6Wd',
    output='data/raw/sdd',
    quiet=False
)
print("Done. SDD saved to data/raw/sdd/")
