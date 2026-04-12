import sys
from pathlib import Path

# Add root dir to path so we can import 'app' and 'environment'
root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from app import app

def main():
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
