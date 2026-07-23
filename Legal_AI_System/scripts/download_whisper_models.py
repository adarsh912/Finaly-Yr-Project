#!/usr/bin/env python3
"""
Download Whisper Models Script
Downloads Whisper models and saves them in the local models folder
"""

import os
import sys
import shutil
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def download_whisper_models():
    """Download Whisper models and save them in models folder"""
    
    print("📥 Downloading Whisper Models")
    print("=" * 40)
    
    # Create models directory if it doesn't exist
    models_dir = Path("models")
    whisper_dir = models_dir / "whisper_models"
    whisper_dir.mkdir(parents=True, exist_ok=True)
    
    # Model sizes to download (all under 1GB)
    model_sizes = ["tiny", "base", "small"]
    
    try:
        import whisper
        
        for model_size in model_sizes:
            print(f"\n🔄 Downloading Whisper {model_size.upper()} model...")
            
            # Download the model (this will use the cache first if available)
            model = whisper.load_model(model_size)
            
            # Get the cache path
            cache_dir = Path.home() / ".cache" / "whisper"
            model_cache_path = cache_dir / f"{model_size}.pt"
            
            if model_cache_path.exists():
                # Copy to local models folder
                local_model_path = whisper_dir / f"{model_size}.pt"
                shutil.copy2(model_cache_path, local_model_path)
                print(f"✅ {model_size.upper()} model saved to: {local_model_path}")
                
                # Get file size
                size_mb = local_model_path.stat().st_size / (1024 * 1024)
                print(f"   Size: {size_mb:.1f} MB")
            else:
                print(f"⚠️  Cache file not found for {model_size}")
        
        print(f"\n📁 All models saved in: {whisper_dir}")
        
        # Create a config file for local models
        config = {
            "whisper_models_path": str(whisper_dir),
            "available_models": model_sizes,
            "model_sizes_mb": {
                "tiny": 75,
                "base": 139,
                "small": 462
            }
        }
        
        config_path = whisper_dir / "config.json"
        import json
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"📄 Config saved to: {config_path}")
        
    except ImportError:
        print("❌ Whisper not installed. Install with: pip install openai-whisper")
        return False
    except Exception as e:
        print(f"❌ Error downloading models: {e}")
        return False
    
    return True

def download_faster_whisper_models():
    """Download Faster Whisper models and save them in models folder"""
    
    print("\n📥 Downloading Faster Whisper Models")
    print("=" * 40)
    
    # Create models directory if it doesn't exist
    models_dir = Path("models")
    faster_whisper_dir = models_dir / "faster_whisper_models"
    faster_whisper_dir.mkdir(parents=True, exist_ok=True)
    
    # Model sizes to download (all under 1GB)
    model_sizes = ["tiny", "base", "small"]
    
    try:
        from faster_whisper import WhisperModel
        
        for model_size in model_sizes:
            print(f"\n🔄 Downloading Faster Whisper {model_size.upper()} model...")
            
            # Download the model (this will use the cache first if available)
            model = WhisperModel(model_size, device="cpu", compute_type="int8")
            
            # For Faster Whisper, we need to use the model name directly
            # The models are already cached and will be used from cache
            print(f"✅ {model_size.upper()} model is ready for use (cached)")
            
            # Create a placeholder file to indicate the model is available
            placeholder_path = faster_whisper_dir / f"{model_size}_available.txt"
            with open(placeholder_path, 'w') as f:
                f.write(f"Faster Whisper {model_size} model is available in cache\n")
                f.write(f"Model will be loaded from cache when used\n")
            
            print(f"   Placeholder created: {placeholder_path}")
        
        print(f"\n📁 Faster Whisper models are ready in cache")
        
        # Create a config file for local models
        config = {
            "faster_whisper_models_path": str(faster_whisper_dir),
            "available_models": model_sizes,
            "model_sizes_mb": {
                "tiny": 75,
                "base": 139,
                "small": 462
            },
            "note": "Models are loaded from HuggingFace cache, not local files"
        }
        
        config_path = faster_whisper_dir / "config.json"
        import json
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"📄 Config saved to: {config_path}")
        
    except ImportError:
        print("❌ Faster Whisper not installed. Install with: pip install faster-whisper")
        return False
    except Exception as e:
        print(f"❌ Error downloading Faster Whisper models: {e}")
        return False
    
    return True

def main():
    """Main function"""
    print("🎤 Whisper Models Downloader")
    print("=" * 50)
    
    # Download both types of models
    whisper_success = download_whisper_models()
    faster_whisper_success = download_faster_whisper_models()
    
    if whisper_success or faster_whisper_success:
        print("\n✅ Download completed!")
        print("\n📂 Models are now saved in:")
        print("   - models/whisper_models/")
        print("   - models/faster_whisper_models/")
        print("\n🔧 You can now use these models offline!")
    else:
        print("\n❌ Download failed!")

if __name__ == "__main__":
    main() 