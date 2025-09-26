"""
Project Structure Setup Script
This will organize your files in the structure Briefcase expects
"""

import os
import shutil
import sys

def setup_briefcase_structure():
    """Set up the correct directory structure for Briefcase"""
    
    print("🏗️ Setting up Briefcase project structure...")
    
    # Current files to look for
    app_files = [
        "onnx_money_detector_app.py",
        "android_specific_app.py",
        "money_detector_app.py",  # Original file name
    ]
    
    model_files = [
        "Money_lite/best_money_model.onnx",
        "best_money_model.onnx",
    ]
    
    # Create directory structure
    directories = [
        "src",
        "src/moneydetector", 
        "src/moneydetector/resources",
        "icons",
    ]
    
    print("📁 Creating directories...")
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"   ✅ {directory}")
    
    # Find and copy the main app file
    app_file_found = None
    for app_file in app_files:
        if os.path.exists(app_file):
            app_file_found = app_file
            break
    
    if app_file_found:
        destination = "src/moneydetector/app.py"
        shutil.copy2(app_file_found, destination)
        print(f"✅ Copied {app_file_found} -> {destination}")
        
        # Also create __main__.py as entry point
        main_py_content = f'''"""
Money Detector App - Main Entry Point
"""
from .app import *

def main():
    """Main entry point"""
    if __name__ == '__main__':
        # Import the main app class and run it
        try:
            # Try different possible app class names
            if 'AndroidMoneyDetectorApp' in globals():
                app = AndroidMoneyDetectorApp()
            elif 'ONNXMoneyDetectorApp' in globals():
                app = ONNXMoneyDetectorApp()
            elif 'MoneyDetectorApp' in globals():
                app = MoneyDetectorApp()
            else:
                print("No app class found")
                return
            
            app.run()
        except Exception as e:
            print(f"Error starting app: {{e}}")

if __name__ == "__main__":
    main()
'''
        
        with open("src/moneydetector/__main__.py", "w") as f:
            f.write(main_py_content)
        print("✅ Created src/moneydetector/__main__.py")
        
    else:
        print("❌ No app file found! Looking for:")
        for app_file in app_files:
            print(f"   - {app_file}")
        print("\nPlease make sure you have one of these files in the current directory.")
        return False
    
    # Create __init__.py
    init_py_content = '''"""
Money Detector App
AI-powered currency detection for the visually impaired
"""

__version__ = "1.0.0"
__author__ = "Your Name"

# Import main components
from .app import *

__all__ = ["AndroidMoneyDetectorApp", "ONNXMoneyDetectorApp", "MoneyDetectorApp"]
'''
    
    with open("src/moneydetector/__init__.py", "w") as f:
        f.write(init_py_content)
    print("✅ Created src/moneydetector/__init__.py")
    
    # Find and copy model file
    model_file_found = None
    for model_file in model_files:
        if os.path.exists(model_file):
            model_file_found = model_file
            break
    
    if model_file_found:
        destination = "src/moneydetector/resources/best_money_model.onnx"
        shutil.copy2(model_file_found, destination)
        print(f"✅ Copied {model_file_found} -> {destination}")
    else:
        print("⚠️ ONNX model not found! Looking for:")
        for model_file in model_files:
            print(f"   - {model_file}")
        print("   You'll need to copy the model manually or run the ONNX export first.")
    
    # Create simple icon if PIL is available
    create_simple_icon()
    
    # Show current structure
    print("\n📊 Current project structure:")
    show_directory_tree(".", max_depth=3)
    
    print("\n🎉 Project structure setup complete!")
    print("\n📝 Next steps:")
    print("1. Verify the structure above looks correct")
    print("2. Run: briefcase create android")
    print("3. Run: briefcase build android")
    
    return True

def create_simple_icon():
    """Create a simple icon using PIL if available"""
    try:
        from PIL import Image, ImageDraw
        
        # Create simple 512x512 icon
        icon = Image.new('RGBA', (512, 512), (41, 128, 185, 255))  # Blue background
        draw = ImageDraw.Draw(icon)
        
        # Draw a simple money symbol
        draw.ellipse([128, 128, 384, 384], fill=(255, 255, 255, 255), outline=(0, 0, 0, 255), width=8)
        
        # Try to draw rupee symbol (might not work without proper font)
        try:
            draw.text((220, 220), "₹", fill=(0, 0, 0, 255))
        except:
            draw.text((220, 220), "$", fill=(0, 0, 0, 255))
        
        # Save different sizes
        sizes = [512, 256, 128, 64, 32]
        for size in sizes:
            resized = icon.resize((size, size), Image.Resampling.LANCZOS)
            resized.save(f"icons/icon-{size}.png")
        
        # Main icon
        icon.save("icons/icon.png")
        print("✅ Created simple placeholder icons")
        
    except ImportError:
        print("⚠️ PIL not available for icon creation")
        print("   You can create icons manually or install: pip install Pillow")
        
        # Create empty icon files so Briefcase doesn't complain
        try:
            with open("icons/icon.png", "w") as f:
                f.write("")  # Empty file as placeholder
            print("✅ Created empty icon placeholder")
        except:
            pass

def show_directory_tree(path, prefix="", max_depth=3, current_depth=0):
    """Show directory structure"""
    if current_depth >= max_depth:
        return
        
    try:
        items = sorted(os.listdir(path))
        for i, item in enumerate(items):
            if item.startswith('.'):
                continue
                
            item_path = os.path.join(path, item)
            is_last = i == len(items) - 1
            
            current_prefix = "└── " if is_last else "├── "
            print(f"{prefix}{current_prefix}{item}")
            
            if os.path.isdir(item_path) and current_depth < max_depth - 1:
                extension_prefix = "    " if is_last else "│   "
                show_directory_tree(item_path, prefix + extension_prefix, max_depth, current_depth + 1)
                
    except PermissionError:
        pass

def check_current_files():
    """Check what files are currently in the directory"""
    print("🔍 Current files in directory:")
    
    current_files = []
    for item in os.listdir("."):
        if os.path.isfile(item):
            current_files.append(item)
    
    python_files = [f for f in current_files if f.endswith('.py')]
    toml_files = [f for f in current_files if f.endswith('.toml')]
    model_files = [f for f in current_files if 'model' in f.lower()]
    
    print(f"📄 Python files: {python_files}")
    print(f"📄 TOML files: {toml_files}")
    print(f"📄 Model files: {model_files}")
    
    # Check for Money_lite directory
    if os.path.isdir("Money_lite"):
        print("📁 Money_lite directory exists")
        try:
            money_lite_files = os.listdir("Money_lite")
            print(f"   Contents: {money_lite_files}")
        except:
            pass
    else:
        print("❌ Money_lite directory not found")

if __name__ == "__main__":
    print("🚀 Briefcase Project Structure Setup")
    print("=" * 50)
    
    # First, show what we have
    check_current_files()
    
    print("\n" + "=" * 30)
    
    # Set up the structure
    if setup_briefcase_structure():
        print("\n✅ Setup completed successfully!")
    else:
        print("\n❌ Setup failed - please check the errors above")
        
    print("\n💡 If you still get errors:")
    print("1. Make sure pyproject.toml exists and is valid")
    print("2. Check that src/moneydetector/app.py contains your app code")
    print("3. Verify the model file is in src/moneydetector/resources/")
    print("4. Run: briefcase create android -v (for verbose output)")