"""
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
            print(f"Error starting app: {e}")

if __name__ == "__main__":
    main()
