"""
Money Detector App - Main Entry Point
Save this as: src/moneydetector/__main__.py
"""

def main():
    """Main entry point for the Money Detector app"""
    try:
        # Import your app class
        from .app import AndroidMoneyDetectorApp
        
        print("🚀 Starting Money Detector App")
        app = AndroidMoneyDetectorApp()
        app.run()
        
    except ImportError as e:
        print(f"Import error: {e}")
        try:
            # Fallback to other possible app classes
            from .app import MoneyDetectorApp
            app = MoneyDetectorApp()
            app.run()
        except ImportError:
            print("❌ Could not find app class")
            print("Make sure your app.py file contains a MoneyDetectorApp class")

if __name__ == "__main__":
    main()