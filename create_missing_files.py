# create_missing_files.py
import os

def create_missing_structure():
    """Create the required project structure"""
    
    # Create directories
    directories = ["agents", "data_loader", "data"]
    for dir_name in directories:
        os.makedirs(dir_name, exist_ok=True)
        print(f"✅ Created directory: {dir_name}/")
    
    # Create empty __init__.py files
    init_files = ["agents/__init__.py", "data_loader/__init__.py"]
    for file_path in init_files:
        with open(file_path, 'w') as f:
            f.write("# Package initialization\n")
        print(f"✅ Created file: {file_path}")
    
    print("\n📁 Project structure created!")
    print("👉 Now add your PDF and CSV files to the data/ directory")
    print("👉 Then run: streamlit run app.py")

if __name__ == "__main__":
    create_missing_structure()