# test_structure.py
import os
import sys

def check_project_structure():
    print("🔍 Checking project structure...")
    
    # Check current directory
    current_dir = os.getcwd()
    print(f"📁 Current directory: {current_dir}")
    
    # Check for required directories
    required_dirs = ["agents", "data_loader", "data"]
    for dir_name in required_dirs:
        exists = os.path.exists(dir_name)
        status = "✅" if exists else "❌"
        print(f"{status} {dir_name}/: {'Exists' if exists else 'MISSING'}")
        
        if exists and dir_name != "data":
            files = os.listdir(dir_name)
            print(f"   Files: {files}")
    
    # Check for required files
    required_files = [
        "app.py",
        "agents/__init__.py",
        "agents/multi_agent_system.py", 
        "agents/base_agent.py",
        "agents/maintenance_scheduler.py",
        "agents/field_support.py",
        "agents/workload_manager.py",
        "data_loader/__init__.py",
        "data_loader/document_processor.py"
    ]
    
    for file_path in required_files:
        exists = os.path.exists(file_path)
        status = "✅" if exists else "❌"
        print(f"{status} {file_path}: {'Exists' if exists else 'MISSING'}")
    
    # Check data directory contents
    data_dir = "data/"
    if os.path.exists(data_dir):
        files = os.listdir(data_dir)
        print(f"📊 Files in data/: {len(files)}")
        for file in files:
            print(f"   - {file}")
    else:
        print("❌ data/ directory does not exist!")
    
    print("\n🎯 To fix the 'System not initialized' error:")
    print("1. Ensure all required files and directories exist")
    print("2. Add PDF/CSV files to the data/ directory")
    print("3. Run: streamlit run app.py")
    print("4. Click 'Initialize System' in the sidebar")

if __name__ == "__main__":
    check_project_structure()