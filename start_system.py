#!/usr/bin/env python3
"""
Legal AI System - Root Startup Script Wrapper
This script runs the actual startup script in Legal_AI_System folder.
"""

import sys
import os
from pathlib import Path

def main():
    # Add Legal_AI_System to the python path
    system_dir = Path(__file__).resolve().parent / "Legal_AI_System"
    sys.path.insert(0, str(system_dir))
    
    # Import and run the actual startup script
    from Legal_AI_System.start_system import main as run_system
    run_system()

if __name__ == "__main__":
    main()
