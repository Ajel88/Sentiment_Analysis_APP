#!/usr/bin/env python3
"""
Alternative runner script
"""
import subprocess
import sys
import os

def main():
    # Ensure we're in the right directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Run the main application
    subprocess.run([sys.executable, "main.py"] + sys.argv[1:])

if __name__ == "__main__":
    main()