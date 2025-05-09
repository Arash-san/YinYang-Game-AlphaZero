#!/usr/bin/env python
import os
import sys

if __name__ == "__main__":
    # Add the current directory to Python path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    # Run the example script
    example_path = os.path.join("src", "yin_yang", "ai", "example.py")
    
    # Check if the example file exists
    if not os.path.exists(example_path):
        print(f"Error: Example file not found at {example_path}")
        sys.exit(1)
    
    # Execute the example
    os.system(f"python {example_path}") 