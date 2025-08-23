#!/usr/bin/env python3
"""
Script to run all Python scripts recursively in the examples folder.
This script automatically discovers all .py files and executes them,
making it easy to add new scripts without modifying this file.
"""

import os
import sys
import subprocess
import time
from pathlib import Path
from typing import List, Tuple, Dict
import traceback


def find_python_scripts(directory: Path) -> List[Path]:
    """
    Recursively find all Python scripts in the given directory.
    
    Args:
        directory: Root directory to search in
        
    Returns:
        List of Path objects for all .py files found
    """
    scripts = []
    for root, dirs, files in os.walk(directory):
        # Skip __pycache__ and other common directories to avoid
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
        
        for file in files:
            if file.endswith('.py') and file != 'run_all.py':
                script_path = Path(root) / file
                scripts.append(script_path)
    
    return sorted(scripts)


def run_script(script_path: Path, timeout: int = 300) -> Tuple[bool, str, float]:
    """
    Run a single Python script and capture its output.
    
    Args:
        script_path: Path to the script to run
        timeout: Maximum execution time in seconds
        
    Returns:
        Tuple of (success, output, execution_time)
    """
    start_time = time.time()
    
    try:
        # Run the script with subprocess
        result = subprocess.run(
            [sys.executable, script_path.name],  # Use just the filename
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=script_path.parent  # Run from script's directory
        )
        
        execution_time = time.time() - start_time
        
        if result.returncode == 0:
            return True, result.stdout, execution_time
        else:
            error_msg = f"Script failed with return code {result.returncode}\n"
            if result.stderr:
                error_msg += f"STDERR: {result.stderr}\n"
            if result.stdout:
                error_msg += f"STDOUT: {result.stdout}"
            return False, error_msg, execution_time
            
    except subprocess.TimeoutExpired:
        execution_time = time.time() - start_time
        return False, f"Script timed out after {timeout} seconds", execution_time
    except Exception as e:
        execution_time = time.time() - start_time
        return False, f"Error running script: {str(e)}\n{traceback.format_exc()}", execution_time


def main():
    """Main function to run all discovered scripts."""
    # Get the directory where this script is located
    examples_dir = Path(__file__).parent
    
    print(f"🔍 Discovering Python scripts in: {examples_dir}")
    print("=" * 60)
    
    # Find all Python scripts
    scripts = find_python_scripts(examples_dir)
    
    if not scripts:
        print("❌ No Python scripts found!")
        return
    
    print(f"📁 Found {len(scripts)} Python script(s):")
    for script in scripts:
        print(f"   - {script.relative_to(examples_dir)}")
    print()
    
    # Run each script
    results: Dict[Path, Tuple[bool, str, float]] = {}
    
    for i, script in enumerate(scripts, 1):
        relative_path = script.relative_to(examples_dir)
        print(f"🚀 [{i}/{len(scripts)}] Running: {relative_path}")
        
        success, output, execution_time = run_script(script)
        results[script] = (success, output, execution_time)
        
        if success:
            print(f"   ✅ Success ({execution_time:.2f}s)")
            if output.strip():
                # Show first few lines of output if any
                lines = output.strip().split('\n')
                if len(lines) > 3:
                    print(f"   📄 Output (first 3 lines):")
                    for line in lines[:3]:
                        print(f"      {line}")
                    if len(lines) > 3:
                        print(f"      ... ({len(lines) - 3} more lines)")
                else:
                    print(f"   📄 Output: {output.strip()}")
        else:
            print(f"   ❌ Failed ({execution_time:.2f}s)")
            print(f"   📄 Error: {output}")
        
        print()
    
    # Summary
    print("=" * 60)
    print("📊 SUMMARY:")
    
    successful = sum(1 for success, _, _ in results.values() if success)
    failed = len(results) - successful
    
    print(f"   ✅ Successful: {successful}")
    print(f"   ❌ Failed: {failed}")
    print(f"   📁 Total: {len(results)}")
    
    if failed > 0:
        print("\n❌ Failed scripts:")
        for script, (success, output, execution_time) in results.items():
            if not success:
                relative_path = script.relative_to(examples_dir)
                print(f"   - {relative_path} ({execution_time:.2f}s)")
    
    # Exit with error code if any script failed
    if failed > 0:
        sys.exit(1)
    else:
        print("\n🎉 All scripts completed successfully!")


if __name__ == "__main__":
    main()
