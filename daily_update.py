import os
import subprocess
import sys
from pathlib import Path

# Change to script directory so file paths work correctly
script_dir = Path(__file__).parent.absolute()
os.chdir(script_dir)

def run_script(script_name):
    """Run a Python script and return True if successful, False if failed"""
    try:
        #Runs another Python script (script_name) as a subprocess
        result = subprocess.run(
            [sys.executable, script_name],  #Command: run script_name using the same Python interpreter
            capture_output=True,  # Captures stdout/stderr (instead of printing to terminal)
            text=True,  #Returns output as strings (not bytes)
            check=True  #Raises CalledProcessError if the subprocess fails
        )
        return True
    except:
        # If anything goes wrong, return False
        return False

def main():
    #Run all 3 scripts and report results
    # List of scripts to run in order
    scripts = [
        "collect_weather_grid.py",      #Get today's weather data
        "merge_weather_and_labels.py",  #Combine weather with fire history
        "fire_risk.py"                  #Train model and make predictions
    ]
    
    failed_scripts = []  # Keep track of which scripts failed
    
    # Run each script
    for script in scripts:
        if os.path.exists(script):  # Make sure file exists
            if not run_script(script):  # Try to run it
                failed_scripts.append(script)  # Add to failed list if it didn't work
        else:
            failed_scripts.append(f"{script} (file not found)")  # File missing
    
    # Display final results
    if len(failed_scripts) == 0:
        print("Pipeline completed successfully!")
    else:
        print("Pipeline failed!")
        print("Failed scripts:")
        for script in failed_scripts:
            print(f"  - {script}")

#Checks if the script is being run directly (not imported as a module)
if __name__ == "__main__":
    main()  #Calls the main() function only when the script is executed directly