import os
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime
from logging_config import setup_logging, get_logger, cleanup_old_logs

setup_logging()
logger = get_logger(__name__)

# Change to script directory so file paths work correctly
# Important when script is run from cron or different directory
script_dir = Path(__file__).parent.absolute()
os.chdir(script_dir)

#Subprocess execution function
def run_script(script_name):
    """Run a Python script and return True if successful, False if failed"""
    try:
        print(f"Running {script_name}...")
        # Runs another Python script (script_name) as a subprocess
        result = subprocess.run(
            [sys.executable, script_name],  # Command: run script_name using the same Python interpreter
            capture_output=True,  # Captures stdout/stderr (instead of printing to terminal)
            text=True,  # Returns output as strings (not bytes)
            check=True  # Raises CalledProcessError if the subprocess fails
        )
        print(f"{script_name} completed successfully")
        # Log output if present
        if result.stdout:
            print(f"Output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        # Script returned non-zero exit code (error occurred)
        print(f"{script_name} failed with exit code {e.returncode}")
        if e.stderr:
            print(f"Error: {e.stderr}")
        if e.stdout:
            print(f"Output: {e.stdout}")
        return False
    except Exception as e:
        print(f"{script_name} failed with exception: {str(e)}")
        return False

#API server launcher
def start_api_server():
    """Start the FastAPI server in the background"""
    try:
        # Start server as background process (non-blocking)
        process = subprocess.Popen(
            [sys.executable, "main.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # Give the server time to start
        time.sleep(5)

        # Check if process is still running (didn't crash during startup)
        if process.poll() is None:
            # poll() returns None if process is running, exit code if terminated
            print("Fire Weather Index API server started successfully on http://localhost:8000")
            return process
        else:
            # Process terminated during startup
            stdout, stderr = process.communicate()
            print(f"Fire Weather Index API server failed to start")
            if stderr:
                print(f"Error: {stderr}")
            return None
    except Exception as e:
        print(f"Failed to start API server: {str(e)}")
        return None

#Data pipeline orchestrator
def run_data_pipeline():
    """Execute the Fire Weather Index data pipeline in sequence.
    Runs weather collection followed by FWI calculations.
    
    Returns:
        bool: True if all scripts succeeded, False if any failed
    """
    # List of scripts to run in order
    scripts = [
        "collect_weather_grid.py",  # Get today's weather data
        "fire_risk.py"              # Calculate Fire Weather Index predictions
    ]

    failed_scripts = []
    
    # Run each script in sequence
    for script in scripts:
        # Check if script file exists before attempting to run
        if os.path.exists(script):
            if not run_script(script):
                failed_scripts.append(script)
        else:
            failed_scripts.append(f"{script} (file not found)")
    
    if len(failed_scripts) == 0:
        print("Fire Weather Index data pipeline completed successfully!")
        return True
    else:
        print("Fire Weather Index pipeline failed scripts:")
        for script in failed_scripts:
            print(f"  - {script}")
        return False

#Full system startup 
def main():
    """Main function to run the complete Fire Weather Index system. Executes pipeline then starts API server for continuous operation.
    """
    try: 
        print("=" * 70)
        print("Forest Fire Risk Prediction System - Fire Weather Index")
        print(f"Starting system at {datetime.now()}")
        print("System: Canadian Fire Weather Index Algorithm")
        print("=" * 70)
        
        # Step 1: Run the data pipeline (weather + fire weather calculations)
        pipeline_success = run_data_pipeline() 

        if not pipeline_success:
            print("\n Fire Weather Index pipeline failed, but continuing with API server...")
        else:
            print("\n Fire Weather Index calculations complete")
        
        cleanup_old_logs(days_to_keep=45)
        
        # Step 2: Start the API server
        server_process = start_api_server()

        if server_process:
            # Server started successfully
            print("Fire Weather Index System Running")
            print("=" * 50)
            print(" Backend API: http://localhost:8000")
            print(" Health Check: http://localhost:8000/health")
            print(" API Info: http://localhost:8000/api/model/info")
            print(" Fire Risk Data: http://localhost:8000/api/predict/fire-risk")

            # Keep script running and monitor server process
            try:
                server_process.wait() # Block until server process terminates
            except KeyboardInterrupt:
                # User pressed Ctrl+C to stop server
                print("\n\nShutting down Fire Weather Index server...")
                server_process.terminate()
                server_process.wait() # Wait for process to fully terminate
                print("Server stopped")
        else:
            # Server failed to start
            print("\nFailed to start Fire Weather Index API server")
            return 1 # Exit with error code
        
    except KeyboardInterrupt:
        # User interrupted during pipeline execution
        print("\n\nOperation cancelled")
        return 1
    except Exception as e:
        print(f"\n Unexpected error: {str(e)}")
        return 1
    
    return 0 #Success

#Pipeline only mode, for cron jobs
def pipeline_only():
    """Run only the Fire Weather Index data pipeline without starting the server. Used by cron jobs for hourly data updates."""
    print("=" * 70)
    print("Fire Weather Index Pipeline Mode")
    print(f"Running pipeline-only mode at {datetime.now()}")
    print("=" * 70)
    
    # Run only the data pipeline
    success = run_data_pipeline()
    
    # Trigger API server reload if pipeline succeeded
    if success: 
        try:
            # Send HTTP POST request to reload endpoint
            import requests
            response = requests.post('http://localhost:8000/api/system/reload', timeout=10)
            print(f"Fire Weather Index system reload triggered: {response.status_code}")
        except Exception as e:
            print(f"Could not trigger system reload: {e}")

    # Report final status      
    if success:
        print("Fire Weather Index pipeline completed successfully")
        return 0
    else:
        print("Fire Weather Index pipeline failed")
        return 1

# Checks if the script is being run directly (not imported as a module)
if __name__ == "__main__":
    # Parse command line arguments to determine mode
    if len(sys.argv) > 1 and sys.argv[1] == "--pipeline-only":
        # Just run the pipeline for cron jobs
        sys.exit(pipeline_only())
    else:
        # Full mode: Run pipeline + start server
        sys.exit(main())