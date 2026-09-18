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
script_dir = Path(__file__).parent.absolute()
os.chdir(script_dir)

def run_script(script_name):
    """Run a Python script and return True if successful, False if failed.

    stdout/stderr are inherited from the parent (not captured), so output
    streams live -- a 20-minute ECCC pull otherwise shows nothing in a CI
    log until it finishes and looks hung.
    """
    try:
        print(f"Running {script_name}...")
        subprocess.run(
            [sys.executable, "-u", script_name],
            check=True,  # Raises CalledProcessError if the subprocess fails
        )
        print(f"{script_name} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"{script_name} failed with exit code {e.returncode}")
        return False
    except Exception as e:
        print(f"{script_name} failed with exception: {str(e)}")
        return False

def start_api_server():
    """Start the FastAPI server in the background"""
    try:
        process = subprocess.Popen(
            [sys.executable, "main.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # Give the server time to start
        time.sleep(5)

        # Check if process is still running
        if process.poll() is None:
            print("Fire Weather Index API server started successfully on http://localhost:8000")
            return process
        else:
            stdout, stderr = process.communicate()
            print(f"Fire Weather Index API server failed to start")
            if stderr:
                print(f"Error: {stderr}")
            return None
    except Exception as e:
        print(f"Failed to start API server: {str(e)}")
        return None

def run_data_pipeline():
    """Run the Fire Weather Index data pipeline.

    collect_weather_grid_eccc.py (ECCC HRDPS/GDPS/HRDPA) is the primary
    weather collector; collect_weather_grid.py (Open-Meteo) is kept as a
    fallback, used only if the primary fails outright -- matching the
    "kept as fallback" intent already documented in that module's own
    docstring. This function used to call the Open-Meteo collector
    unconditionally, a leftover from before the ECCC migration.
    """
    collectors = ["collect_weather_grid_eccc.py", "collect_weather_grid.py"]

    collected = None
    for collector in collectors:
        if not os.path.exists(collector):
            print(f"{collector} not found, skipping")
            continue
        if run_script(collector):
            collected = collector
            break
        print(f"{collector} failed; trying next collector")

    if collected is None:
        print("Fire Weather Index pipeline failed: no weather collector succeeded")
        return False
    if collected != collectors[0]:
        print(f"WARNING: fell back to {collected} (primary ECCC collector unavailable)")

    if not run_script("fire_risk.py"):
        print("Fire Weather Index pipeline failed: fire_risk.py")
        return False

    print("Fire Weather Index data pipeline completed successfully!")
    return True
  
def main():
    """Main function to run the complete Fire Weather Index system"""
    try: 
        print("=" * 70)
        print("Forest Fire Risk Prediction System - Fire Weather Index")
        print(f"Starting system at {datetime.now()}")
        print("System: Canadian Fire Weather Index Algorithm")
        print("=" * 70)
        
        # Step 1: Run the data pipeline (weather + fire weather calculations)
        pipeline_success = run_data_pipeline() 

        if not pipeline_success:
            print("\nFire Weather Index pipeline failed, but continuing with API server...")
        else:
            print("\nFire Weather Index calculations complete")
        
        cleanup_old_logs(days_to_keep=30)
        
        # Step 2: Start the API server
        server_process = start_api_server()

        if server_process:
            print("\n" + "=" * 50)
            print("Fire Weather Index System Running")
            print("=" * 50)
            print("Backend API: http://localhost:8000")
            print("Health Check: http://localhost:8000/health")
            print("API Info: http://localhost:8000/api/model/info")
            print("Fire Risk Data: http://localhost:8000/api/predict/fire-risk")
            print("=" * 50)
            print("Press Ctrl+C to stop the server")
            print("=" * 50)

            # Keep the server running
            try:
                server_process.wait()
            except KeyboardInterrupt:
                print("\n\nShutting down Fire Weather Index server...")
                server_process.terminate()
                server_process.wait()
                print("Server stopped")
        else:
            print("\nFailed to start Fire Weather Index API server")
            return 1
        
    except KeyboardInterrupt:
        print("\n\nOperation cancelled")
        return 1
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        return 1
    
    return 0

def pipeline_only():
    """Run only the Fire Weather Index data pipeline without starting the server"""
    print("=" * 70)
    print("Fire Weather Index Pipeline Mode")
    print(f"Running pipeline-only mode at {datetime.now()}")
    print("=" * 70)
    
    # Run only the data pipeline
    success = run_data_pipeline()
    
    if success:
        print("Fire Weather Index pipeline completed successfully")
        print("Updated files will be loaded automatically by the API")
        cleanup_old_logs(days_to_keep=30)
        return 0
    else:
        print("Fire Weather Index pipeline failed")
        return 1

# Checks if the script is being run directly (not imported as a module)
if __name__ == "__main__":
    # Check command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "--pipeline-only":
        # Just run the pipeline for cron jobs
        sys.exit(pipeline_only())
    else:
        # Run full system
        sys.exit(main())