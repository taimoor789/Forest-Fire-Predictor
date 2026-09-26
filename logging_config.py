import logging
import logging.handlers
import os
from datetime import datetime
import sys

def setup_logging(
    log_level=logging.INFO,
    log_dir="logs",
    console_output=True,
    file_output=True
):
    """
    Configure logging with rotation and proper formatting.
    
    Args:
        log_level: Logging level (default: INFO)
        log_dir: Directory for log files (default: "logs")
        console_output: Whether to log to console (default: True)
        file_output: Whether to log to files (default: True)
    
    Returns:
        Logger instance
    """
    
    # Create logs directory if it doesn't exist
    if file_output:
        os.makedirs(log_dir, exist_ok=True)
    
    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Clear any existing handlers
    root_logger.handlers.clear()
    
    # Formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler (if enabled)
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # File handlers (if enabled)
    if file_output:
        # General log file with rotation (10MB max, keep 5 files)
        general_log = os.path.join(log_dir, "fire_weather.log")
        file_handler = logging.handlers.RotatingFileHandler(
            general_log,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        
        # Error log file (errors only)
        error_log = os.path.join(log_dir, "errors.log")
        error_handler = logging.handlers.RotatingFileHandler(
            error_log,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)
        root_logger.addHandler(error_handler)
        
        # Daily log file (one per day)
        today = datetime.now().strftime("%Y-%m-%d")
        daily_log = os.path.join(log_dir, f"daily_{today}.log")
        daily_handler = logging.FileHandler(daily_log, encoding='utf-8')
        daily_handler.setLevel(log_level)
        daily_handler.setFormatter(formatter)
        root_logger.addHandler(daily_handler)
    
    # Suppress noisy third-party loggers
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("Logging system initialized")
    logger.info(f"Log level: {logging.getLevelName(log_level)}")
    logger.info(f"Console output: {console_output}")
    logger.info(f"File output: {file_output}")
    if file_output:
        logger.info(f"Log directory: {os.path.abspath(log_dir)}")
    logger.info("=" * 60)
    
    return logger


def get_logger(name):
    """
    Get a logger instance for a specific module.
    
    Usage:
        logger = get_logger(__name__)
    """
    return logging.getLogger(name)


def cleanup_old_logs(log_dir="logs", days_to_keep=30):
    """
    Delete log files older than specified days.
    
    Args:
        log_dir: Directory containing log files
        days_to_keep: Number of days to retain logs
    """
    import glob
    import time
    
    logger = get_logger(__name__)
    
    if not os.path.exists(log_dir):
        return
    
    now = time.time()
    cutoff = now - (days_to_keep * 86400)  # 86400 seconds = 1 day
    
    deleted_count = 0
    for log_file in glob.glob(os.path.join(log_dir, "*.log*")):
        try:
            if os.path.getmtime(log_file) < cutoff:
                os.remove(log_file)
                deleted_count += 1
                logger.debug(f"Deleted old log: {log_file}")
        except Exception as e:
            logger.warning(f"Could not delete {log_file}: {e}")
    
    if deleted_count > 0:
        logger.info(f"Cleaned up {deleted_count} old log files")


if __name__ == "__main__":
    # Test the logging configuration
    logger = setup_logging(log_level=logging.DEBUG)
    
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    
    # Test from different module
    test_logger = get_logger("test_module")
    test_logger.info("This is from a test module")
    
