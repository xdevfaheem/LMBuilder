import logging
import os
import time

class LMBuilderLogger:
    
    """
    A flexible and customizable logger class for handling log messages with advanced features.

    This class allows you to create and configure a logger with various options, such as custom log levels,
    log file rotation, log format, log file size limits, and more. It provides the ability to log messages
    to both files and the console with different log levels.

    Attributes:
        log_dir (str): The directory where log files will be stored. If the directory doesn't exist, it will be created.
        file_name (str): The base name of the log file.
        log_level (int): The global log level for the logger.
        console_log_level (int): The log level for console output.
        log_format (str): The log message format.
        max_log_file_size (int): Maximum size (in bytes) for each log file before rotation.
        backup_count (int): Number of backup log files to retain.
        logger (logging.Logger): The configured logger instance

    Examples:
        To create an LMBuilderLogger instance and log messages:

        >>> logger_instance = LMBuilderLogger(
        ...     log_dir="lmbuilder_logs",
        ...     file_name="data_prep",
        ...     log_level=logging.DEBUG,
        ...     console_log_level=logging.INFO,
        ...     log_format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        ...     max_log_file_size=10*1024*1024,
        ...     backup_count=5
        ... )
        >>> logger = logger_instance.logger
        >>> logger.info("Starting Data Preparation...")
        >>> logger_instance.log_success("Data Preparation Completed!")
        >>> logger_instance.log_timestamp()
        
    Expected Output:
        2023-11-02 12:34:56 - LMBuilderLogger - INFO - Starting Data Preparation...
        2023-11-02 12:57:34 - LMBuilderLogger - SUCCESS - Data Preparation Completed!
        2023-11-02 12:57:34 - LMBuilderLogger - INFO - Timestamp: 2023-11-02 12:34:56

    Methods:
        log_success:
            Log a custom success message with a custom log level.

        log_timestamp:
            Log a timestamp message with the default INFO log level.
    """
    
    def __init__(
        self,
        log_dir="logs",
        file_name="my_log",
        log_level=logging.DEBUG,
        console_log_level=logging.INFO,
        log_format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        max_log_file_size=10*1024*1024,
        backup_count=3
        ):
        
        """
        Initialize the LMBuilderLogger instance with configurable logging settings.

        Args:
            log_dir (str): The directory where log files will be stored.
            file_name (str): The base name of the log file.
            log_level (int): The global log level for the logger.
            console_log_level (int): The log level for console output.
            log_format (str): The log message format.
            max_log_file_size (int): Maximum size (in bytes) for each log file before rotation.
            backup_count (int): Number of backup log files to retain.
        """
        
        
        self.log_dir = log_dir
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.file_name = file_name
        self.log_level = log_level
        self.console_log_level = console_log_level
        self.log_format = log_format
        self.max_log_file_size = max_log_file_size
        self.backup_count = backup_count
        self.logger = self.configure_logging()

    def configure_logging(self):
        
        """
        Configure the logger with the specified settings.

        This method sets up log file handlers, console handlers, log levels, and log formatting.
        It also adds a custom log level for success messages.

        Returns:
            logger (logging.Logger): The custom configured logger instance for logging messages.
        """

        logger = logging.getLogger(__name__)
        logger.setLevel(self.log_level)

        formatter = logging.Formatter(self.log_format)

        log_file = os.path.join(self.log_dir, f"{self.file_name}.log")

        logging.SUCCESS = 25
        logging.addLevelName(logging.SUCCESS, "SUCCESS")

        file_handler = logging.handlers.RotatingFileHandler(log_file, maxBytes=self.max_log_file_size, backupCount=self.backup_count)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(self.log_level)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(self.console_log_level)

        class InfoFilter(logging.Filter):
            def filter(self, record):
                return record.levelno in {logging.INFO, logging.SUCCESS}

        file_handler.addFilter(InfoFilter())

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger

    def log_success(self, message):
        """
        Log a success message with a custom log level.

        Args:
            message (str): The success message to be logged.
        """
        self.logger.log(logging.SUCCESS, message)

    def log_timestamp(self):
        """
        Log a timestamp message with the default INFO log level.
        """
        self.logger.info(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
