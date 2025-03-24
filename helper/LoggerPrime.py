import os
from datetime import datetime
from enum import Enum
from typing import Optional, Union, Any

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.syntax import Syntax
from rich.box import ROUNDED


class LogLevel(Enum):
    """Log level enumeration with corresponding colors for console output"""
    DEBUG = ("debug", "bright_black")
    INFO = ("info", "bright_blue")
    SUCCESS = ("success", "green")
    WARNING = ("warning", "yellow")
    ERROR = ("error", "red")
    CRITICAL = ("critical", "bold red")


class LoggerPrime:
    """
    Unified logging class that handles both console and file logging
    with rich formatting for console output and file logging capabilities.
    """
    
    def __init__(
        self, 
        name: str = "app", 
        log_file: Optional[str] = "log.txt", 
        console: Optional[Console] = None,
        min_console_level: LogLevel = LogLevel.INFO,
        min_file_level: LogLevel = LogLevel.DEBUG,
        enabled: bool = True
    ):
        """
        Initialize the logger.
        
        Args:
            name: Logger name (used in log messages)
            log_file: Path to log file (None to disable file logging)
            console: Rich console instance (creates a new one if None)
            min_console_level: Minimum level for console output
            min_file_level: Minimum level for file output
            enabled: Whether logging is enabled
        """
        self.name = name
        self.log_file = log_file
        self.console = console or Console()
        self.min_console_level = min_console_level
        self.min_file_level = min_file_level
        self.enabled = enabled
        
        # Initialize log file if specified
        if log_file and enabled:
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)
            
            with open(log_file, 'a') as f:
                f.write(f"\n{'=' * 50}\n")
                f.write(f"Log initialized: {datetime.now()}\n")
                f.write(f"Logger: {name}\n")
                f.write(f"{'=' * 50}\n")
    
    def _should_log(self, level: LogLevel, to_console: bool) -> bool:
        """Check if we should log at this level"""
        if not self.enabled:
            return False
            
        min_level = self.min_console_level if to_console else self.min_file_level
        return level.value[0] >= min_level.value[0]
    
    def _format_message(self, level: LogLevel, message: str) -> str:
        """Format a message for logging to file"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"[{timestamp}] [{level.value[0].upper()}] [{self.name}] {message}"
    
    def _log_to_file(self, level: LogLevel, message: str) -> None:
        """Log a message to the file if file logging is enabled"""
        if not self.log_file or not self._should_log(level, to_console=False):
            return
            
        formatted = self._format_message(level, message)
        try:
            with open(self.log_file, 'a') as f:
                f.write(formatted + '\n')
        except Exception as e:
            # Fallback to console if file logging fails
            self.console.print(f"[bold red]Failed to write to log file: {str(e)}[/bold red]")
    
    def log(self, level: LogLevel, message: str, **kwargs) -> None:
        """
        Log a message at the specified level.
        
        Args:
            level: The log level
            message: The message to log
            **kwargs: Additional arguments for rich.console.print
        """
        # Log to file first
        self._log_to_file(level, message)
        
        # Log to console if enabled for this level
        if self._should_log(level, to_console=True):
            style = kwargs.pop('style', level.value[1])
            self.console.print(f"[{style}]{message}[/{style}]", **kwargs)
    
    def debug(self, message: str, **kwargs) -> None:
        """Log a debug message"""
        self.log(LogLevel.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs) -> None:
        """Log an info message"""
        self.log(LogLevel.INFO, message, **kwargs)
    
    def success(self, message: str, **kwargs) -> None:
        """Log a success message"""
        self.log(LogLevel.SUCCESS, message, **kwargs)
    
    def warning(self, message: str, **kwargs) -> None:
        """Log a warning message"""
        self.log(LogLevel.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs) -> None:
        """Log an error message"""
        self.log(LogLevel.ERROR, message, **kwargs)
    
    def critical(self, message: str, **kwargs) -> None:
        """Log a critical error message"""
        self.log(LogLevel.CRITICAL, message, **kwargs)
    
    def panel(self, content: Union[str, Text], **kwargs) -> None:
        """
        Display a rich panel with the given content.
        
        Args:
            content: Panel content (string or rich Text)
            **kwargs: Additional arguments for rich.panel.Panel
        """
        if not self.enabled:
            return
            
        # Default styling based on kwargs or info style
        style = kwargs.pop('style', 'bright_blue')
        title = kwargs.pop('title', None)
        
        # Create and display the panel
        panel = Panel(
            content,
            title=title,
            style=style,
            border_style=kwargs.pop('border_style', style),
            box=kwargs.pop('box', ROUNDED),
            **kwargs
        )
        self.console.print(panel)
    
    def rule(self, title: str = "", **kwargs) -> None:
        """Display a horizontal rule with optional title"""
        if not self.enabled:
            return
            
        style = kwargs.pop('style', 'bright_blue')
        self.console.rule(f"[bold {style}]{title}[/bold {style}]", **kwargs)
    
    def code(self, code: str, language: str = "python", **kwargs) -> None:
        """
        Display formatted code with syntax highlighting.
        
        Args:
            code: The code to display
            language: Programming language for syntax highlighting
            **kwargs: Additional arguments for rich.syntax.Syntax
        """
        if not self.enabled:
            return
            
        syntax = Syntax(
            code,
            language,
            theme=kwargs.pop('theme', 'monokai'),
            line_numbers=kwargs.pop('line_numbers', True),
            **kwargs
        )
        self.console.print(syntax)
    
    def table(self, data: Any, **kwargs) -> None:
        """
        Display data in a table format. Accepts any data supported by rich.table.
        
        Args:
            data: The data to display in table format
            **kwargs: Additional arguments for console.print
        """
        if not self.enabled:
            return
            
        self.console.print(data, **kwargs)
    
    def log_llm_interaction(self, prompt: str, response: str) -> None:
        """
        Log interactions with an LLM, both to file and optionally console if debug is enabled.
        
        Args:
            prompt: The prompt sent to the LLM
            response: The response received from the LLM
        """
        if not self.log_file or not self.enabled:
            return
            
        try:
            with open(self.log_file, 'a') as f:
                f.write("\n>>>>>>>>>> TO LLM:\n")
                f.write(f"{prompt}\n")
                f.write("\n>>>>>>>>>> FROM LLM:\n")
                f.write(f"{response}\n")
        except Exception as e:
            self.error(f"Failed to log LLM interaction: {str(e)}")
        
        # Debug output to console if enabled
        if self._should_log(LogLevel.DEBUG, to_console=True):
            self.debug("LLM Interaction logged to file")
    
    def set_level(self, console_level: LogLevel = None, file_level: LogLevel = None) -> None:
        """Update log levels for console and/or file"""
        if console_level:
            self.min_console_level = console_level
        if file_level:
            self.min_file_level = file_level
    
    def disable(self) -> None:
        """Disable logging"""
        self.enabled = False
    
    def enable(self) -> None:
        """Enable logging"""
        self.enabled = True


# Example usage:
if __name__ == "__main__":
    # Create a logger with default settings
    logger = LoggerPrime(name="example")
    
    # Basic logging
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.success("This is a success message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")
    
    # Rich panels
    logger.panel("This is a panel with default styling")
    logger.panel("This is a success panel", style="green", title="Success")
    
    # Code highlighting
    sample_code = """def hello_world():
    print("Hello, world!")
    return 42"""
    logger.code(sample_code)
    
    # Rules
    logger.rule("Section Divider")