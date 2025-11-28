"""Logging configuration for GraphRAG pipeline."""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


def setup_logging(
    log_dir: Path = Path("logs"),
    level: int = logging.INFO,
    log_to_file: bool = False,  # Disabled by default - we use MarkdownLogger instead
    log_to_console: bool = False,  # Disabled by default - CLI handles output
) -> logging.Logger:
    """Set up logging configuration."""
    log_dir.mkdir(exist_ok=True)

    # Create logger
    logger = logging.getLogger("graphrag")
    logger.setLevel(level)

    # Clear existing handlers
    logger.handlers.clear()

    # Create formatters
    console_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )
    file_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    # File handler (disabled by default - use MarkdownLogger for structured logs)
    if log_to_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"graphrag_{timestamp}.log"
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


class MarkdownLogger:
    """Logger that writes structured markdown logs."""

    def __init__(self, log_dir: Path = Path("logs"), prefix: str = "graphrag"):
        """Initialize markdown logger."""
        log_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = log_dir / f"{prefix}_{timestamp}.md"
        self._write_header()

    def _write_header(self):
        """Write log file header."""
        with open(self.log_file, "w", encoding="utf-8") as f:
            f.write(f"# GraphRAG Execution Log\n\n")
            f.write(f"**Started:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("---\n\n")

    def _write(self, content: str):
        """Write content to log file."""
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(content)

    def section(self, title: str):
        """Write a section header."""
        self._write(f"\n## {title}\n\n")

    def subsection(self, title: str):
        """Write a subsection header."""
        self._write(f"\n### {title}\n\n")

    def info(self, message: str):
        """Write an info message."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self._write(f"**[{timestamp}]** {message}\n\n")

    def success(self, message: str):
        """Write a success message."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self._write(f"✅ **[{timestamp}]** {message}\n\n")

    def warning(self, message: str):
        """Write a warning message."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self._write(f"⚠️ **[{timestamp}]** {message}\n\n")

    def error(self, message: str):
        """Write an error message."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self._write(f"❌ **[{timestamp}]** {message}\n\n")

    def code_block(self, content: str, language: str = ""):
        """Write a code block."""
        self._write(f"```{language}\n{content}\n```\n\n")

    def json_block(self, content: str):
        """Write a JSON code block."""
        self.code_block(content, "json")

    def cypher_block(self, content: str):
        """Write a Cypher code block."""
        self.code_block(content, "cypher")

    def text_block(self, content: str):
        """Write a text code block."""
        self.code_block(content, "text")

    def list_items(self, items: list, bullet: str = "-"):
        """Write a list of items."""
        for item in items:
            self._write(f"{bullet} {item}\n")
        self._write("\n")

    def table(self, headers: list, rows: list):
        """Write a markdown table."""
        # Header row
        self._write("| " + " | ".join(headers) + " |\n")
        # Separator
        self._write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        # Data rows
        for row in rows:
            self._write("| " + " | ".join(str(cell) for cell in row) + " |\n")
        self._write("\n")

    def stats(self, stats: dict):
        """Write statistics as a formatted list."""
        for key, value in stats.items():
            self._write(f"- **{key}:** {value}\n")
        self._write("\n")

    def finalize(self):
        """Write log file footer."""
        self._write("---\n\n")
        self._write(f"**Completed:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
