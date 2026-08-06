from loguru import logger as _loguru
from rich.logging import RichHandler

_loguru.remove()
_loguru.level("DEBUG", color="<cyan><bold>")
_loguru.level("INFO", color="<blue><bold>")
_loguru.level("WARNING", color="<white><bold><bg yellow>")
_loguru.level("ERROR", color="<white><bold><bg red>")
_loguru.level("CRITICAL", color="<white><bold><bg red>")
_loguru.level("SUCCESS", color="<black><bold><bg green>")


class Logger:
    """Module-level logging singleton: loguru, rendered through a Rich handler."""

    def __init__(self):
        self.logger = _loguru
        self._console = None
        self._log_to_console()

    def _log_to_console(self):
        if self._console is None:
            rich_handler = RichHandler(
                rich_tracebacks=True,
                markup=True,
                log_time_format="%H:%M:%S",
                show_path=False,
            )
            self._console = self.logger.add(
                rich_handler, format="{message}", colorize=True
            )

    def _unlog_from_console(self):
        if self._console is not None:
            try:
                self.logger.remove(self._console)
            except ValueError:
                pass  # already removed (e.g. set_verbosity(False) called twice)
            self._console = None

    def set_verbosity(self, verbose):
        """Turn console logging on or off.

        Parameters
        ----------
        verbose : bool
            ``True`` attaches the Rich console handler; ``False`` detaches it,
            silencing output without affecting other configured sinks.
        """
        if verbose:
            self._log_to_console()
        else:
            self._unlog_from_console()

    def debug(self, text):
        """Log ``text`` at DEBUG level.

        Parameters
        ----------
        text : str
        """
        self.logger.debug(text)

    def info(self, text):
        """Log ``text`` at INFO level.

        Parameters
        ----------
        text : str
        """
        self.logger.info(text)

    def warning(self, text):
        """Log ``text`` at WARNING level.

        Parameters
        ----------
        text : str
        """
        self.logger.warning(text)

    def error(self, text):
        """Log ``text`` at ERROR level.

        Parameters
        ----------
        text : str
        """
        self.logger.error(text)

    def critical(self, text):
        """Log ``text`` at CRITICAL level.

        Parameters
        ----------
        text : str
        """
        self.logger.critical(text)

    def success(self, text):
        """Log ``text`` at loguru's SUCCESS level.

        Parameters
        ----------
        text : str
        """
        self.logger.success(text)


logger = Logger()
