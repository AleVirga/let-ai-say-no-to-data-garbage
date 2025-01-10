
import datetime as dt
import json
import logging

from dqa.attributes.logging import LOG_RECORD_BUILTIN


class JSONFormatter(logging.Formatter):
    """
    A custom JSON formatter for Python logging.

    This formatter converts log records to JSON format,
    making it easier to parse log data programmatically.

    Attributes:
        fmt_keys (dict[str, str]): A mapping from the log record attributes to
        the desired JSON keys.

    Example:
        logger = logging.getLogger('my_logger')
        handler = logging.StreamHandler()
        formatter = JSONFormatter(fmt_keys={'levelname': 'severity', 'msg': 'message'})
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.warning('This is a test message.')
    """

    def __init__(
        self,
        *,
        fmt_keys=None,
    ):
        super().__init__()
        self.fmt_keys = fmt_keys if fmt_keys is not None else {}

    def format(self, record: logging.LogRecord) -> str:
        """Converts a LogRecord object to JSON format.

        Args:
            record (logging.LogRecord): The log record to format.

        Returns:
            str: A JSON string representing the log record.
        """
        message = self._prepare_log_dict(record)
        return json.dumps(message, default=str)

    def _prepare_log_dict(self, record: logging.LogRecord):
        """Prepares a dictionary for the log record, applying custom formatting.

        Args:
            record (logging.LogRecord): The log record to prepare.

        Returns:
            dict: A dictionary containing the formatted log record.
        """
        # Fixed fields that will always be included in the log record.
        always_fields = {
            "message": record.getMessage(),
            "timestamp": dt.datetime.fromtimestamp(
                record.created, tz=dt.timezone.utc
            ).isoformat(),
        }
        # Add exception info if present.
        if record.exc_info is not None:
            always_fields["exc_info"] = self.formatException(record.exc_info)

        # Add stack information if present.
        if record.stack_info is not None:
            always_fields["stack_info"] = self.formatStack(record.stack_info)

        # Map custom format keys and update with fixed fields.
        message = {
            key: (
                msg_val
                if (msg_val := always_fields.pop(val, None)) is not None
                else getattr(record, val)
            )
            for key, val in self.fmt_keys.items()
        }
        message.update(always_fields)

        # Include any additional attributes set on the record.
        for key, val in record.__dict__.items():
            if key not in LOG_RECORD_BUILTIN:
                message[key] = val

        return message


class NonErrorFilter(logging.Filter):
    """
    A custom logging filter that filters out all records with a level higher than INFO.

    This can be used to exclude warning, error, critical,
    and exception logs from a specific handler.

    Example:
        logger = logging.getLogger('my_logger')
        handler = logging.StreamHandler()
        filter = NonErrorFilter()
        handler.addFilter(filter)
        logger.addHandler(handler)
        logger.warning('This warning will not be shown.')
    """

    # @override
    def filter(self, record: logging.LogRecord):
        """Determines if the specified record should be logged.

        Args:
            record (logging.LogRecord): The log record to filter.

        Returns:
            bool: True if the record level is WARNING or lower, False otherwise.
        """
        return record.levelno <= logging.WARNING
