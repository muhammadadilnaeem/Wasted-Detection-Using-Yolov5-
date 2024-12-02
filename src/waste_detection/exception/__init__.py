import sys  # Import the sys module to access system-specific parameters and functions


def error_message_detail(error, error_detail: sys):
    # Extract the exception traceback details
    _, _, exc_tb = error_detail.exc_info()

    # Get the name of the file where the exception occurred
    file_name = exc_tb.tb_frame.f_code.co_filename

    # Format the error message to include the file name, line number, and error message
    error_message = "Error occurred in Python script name [{0}] line number [{1}] error message [{2}]".format(
        file_name, exc_tb.tb_lineno, str(error)
    )

    return error_message  # Return the formatted error message


class AppException(Exception):
    # Custom exception class that inherits from the built-in Exception class
    def __init__(self, error_message, error_detail):
        """
        Initialize the AppException with a custom error message and details.

        :param error_message: error message in string format
        """
        super().__init__(error_message)  # Call the base class constructor with the error message

        # Generate a detailed error message using the provided error details
        self.error_message = error_message_detail(
            error_message, error_detail=error_detail
        )

    def __str__(self):
        # Override the string representation of the exception to return the custom error message
        return self.error_message