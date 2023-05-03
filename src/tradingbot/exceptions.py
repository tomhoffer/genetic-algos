class InvalidTradeActionException(Exception):
    """Exception raised for invalid trading actions.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message="Invalid trade action!"):
        self.message = message
        super().__init__(self.message)
