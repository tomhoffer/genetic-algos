class InvalidTradeActionException(Exception):
    """Exception raised for invalid trading actions.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message="Invalid trade action!"):
        self.message = message
        super().__init__(self.message)


class BadTradingWeightsException(Exception):
    """Exception raised for bad trading weights.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message="Bad Trading weights encountered!"):
        self.message = message
        super().__init__(self.message)


class MissingHistoricalDataException(Exception):
    """Exception raised when doing computations over missing historical data (empty dataframe)

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message="Historical data not available. Download them before further computation."):
        self.message = message
        super().__init__(self.message)


class NotEnoughDataException(Exception):
    """Exception raised when size of the dataframe is too small to perform desired operation

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message="Too few rows in historical data!"):
        self.message = message
        super().__init__(self.message)


class SentimentDataDownloadFailedException(Exception):
    """Exception raised on error downloading sentiment data

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class NoDataFoundException(Exception):
    """Exception raised when no record has been found in the DB matching the query

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message):
        self.message = message
        super().__init__(self.message)
