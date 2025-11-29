class DetectionError(Exception):
    """Exception raised when points detected from screenshot don't add up to any amount of races from target number of players

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message):
        self.message = message
        super().__init__(self.message)