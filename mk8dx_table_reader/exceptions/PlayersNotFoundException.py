class LoadImgException(Exception):
    """Exception raised when points detected from screenshot don't add up to any amount of races from target number of players

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, ):
        self.message = "Table not found in the image"
        super().__init__(self.message)