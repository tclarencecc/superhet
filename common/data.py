from enum import Enum

from common.serde import Serde

class DataType(Enum):
    NOTIFICATION = 1
    QUERY = 2
    ANSWER = 3
    REQUEST_FILE = 4
    SEND_FILE = 5

class Notification(Serde):
    def __init__(self, json_str=""):
        self.message = ""
        super().__init__(DataType.NOTIFICATION, json_str)

class Query(Serde):
    def __init__(self, json_str=""):
        self.id = ""
        self.text = ""
        super().__init__(DataType.QUERY, json_str)

class Answer(Serde):
    def __init__(self, json_str=""):
        self.id = ""
        self.word = ""
        self.end = False
        super().__init__(DataType.ANSWER, json_str)

class Request_File(Serde):
    def __init__(self, json_str=""):
        self.id = ""
        self.file_type = ""
        super().__init__(DataType.REQUEST_FILE, json_str)

class Send_File(Serde):
    def __init__(self, json_str=""):
        self.id = ""
        self.part = ""
        self.binary = False
        self.end = False
        super().__init__(DataType.SEND_FILE, json_str)
