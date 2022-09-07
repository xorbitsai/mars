import logging
import os


class FileLoggingHandler(logging.FileHandler):
    mars_temp_log = "MARS_TEMP_LOG"

    def __init__(self):
        file_name = os.environ.get(self.mars_temp_log)

        super(FileLoggingHandler, self).__init__(file_name, "a")
