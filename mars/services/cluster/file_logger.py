
import os
import tempfile

from mars import oscar as mo


class FileLoggerActor(mo.Actor):
    mars_temp_log = "MARS_TEMP_LOG"

    def __init__(self):
        file_path = os.environ.get(self.mars_temp_log)
        if file_path is None:
            raise ValueError('Env {0} is not set!'.format(self.mars_temp_log))
        self._log_filename = file_path

    def fetch_logs(self, size) -> str:
        content = self._get_n_bytes_tail_file(size)
        return content

    def _get_n_bytes_tail_file(self, bytes_num) -> str:
        if bytes_num != -1:
            f_size = os.stat(self._log_filename).st_size
            target = f_size - bytes_num if f_size > bytes_num else 0
        else:
            target = 0
        with open(self._log_filename) as f:
            f.seek(target)
            if target == 0:
                res = f.read()
            else:
                f.readline()
                res = f.read()

        return res

    def __pre_destroy__(self):
        os.remove(self._log_filename)
