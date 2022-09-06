
import os
import tempfile

from mars import oscar as mo


class FileLoggerActor(mo.Actor):
    mars_temp_log = "MARS_TEMP_LOG"

    def __init__(self):

        # if config is None:
        #     raise ValueError('Read yml config failed!')
        # cluster_config: dict = self.config.get('cluster')
        # if cluster_config is None:
        #     raise KeyError('\"cluster\" key is missing!')
        # log_dir = cluster_config.get("log_dir")
        # prefix = "mars_"
        # # default config, then create a temp file
        # if log_dir is None or log_dir == 'null':
        #     _, file_path = tempfile.mkstemp(prefix=prefix)
        # else:
        #     _, file_path = tempfile.mkstemp(prefix=prefix, dir=log_dir)
        file_path = os.environ.get(self.mars_temp_log)
        if file_path is None:
            raise ValueError('Env {0} is not set!'.format(self.mars_temp_log))
        self._log_filename = file_path

    def fetch_logs(self) -> str:
        content = self._get_n_bytes_tail_file(5 * 1024 * 1024)
        return content

    def _get_n_bytes_tail_file(self, bytes_num) -> str:
        f_size = os.stat(self._log_filename).st_size
        target = f_size - bytes_num if f_size > bytes_num else 0
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
