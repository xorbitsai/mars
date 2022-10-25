# Copyright 2022 XProbe Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import os
import tempfile

from ... import oscar as mo
from ...constants import MARS_LOG_PATH_KEY, MARS_LOG_PREFIX, MARS_TMP_DIR_PREFIX

logger = logging.getLogger(__name__)


class FileLoggerActor(mo.Actor):
    """
    Read log file path from env (source from yaml config) for each node (including supervisor and all the workers).
    Expose interface for web frontend to fetch log content.
    """

    def __init__(self):
        file_path = os.environ.get(MARS_LOG_PATH_KEY)
        # other situations: start cluster not from cmdline
        if file_path is None:
            logger.debug("Env {0} is not set!".format(MARS_LOG_PATH_KEY))
            mars_tmp_dir = tempfile.mkdtemp(prefix=MARS_TMP_DIR_PREFIX)
            _, file_path = tempfile.mkstemp(prefix=MARS_LOG_PREFIX, dir=mars_tmp_dir)
            os.environ[MARS_LOG_PATH_KEY] = file_path
        self._log_filename = file_path

    def fetch_logs(self, size: int) -> str:
        """
        Externally exposed interface.

        Parameters
        ----------
        size

        Returns
        -------

        """

        content = self._get_n_bytes_tail_file(size)
        return content

    def _get_n_bytes_tail_file(self, bytes_num: int) -> str:
        """
        Read last n bytes of file.

        Parameters
        ----------
        bytes_num: the bytes to read. -1 means read the whole file.

        Returns
        -------

        """
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
