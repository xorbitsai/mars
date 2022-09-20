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

import pytest

from ..file_logger import FileLoggerActor
from .... import oscar as mo
from ....utils import clean_mars_tmp_dir, get_mars_log_env_keys

mars_temp_log, prefix, mars_tmp_dir_prefix = get_mars_log_env_keys()
full_content = "qwert\nasdfg\nzxcvb\nyuiop\nhjkl;\nnm,./"


@pytest.fixture
async def actor_pool():
    pool = await mo.create_actor_pool("127.0.0.1", n_process=0)
    async with pool:
        yield pool

    # clean
    clean_mars_tmp_dir()


@pytest.mark.asyncio
async def test_file_logger_with_env(actor_pool, caplog):
    # prepare
    mars_tmp_dir = tempfile.mkdtemp(prefix=mars_tmp_dir_prefix)
    _, file_path = tempfile.mkstemp(prefix=prefix, dir=mars_tmp_dir)
    os.environ[mars_temp_log] = file_path

    pool_addr = actor_pool.external_address
    _ = await mo.create_actor(
        FileLoggerActor,
        uid=FileLoggerActor.default_uid(),
        address=pool_addr,
    )
    assert len(caplog.text) == 0


@pytest.mark.asyncio
async def test_file_logger_without_env(actor_pool, caplog):
    pool_addr = actor_pool.external_address
    with caplog.at_level(logging.WARN):
        logger_ref = await mo.create_actor(
            FileLoggerActor,
            uid=FileLoggerActor.default_uid(),
            address=pool_addr,
        )

    filename = os.environ.get(mars_temp_log)
    assert filename is not None
    assert os.path.exists(filename)
    assert os.path.basename(filename).startswith(prefix)
    assert os.path.basename(os.path.dirname(filename)).startswith(mars_tmp_dir_prefix)
    with open(filename, "w", newline="\n") as f:
        f.write(full_content)

    byte_num = 5
    expected_data = ""
    content = await logger_ref.fetch_logs(byte_num)
    assert content == expected_data

    byte_num = 6
    expected_data = "nm,./"
    content = await logger_ref.fetch_logs(byte_num)
    assert content == expected_data

    byte_num = 11
    expected_data = "nm,./"
    content = await logger_ref.fetch_logs(byte_num)
    assert content == expected_data

    byte_num = 12
    expected_data = "hjkl;\nnm,./"
    content = await logger_ref.fetch_logs(byte_num)
    assert content == expected_data

    byte_num = 50
    expected_data = "qwert\nasdfg\nzxcvb\nyuiop\nhjkl;\nnm,./"
    content = await logger_ref.fetch_logs(byte_num)
    assert content == expected_data

    byte_num = -1
    expected_data = "qwert\nasdfg\nzxcvb\nyuiop\nhjkl;\nnm,./"
    content = await logger_ref.fetch_logs(byte_num)
    assert content == expected_data
