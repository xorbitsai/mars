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
import shutil
import sys
import tempfile

import pytest

from .... import oscar as mo
from ..file_logger import FileLoggerActor

mars_temp_log = "MARS_TEMP_LOG"
prefix = "mars_"
mars_tmp_dir_prefix = "mars_tmp"
full_content = "qwert\nasdfg\nzxcvb\nyuiop\nhjkl;\nnm,./"


@pytest.fixture
async def actor_pool():
    pool = await mo.create_actor_pool("127.0.0.1", n_process=0)
    async with pool:
        yield pool

    # clean
    filename = os.environ.get(mars_temp_log)
    mars_tmp_dir = os.path.dirname(filename)
    # on windows platform, cannot delete this dir
    shutil.rmtree(mars_tmp_dir, ignore_errors=True)
    if not sys.platform.startswith("win"):
        assert not os.path.exists(mars_tmp_dir)
        assert not os.path.exists(filename)


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
    assert os.path.basename(filename).startswith("mars_")
    assert os.path.basename(os.path.dirname(filename)).startswith("mars_tmp")
    with open(filename, "w") as f:
        f.write(full_content)
        f.close()

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
