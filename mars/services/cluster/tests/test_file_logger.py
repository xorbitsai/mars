import os
import tempfile

import pytest

from .... import oscar as mo
from ..file_logger import FileLoggerActor

mars_temp_log = "MARS_TEMP_LOG"
full_content = "qwert\nasdfg\nzxcvb\nyuiop\nhjkl;\nnm,./"


@pytest.fixture
async def actor_pool():
    pool = await mo.create_actor_pool("127.0.0.1", n_process=0)
    async with pool:
        yield pool


@pytest.fixture
def prepare_file(actor_pool):
    _, filename = tempfile.mkstemp()
    with open(filename, 'w') as f:
        f.write(full_content)
        f.close()
    os.environ[mars_temp_log] = filename
    return actor_pool.external_address, filename


@pytest.mark.asyncio
async def test_file_logger_with_no_env(actor_pool):
    pool_addr = actor_pool.external_address
    match = 'Env {0} is not set!'.format(mars_temp_log)
    with pytest.raises(ValueError, match=match):
        await mo.create_actor(
            FileLoggerActor,
            uid=FileLoggerActor.default_uid(),
            address=pool_addr,
        )

    with pytest.raises(ValueError, match=match):
        # wrong env key
        os.environ["MARS_LOG"] = "1234.txt"
        await mo.create_actor(
            FileLoggerActor,
            uid=FileLoggerActor.default_uid(),
            address=pool_addr,
        )


@pytest.mark.asyncio
async def test_file_logger(prepare_file):
    pool_addr, filename = prepare_file
    logger_ref = await mo.create_actor(
        FileLoggerActor,
        uid=FileLoggerActor.default_uid(),
        address=pool_addr,
    )

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

    assert os.path.exists(filename)
    await logger_ref.destroy()
    assert not os.path.exists(filename)
