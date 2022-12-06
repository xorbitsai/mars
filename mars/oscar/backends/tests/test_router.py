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

import pytest

from ....tests.core import mock
from ..communication import SocketClient, UnixSocketClient, DummyClient
from ..router import Router


@pytest.mark.asyncio
@mock.patch.object(Router, "_create_client")
async def test_router(fake_create_client):
    class FakeClient:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs
            self.closed = False

    fake_create_client.side_effect = FakeClient

    router = Router(
        external_addresses=["test"],
        local_address="dummy://1",
        mapping={
            "test": "unixsocket://local_test1",
            "test2": "unixsocket://local_test2",
            "test3": "unixsocket://local_test3",
        },
    )
    client, is_cache = await router.get_client("test2", return_from_cache=True)
    assert not is_cache
    client2, is_cache = await router.get_client("test2", return_from_cache=True)
    assert is_cache
    assert client2 is client
    # close fake client
    client.closed = True
    client3, is_cache = await router.get_client("test2", return_from_cache=True)
    assert not is_cache
    assert client3 is not client

    all_client_tyeps = router.get_all_client_types("test")
    assert set(all_client_tyeps) == {UnixSocketClient, SocketClient, DummyClient}

    client = await router.get_client_via_type("test", DummyClient)
    client2, is_cache = await router.get_client_via_type(
        "test", DummyClient, return_from_cache=True
    )
    assert client is client2
    assert is_cache
    # close client
    client.closed = True
    client3, is_cache = await router.get_client_via_type(
        "test", DummyClient, return_from_cache=True
    )
    assert not is_cache
    assert client is not client3
    client4 = await router.get_client_via_type("test", DummyClient)
    assert client3 is client4
    assert client3.args[-1].startswith("dummy://")
