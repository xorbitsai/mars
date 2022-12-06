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

import asyncio
import multiprocessing
import os
from typing import Type

import numpy as np

from mars.lib.aio import AioEvent
from mars.lib.nvutils import get_device_count
from mars.oscar.backends.communication import (
    Channel,
    Server,
    SocketServer,
    UCXServer,
    UnixSocketServer,
)
from mars.utils import lazy_import, get_next_port, Timer, readable_size


cupy = lazy_import("cupy")
ucp = lazy_import("ucp")


def send_back_data(
    server_started_event: multiprocessing.Event,
    conf: dict,
    type_: Type[Server],
    env: dict,
):
    os.environ.update(env)

    async def arun():
        async def run(chan: Channel):
            while True:
                # receive & write back
                data = await chan.recv()
                if data is None:
                    break
                await chan.send(data)

        conf["handle_channel"] = run

        # create server
        server = await type_.create(conf)
        await server.start()
        server_started_event.set()
        await server.join()

    asyncio.run(arun())


def transfer_data(
    server_type: Type[Server],
    data_size: int = 256 * 1024**2,
    gpu: bool = False,
    config: dict = None,
):
    raw_env = os.environ.copy()
    if gpu:
        # limit main process to device 0
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    async def arun():
        xp = cupy if gpu else np
        arr = np.random.randint(2, size=data_size, dtype=bool)
        if gpu:
            arr = cupy.asarray(arr)

        mp_ctx = multiprocessing.get_context("forkserver")
        server_started = mp_ctx.Event()
        port = get_next_port()
        conf = config or dict()
        if server_type is UnixSocketServer:
            process_index = int(np.random.randint(30000))
            conf.update(dict(process_index=process_index))
            addr = f"uixsocket:///{process_index}"
        else:
            conf.update(dict(host="127.0.0.1", port=port))
            addr = f"127.0.0.1:{port}"
        env = dict(os.environ)
        if gpu:
            # limit subprocess to device 1
            env["CUDA_VISIBLE_DEVICES"] = "1"
        p = mp_ctx.Process(
            target=send_back_data,
            args=(server_started, conf, server_type, env),
        )
        p.daemon = True
        p.start()

        await AioEvent(server_started).wait()

        # create client
        client = await server_type.client_type.connect(addr)

        # warmup
        small_arr = arr[:3]
        await client.channel.send(small_arr)
        xp.testing.assert_array_equal(small_arr, await client.channel.recv())

        with Timer() as timer:
            await client.channel.send(arr)
            arr2 = await client.channel.recv()

        # stop
        await client.channel.send(None)

        xp.testing.assert_array_equal(arr, arr2)

        print(
            f'Transfer {"CPU" if not gpu else "GPU"} data by {server_type.scheme or "socket"} '
            f"at {readable_size(data_size * 2 / timer.duration)}B/s"
        )
        return timer.duration

    try:
        return asyncio.run(arun())
    finally:
        os.environ = raw_env


class MultiprocessChannelSuite:
    """
    Benchmark that times performance of channel.
    """

    def time_socket_cpu(self):
        transfer_data(SocketServer)

    def time_unixsocket_cpu(self):
        transfer_data(UnixSocketServer)


class MultiprocessUCXChannelsuite:
    """
    Benchmark that times performance of channel of UCX.
    """

    def setup(self):
        if ucp is None:
            raise NotImplementedError

    def time_ucx_cpu(self):
        transfer_data(UCXServer)


class MultiprocessGPUChannelSuite:
    def setup(self):
        if (get_device_count() or 0) < 2:
            # skip benmark when n_gpu < 2
            raise NotImplementedError

    def time_socket_gpu(self):
        transfer_data(SocketServer, gpu=True)

    def time_unixsocket_gpu(self):
        transfer_data(UnixSocketServer, gpu=True)

    def time_ucx_gpu(self):
        transfer_data(UCXServer, gpu=True)


if __name__ == "__main__":
    s = MultiprocessChannelSuite()
    s.time_socket_cpu()
    s.time_unixsocket_cpu()
    if ucp is not None:
        s2 = MultiprocessUCXChannelsuite()
        s2.time_ucx_cpu()
    s3 = MultiprocessGPUChannelSuite()
    try:
        s3.setup()
        s3.time_socket_gpu()
        s3.time_unixsocket_gpu()
        if cupy is not None:
            s3.time_ucx_gpu()
    except NotImplementedError:
        pass
