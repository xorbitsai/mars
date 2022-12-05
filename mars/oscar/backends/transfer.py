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
import contextlib
import sys
from typing import List, Union

from ...lib.aio import AioFileObject
from ..core import BufferRef, FileObjectRef
from .communication import Client, Channel
from .message import (
    CopytoBuffersMessage,
    CopytoFileObjectsMessage,
    new_message_id,
    MessageType,
    ResultMessage,
    ErrorMessage,
    ControlMessage,
    ControlMessageType,
)
from .router import Router


DEFAULT_TRANSFER_BLOCK_SIZE = 4 * 1024**2


def _get_buffer_size(buf) -> int:
    try:
        return buf.nbytes
    except AttributeError:  # pragma: no cover
        return len(buf)


class TransferClient:
    def __init__(self):
        self._lock = asyncio.Lock()

    @staticmethod
    def _handle_ack(message: Union[ResultMessage, ErrorMessage]):
        if message.message_type == MessageType.result:
            assert message.result
        else:
            assert message.message_type == MessageType.error
            raise message.error.with_traceback(message.traceback)

    async def copyto_via_buffers(
        self, local_buffers: list, remote_buffer_refs: List[BufferRef]
    ):
        assert (
            len({ref.address for ref in remote_buffer_refs}) == 1
        ), "remote buffers for `copy_via_buffers` can support only 1 destination"
        assert len(local_buffers) == len(remote_buffer_refs), (
            f"Buffers from local and remote must have same size, "
            f"local: {len(local_buffers)}, remote: {len(remote_buffer_refs)}"
        )

        router = Router.get_instance()
        assert router is not None, "`copy_via_buffers` can only be used inside pools"
        address = remote_buffer_refs[0].address
        client_types = router.get_all_client_types(address)
        message = CopytoBuffersMessage(
            message_id=new_message_id(), buffer_refs=remote_buffer_refs
        )
        try:
            # use the client that supports buffer copy
            client_type = next(
                client_type
                for client_type in client_types
                if hasattr(client_type, "send_buffers")
            )
        except StopIteration:
            # do not support buffer copy
            # send data in batches
            client, is_cached = await router.get_client(
                address, from_who=self, return_from_cache=True
            )
            if not is_cached:
                # tell server to switch to transfer dedicated channel
                await client.send(self._gen_switch_to_transfer_control_message())

            async with self._lock:
                await client.send(message)
                self._handle_ack(await client.recv())
                await self._send_buffers_in_batches(local_buffers, client)
        else:
            client, is_cached = await router.get_client_via_type(
                address, client_type, from_who=self, return_from_cache=True
            )
            if not is_cached:
                # tell server to switch to transfer dedicated channel
                await client.send(self._gen_switch_to_transfer_control_message())

            async with self._lock:
                await client.send(message)
                self._handle_ack(await client.recv())
                await client.send_buffers(local_buffers)

    @staticmethod
    def _gen_switch_to_transfer_control_message():
        return ControlMessage(
            message_id=new_message_id(),
            control_message_type=ControlMessageType.switch_to_transfer,
        )

    @classmethod
    async def _send_buffers_in_batches(cls, local_buffers: list, client: Client):
        for buffer in local_buffers:
            i = 0
            while True:
                curr_buf = buffer[
                    i
                    * DEFAULT_TRANSFER_BLOCK_SIZE : (i + 1)
                    * DEFAULT_TRANSFER_BLOCK_SIZE
                ]
                size = _get_buffer_size(curr_buf)
                if size == 0:
                    break
                await client.send(curr_buf)
                # ack
                message = await client.recv()
                cls._handle_ack(message)
                i += 1

    async def copyto_via_file_objects(
        self,
        local_file_objects: List[AioFileObject],
        remote_file_object_refs: List[FileObjectRef],
    ):
        assert (
            len({ref.address for ref in remote_file_object_refs}) == 1
        ), "remote file objects for `copyto_via_file_objects` can support only 1 destination"

        router = Router.get_instance()
        assert (
            router is not None
        ), "`copyto_via_file_objects` can only be used inside pools"
        address = remote_file_object_refs[0].address
        client, is_cached = await router.get_client(
            address, from_who=self, return_from_cache=True
        )
        if not is_cached:
            # tell server to switch to transfer dedicated channel
            await client.send(self._gen_switch_to_transfer_control_message())

        message = CopytoFileObjectsMessage(
            message_id=new_message_id(), fileobj_refs=remote_file_object_refs
        )
        async with self._lock:
            await client.send(message)
            for fileobj in local_file_objects:
                finished = False
                while not finished:
                    buf = await fileobj.read(DEFAULT_TRANSFER_BLOCK_SIZE)
                    size = _get_buffer_size(buf)
                    if size > 0:
                        await client.send(buf)
                        # ack
                        message = await client.recv()
                        self._handle_ack(message)
                    else:
                        await client.send(None)
                        # ack
                        message = await client.recv()
                        self._handle_ack(message)
                        finished = True


@contextlib.asynccontextmanager
async def _catch_error(channel: Channel, message_id: bytes):
    try:
        yield
        await channel.send(ResultMessage(message_id=message_id, result=True))
    except:  # noqa: E722  # nosec  # pylint: disable=bare-except  # pragma: no cover
        et, err, tb = sys.exc_info()
        await channel.send(
            ErrorMessage(
                message_id=message_id,
                error_type=et,
                error=err,
                traceback=tb,
            )
        )
        raise


class TransferServer:
    @classmethod
    async def handle_transfer_channel(cls, channel: Channel, stopped: asyncio.Event):
        while not stopped.is_set():
            try:
                message = await channel.recv()
            except EOFError:  # pragma: no cover
                # no data to read, check channel
                try:
                    await channel.close()
                except (ConnectionError, EOFError):
                    # close failed, ignore
                    pass
                return
            assert message.message_type in (
                MessageType.copyto_buffers,
                MessageType.copyto_fileobjects,
            )
            await cls._process_message(message, channel)

    @classmethod
    async def _process_message(
        cls,
        message: Union[CopytoBuffersMessage, CopytoFileObjectsMessage],
        channel: Channel,
    ):
        if isinstance(message, CopytoBuffersMessage):
            async with _catch_error(channel, message.message_id):
                buffers = [
                    BufferRef.get_buffer(buffer_ref)
                    for buffer_ref in message.buffer_refs
                ]
            if hasattr(channel, "recv_buffers"):
                await channel.recv_buffers(buffers)
            else:
                await cls._recv_buffers_in_batches(message, buffers, channel)
        else:
            assert isinstance(message, CopytoFileObjectsMessage)
            async with _catch_error(channel, message.message_id):
                file_objects = [
                    FileObjectRef.get_file_object(ref) for ref in message.fileobj_refs
                ]
            await cls._recv_file_objects(message, file_objects, channel)

    @classmethod
    async def _recv_buffers_in_batches(
        cls, message: CopytoBuffersMessage, buffers: list, channel: Channel
    ):
        for buffer in buffers:
            size = _get_buffer_size(buffer)
            acc = 0
            while True:
                async with _catch_error(channel, message.message_id):
                    recv_buffer = await channel.recv()
                    cur_size = _get_buffer_size(recv_buffer)
                    buffer[acc : acc + cur_size] = recv_buffer
                acc += cur_size
                if acc >= size:
                    break

    @classmethod
    async def _recv_file_objects(
        cls,
        message: CopytoFileObjectsMessage,
        file_objects: List[AioFileObject],
        channel: Channel,
    ):
        for fileobj in file_objects:
            finished = False
            while not finished:
                recv_buffer = await channel.recv()
                if recv_buffer is not None:
                    # not finished, receive part data
                    async with _catch_error(channel, message.message_id):
                        await fileobj.write(recv_buffer)
                else:
                    # done, send ack
                    await channel.send(
                        ResultMessage(message_id=message.message_id, result=True)
                    )
                    finished = True
