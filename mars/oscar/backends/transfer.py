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
import itertools
import sys
import weakref
from abc import ABC, abstractmethod
from typing import List, Union, Any

from ...lib.aio import AioFileObject
from ...utils import is_cuda_buffer
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


def _handle_ack(message: Union[ResultMessage, ErrorMessage]):
    if message.message_type == MessageType.result:
        assert message.result
    else:  # pragma: no cover
        assert message.message_type == MessageType.error
        raise message.error.with_traceback(message.traceback)


class _ParallelSender(ABC):
    _message_id_to_futures: weakref.WeakValueDictionary

    def __init__(
        self,
        client: Client,
        local_objs: list,
        remote_obj_refs: List[Union[BufferRef, FileObjectRef]],
    ):
        self.client = client
        self.local_objs = local_objs
        self.remote_obj_refs = remote_obj_refs

        self._message_id_to_futures = weakref.WeakValueDictionary()
        self._n_send = [0] * len(local_objs)

    @staticmethod
    @abstractmethod
    def _new_message(ref: Union[BufferRef, FileObjectRef], buf: Any):
        """new message"""

    @abstractmethod
    async def _read(self, index: int, buffer_or_fileobj: Any):
        """read data"""

    async def _send_one(
        self,
        index: int,
        buffer_or_fileobj: Any,
        remote_ref: Union[BufferRef, FileObjectRef],
    ):
        while True:
            part_buf = await self._read(index, buffer_or_fileobj)
            size = _get_buffer_size(part_buf)
            if size == 0:
                break
            fut = asyncio.get_running_loop().create_future()
            message = self._new_message(remote_ref, part_buf)
            self._message_id_to_futures[message.message_id] = fut
            await self.client.send(message)
            self._n_send[index] += size
            await fut

    async def _recv_ack_in_background(self):
        while True:
            ack_message = await self.client.recv()
            if ack_message is None:
                # receive finished
                break
            fut: asyncio.Future = self._message_id_to_futures[ack_message.message_id]
            try:
                _handle_ack(ack_message)
            except BaseException as e:  # pragma: no cover  # noqa: E722  # nosec  # pylint: disable=bare-except
                fut.set_exception(e)
            else:
                fut.set_result(ack_message.result)

    async def start(self):
        recv_task = asyncio.create_task(self._recv_ack_in_background())
        tasks = []
        for i, local_obj, remote_obj_ref in zip(
            itertools.count(0), self.local_objs, self.remote_obj_refs
        ):
            tasks.append(self._send_one(i, local_obj, remote_obj_ref))
        try:
            await asyncio.gather(*tasks)
        finally:
            # all send finished, send a None to receiver
            await self.client.send(None)
            await recv_task


class _ParallelReceiver(ABC):
    def __init__(
        self,
        channel: Channel,
        local_objs: list,
        remote_obj_refs: List[Union[BufferRef, FileObjectRef]],
    ):
        self.channel = channel
        self.local_objs = local_objs
        self.remote_obj_refs = remote_obj_refs

        self._n_recv = [0] * len(local_objs)
        self._ref_to_i = {ref: i for i, ref in enumerate(remote_obj_refs)}

    @abstractmethod
    async def _write(self, index: int, buffer_or_fileobj: Any):
        """write data"""

    @staticmethod
    @staticmethod
    def _get_ref_from_message(
        message: Union[CopytoBuffersMessage, CopytoFileObjectsMessage]
    ):
        """get ref according to message"""

    async def _recv_part(self, buf: Any, index: int, message_id: bytes):
        async with _catch_error(self.channel, message_id):
            await self._write(index, buf)
            self._n_recv[index] += _get_buffer_size(buf)

    async def start(self):
        tasks = []
        while True:
            message = await self.channel.recv()
            if message is None:
                # send finished
                break
            message_id = message.message_id
            buf = message.content
            ref = self._get_ref_from_message(message)[0]
            i = self._ref_to_i[ref]
            tasks.append(asyncio.create_task(self._recv_part(buf, i, message_id)))
        try:
            await asyncio.gather(*tasks)
        finally:
            # when all done, send a None to finish client
            await self.channel.send(None)


class _BufferSender(_ParallelSender):
    def __init__(
        self, client: Client, local_buffers: list, remote_buffer_refs: List[BufferRef]
    ):
        super().__init__(client, local_buffers, remote_buffer_refs)

    @staticmethod
    def _new_message(ref: Union[BufferRef, FileObjectRef], buf: Any):
        return CopytoBuffersMessage(
            message_id=new_message_id(), buffer_refs=[ref], content=buf
        )

    async def _read(self, index: int, buffer_or_fileobj: Any):
        size = self._n_send[index]
        return buffer_or_fileobj[size : size + DEFAULT_TRANSFER_BLOCK_SIZE]


class _BufferReceiver(_ParallelReceiver):
    def __init__(self, channel: Channel, buffers: list, buffer_refs: List[BufferRef]):
        super().__init__(channel, buffers, buffer_refs)

    async def _write(self, index: int, buffer_or_fileobj: Any):
        full_buf = self.local_objs[index]
        size = _get_buffer_size(buffer_or_fileobj)
        n_recv = self._n_recv[index]

        def copy():
            full_buf[n_recv : n_recv + size] = buffer_or_fileobj

        await asyncio.to_thread(copy)

    @staticmethod
    def _get_ref_from_message(
        message: Union[CopytoBuffersMessage, CopytoFileObjectsMessage]
    ):
        return message.buffer_refs


class _FileObjectSender(_ParallelSender):
    def __init__(
        self,
        client: Client,
        local_file_objects: list,
        remote_file_object_refs: List[FileObjectRef],
    ):
        super().__init__(client, local_file_objects, remote_file_object_refs)

    @staticmethod
    def _new_message(ref: Union[BufferRef, FileObjectRef], buf: Any):
        return CopytoFileObjectsMessage(
            message_id=new_message_id(), fileobj_refs=[ref], content=buf
        )

    async def _read(self, index: int, buffer_or_fileobj: Any):
        return await buffer_or_fileobj.read(DEFAULT_TRANSFER_BLOCK_SIZE)


class _FileObjectReceiver(_ParallelReceiver):
    def __init__(
        self,
        channel: Channel,
        file_objects: list,
        file_object_refs: List[FileObjectRef],
    ):
        super().__init__(channel, file_objects, file_object_refs)

    async def _write(self, index: int, buffer_or_fileobj: Any):
        fileobj = self.local_objs[index]
        await fileobj.write(buffer_or_fileobj)

    @staticmethod
    def _get_ref_from_message(
        message: Union[CopytoBuffersMessage, CopytoFileObjectsMessage]
    ):
        return message.fileobj_refs


class TransferClient:
    def __init__(self):
        self._lock = asyncio.Lock()

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
                _handle_ack(await client.recv())
                await self._send_buffers_in_batches(
                    local_buffers, remote_buffer_refs, client
                )
        else:
            client, is_cached = await router.get_client_via_type(
                address, client_type, from_who=self, return_from_cache=True
            )
            if not is_cached:
                # tell server to switch to transfer dedicated channel
                await client.send(self._gen_switch_to_transfer_control_message())

            async with self._lock:
                await client.send(message)
                _handle_ack(await client.recv())
                await client.send_buffers(local_buffers)

    @staticmethod
    def _gen_switch_to_transfer_control_message():
        return ControlMessage(
            message_id=new_message_id(),
            control_message_type=ControlMessageType.switch_to_transfer,
        )

    @classmethod
    async def _send_buffers_in_batches(
        cls, local_buffers: list, remote_buffer_refs: List[BufferRef], client: Client
    ):
        sender = _BufferSender(client, local_buffers, remote_buffer_refs)
        await sender.start()

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
            _handle_ack(await client.recv())
            sender = _FileObjectSender(
                client, local_file_objects, remote_file_object_refs
            )
            await sender.start()


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
        buffer_refs = message.buffer_refs
        receiver = _BufferReceiver(channel, buffers, buffer_refs)
        await receiver.start()

    @classmethod
    async def _recv_file_objects(
        cls,
        message: CopytoFileObjectsMessage,
        file_objects: List[AioFileObject],
        channel: Channel,
    ):
        file_object_refs = message.fileobj_refs
        receiver = _FileObjectReceiver(channel, file_objects, file_object_refs)
        await receiver.start()
