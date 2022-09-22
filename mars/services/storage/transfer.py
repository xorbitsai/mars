# Copyright 1999-2021 Alibaba Group Holding Ltd.
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
import logging
from dataclasses import dataclass
from typing import Dict, List, Union, Tuple, Set

from ... import oscar as mo
from ...lib.aio import alru_cache
from ...storage import StorageLevel
from ...utils import dataslots
from .core import DataManagerActor, WrappedStorageFileObject
from .handler import StorageHandlerActor

DEFAULT_TRANSFER_BLOCK_SIZE = 4 * 1024**2


logger = logging.getLogger(__name__)


class SenderManagerActor(mo.StatelessActor):
    def __init__(
        self,
        band_name: str = "numa-0",
        transfer_block_size: int = None,
        data_manager_ref: mo.ActorRefType[DataManagerActor] = None,
        storage_handler_ref: mo.ActorRefType[StorageHandlerActor] = None,
    ):
        self._band_name = band_name
        self._data_manager_ref = data_manager_ref
        self._storage_handler = storage_handler_ref
        self._transfer_block_size = transfer_block_size or DEFAULT_TRANSFER_BLOCK_SIZE

    @classmethod
    def gen_uid(cls, band_name: str):
        return f"sender_manager_{band_name}"

    async def __post_create__(self):
        if self._storage_handler is None:  # for test
            self._storage_handler = await mo.actor_ref(
                self.address, StorageHandlerActor.gen_uid("numa-0")
            )

    @staticmethod
    @alru_cache
    async def get_receiver_ref(address: str, band_name: str):
        return await mo.actor_ref(
            address=address, uid=ReceiverManagerActor.gen_uid(band_name)
        )

    async def _send_data(
        self,
        receiver_ref: mo.ActorRefType["ReceiverManagerActor"],
        session_id: str,
        data_keys: List[str],
        infos: List,
        block_size: int,
    ):
        class BufferedSender:
            def __init__(self):
                self._buffers = []
                self._send_keys = []
                self._eof_marks = []

            async def flush(self):
                if self._buffers:
                    await receiver_ref.receive_part_data(
                        self._buffers, session_id, self._send_keys, self._eof_marks
                    )

                self._buffers = []
                self._send_keys = []
                self._eof_marks = []

            async def send(self, buffer, eof_mark, key):
                self._eof_marks.append(eof_mark)
                self._buffers.append(buffer)
                self._send_keys.append(key)
                if sum(len(b) for b in self._buffers) >= block_size:
                    await self.flush()

        sender = BufferedSender()
        open_reader_tasks = []
        for data_key in data_keys:
            open_reader_tasks.append(
                self._storage_handler.open_reader.delay(session_id, data_key)
            )
        readers = await self._storage_handler.open_reader.batch(*open_reader_tasks)

        class AsyncFixedReader:
            def __init__(self, file_obj, fixed_size: int):
                self._file_obj = file_obj
                self._cur = 0
                self._size = fixed_size
                self._end = self._cur + self._size

            def _get_size(self, size):
                max_size = self._end - self._cur
                if size is None:
                    return max_size
                else:
                    return min(max_size, size)

            async def read(self, size=None):
                read_size = self._get_size(size)
                result = await self._file_obj.read(read_size)
                self._cur += read_size
                return result

        for data_key, reader, info in zip(data_keys, readers, infos):
            if info.offset:
                await reader.seek(info.offset)
                reader = AsyncFixedReader(reader, info.store_size)
            while True:
                part_data = await reader.read(block_size)
                # Notes on [How to decide whether the reader reaches EOF?]
                #
                # In some storage backend, e.g., the reported memory usage (i.e., the
                # `store_size`) may not same with the byte size that need to be transferred
                # when moving to a remote worker. Thus, we think the reader reaches EOF
                # when a `read` request returns nothing, rather than comparing the `sent_size`
                # and the `store_size`.
                #
                is_eof = not part_data  # can be non-empty bytes, empty bytes and None
                await sender.send(part_data, is_eof, data_key)
                if is_eof:
                    break
        await sender.flush()

    async def _send_mapper_data(
        self,
        receiver_ref: mo.ActorRefType["ReceiverManagerActor"],
        session_id: str,
        shuffle_main_keys: List,
        data_keys: List[Union[str, Tuple]],
        infos: List,
        block_size: int,
    ):
        for main_key in shuffle_main_keys:
            await receiver_ref.wait_writer_created(main_key)
        await self._send_data(receiver_ref, session_id, data_keys, infos, block_size)

    @mo.extensible
    async def send_batch_data(
        self,
        session_id: str,
        data_keys: List[str],
        address: str,
        level: StorageLevel,
        band_name: str = "numa-0",
        block_size: int = None,
        error: str = "raise",
    ):
        logger.debug(
            "Begin to send data (%s, %s) to %s", session_id, data_keys, address
        )
        block_size = block_size or self._transfer_block_size
        receiver_ref: mo.ActorRefType[
            ReceiverManagerActor
        ] = await self.get_receiver_ref(address, band_name)
        get_infos = []
        pin_tasks = []
        for data_key in data_keys:
            get_infos.append(
                self._data_manager_ref.get_data_info.delay(
                    session_id, data_key, self._band_name, error
                )
            )
            pin_tasks.append(
                self._data_manager_ref.pin.delay(
                    session_id, data_key, self._band_name, error
                )
            )
        await self._data_manager_ref.pin.batch(*pin_tasks)
        infos = await self._data_manager_ref.get_data_info.batch(*get_infos)
        filtered = []
        shuffle_keys = []
        for data_info, data_key in zip(infos, data_keys):
            if data_info is not None:
                filtered.append((data_info, data_key))
                if isinstance(data_key, tuple):
                    shuffle_keys.append(data_key)
        if filtered:
            infos, data_keys = zip(*filtered)
        else:  # pragma: no cover
            # no data to be transferred
            return
        data_sizes = [info.store_size for info in infos]
        if level is None:
            level = infos[0].level
        is_transferring_list, shuffle_main_keys = await receiver_ref.open_writers(
            session_id, data_keys, data_sizes, level
        )
        to_send_keys = []
        to_wait_keys = []
        for data_key, is_transferring in zip(data_keys, is_transferring_list):
            if is_transferring:
                to_wait_keys.append(data_key)
            else:
                to_send_keys.append(data_key)

        send_tasks = []
        if to_send_keys:
            send_tasks.append(
                self._send_data(
                    receiver_ref, session_id, to_send_keys, infos, block_size
                )
            )
        if to_wait_keys:
            send_tasks.append(receiver_ref.wait_transfer_done(session_id, to_wait_keys))
        if shuffle_keys and shuffle_main_keys:
            send_tasks.append(
                self._send_mapper_data(
                    receiver_ref,
                    session_id,
                    list(shuffle_main_keys),
                    shuffle_keys,
                    infos,
                    block_size,
                )
            )
        await asyncio.gather(*send_tasks)
        unpin_tasks = []
        for data_key in data_keys:
            unpin_tasks.append(
                self._data_manager_ref.unpin.delay(
                    session_id, [data_key], self._band_name, error="ignore"
                )
            )
        await self._data_manager_ref.unpin.batch(*unpin_tasks)
        logger.debug(
            "Finish sending data (%s, %s) to %s, total size is %s",
            session_id,
            data_keys,
            address,
            sum(data_sizes),
        )


@dataslots
@dataclass
class WritingInfo:
    writer: WrappedStorageFileObject
    size: int
    level: StorageLevel
    event: asyncio.Event
    ref_counts: int


@dataslots
@dataclass
class ShuffleWritingInfo(WritingInfo):
    lock: asyncio.Lock
    offsets: Dict
    finished_keys: Set


class MapperRecorder:
    def __init__(self, expected_mappers: Set):
        self._expected_mappers = expected_mappers
        self._mapper_info = dict()
        self._total_size = 0

    def append(self, data_key: Union[str, Tuple], data_size: int) -> bool:
        # append data_key, if all keys are ready, return True
        self._mapper_info[data_key] = (self._total_size, data_size)
        self._total_size += data_size
        if set(self._mapper_info) == self._expected_mappers:
            return True
        return False

    @property
    def mapper_info(self):
        return self._mapper_info


@dataslots
@dataclass
class ShuffleInfo:
    mapper_recorder: MapperRecorder
    event: asyncio.Event


class ReceiverManagerActor(mo.StatelessActor):
    def __init__(
        self,
        quota_refs: Dict,
        storage_handler_ref: mo.ActorRefType[StorageHandlerActor] = None,
    ):
        self._quota_refs = quota_refs
        self._storage_handler = storage_handler_ref
        self._writing_infos: Dict[Union[str, Tuple], WritingInfo] = dict()
        self._sub_to_main_key: Dict[Tuple, str] = dict()
        self._shuffle_infos: Dict[str, ShuffleInfo] = dict()
        self._shuffle_writing_status = dict()
        self._lock = asyncio.Lock()

    async def __post_create__(self):
        if self._storage_handler is None:  # for test
            self._storage_handler = await mo.actor_ref(
                self.address, StorageHandlerActor.gen_uid("numa-0")
            )

    @classmethod
    def gen_uid(cls, band_name: str):
        return f"receiver_manager_{band_name}"

    def _decref_writing_key(self, session_id: str, data_key: str):
        self._writing_infos[(session_id, data_key)].ref_counts -= 1
        if self._writing_infos[(session_id, data_key)].ref_counts == 0:
            del self._writing_infos[(session_id, data_key)]

    async def create_writers(
        self,
        session_id: str,
        data_keys: List[Union[str, Tuple]],
        data_sizes: List[int],
        level: StorageLevel,
    ):
        tasks = dict()
        data_key_to_size = dict()
        being_processed = []
        shuffle_main_keys = set()
        for data_key, data_size in zip(data_keys, data_sizes):
            data_key_to_size[data_key] = data_size
            if (session_id, data_key) not in self._writing_infos:
                if isinstance(data_key, tuple):
                    # won't open writer for shuffle data instantly.
                    main_key = self._sub_to_main_key[data_key]
                    shuffle_main_keys.add(main_key)
                    is_last = self._shuffle_infos[main_key].mapper_recorder.append(
                        data_key, data_size
                    )
                    if is_last:
                        # last mapper arrives, open the writer
                        recorder = self._shuffle_infos[main_key].mapper_recorder
                        data_keys = list(recorder.mapper_info.keys())
                        sizes = [info[1] for info in recorder.mapper_info.values()]
                        writer = await self._storage_handler.open_writer(
                            session_id, data_keys, sizes, level, True
                        )
                        writer.set_main_key(main_key)
                        self._shuffle_infos[main_key].event.set()
                        offsets = dict(
                            (key, offset)
                            for key, (offset, _) in self._shuffle_infos[
                                main_key
                            ].mapper_recorder.mapper_info.items()
                        )
                        self._writing_infos[
                            (session_id, main_key)
                        ] = ShuffleWritingInfo(
                            writer,
                            data_size,
                            level,
                            asyncio.Event(),
                            1,
                            asyncio.Lock(),
                            offsets,
                            set(),
                        )
                else:
                    being_processed.append(False)
                    tasks[data_key] = self._storage_handler.open_writer.delay(
                        session_id, data_key, data_size, level
                    )
            else:
                being_processed.append(True)
                self._writing_infos[(session_id, data_key)].ref_counts += 1
        if tasks:
            writers = await self._storage_handler.open_writer.batch(
                *tuple(tasks.values())
            )
            for data_key, writer in zip(tasks, writers):
                self._writing_infos[(session_id, data_key)] = WritingInfo(
                    writer, data_key_to_size[data_key], level, asyncio.Event(), 1
                )
        return being_processed, shuffle_main_keys

    async def open_writers(
        self,
        session_id: str,
        data_keys: List[str],
        data_sizes: List[int],
        level: StorageLevel,
    ):
        async with self._lock:
            future = asyncio.create_task(
                self.create_writers(session_id, data_keys, data_sizes, level)
            )
            try:
                return await future
            except asyncio.CancelledError:
                future.cancel()
                raise

    async def do_write(
        self,
        data_list: list,
        session_id: str,
        data_keys: List[str],
        eof_marks: List[bool],
    ):
        # close may be a high-cost operation, use create_task
        finished_keys = []
        for data, data_key, is_eof in zip(data_list, data_keys, eof_marks):
            if isinstance(data_key, tuple):
                # shuffle data
                main_key = self._sub_to_main_key[data_key]
                writer_info = self._writing_infos[(session_id, main_key)]
                writer = writer_info.writer
                if data:
                    async with writer_info.lock:
                        offset = writer_info.offsets[data_key]
                        await writer.seek(offset)
                        await writer.write(data)
                        writer_info.offsets[data_key] += offset

                if is_eof:
                    writer_info.finished_keys.add(data_key)
                    if writer_info.finished_keys == set(writer_info.offsets):
                        await writer.close()
                        finished_keys.append(main_key)

            else:
                writer_info = self._writing_infos[(session_id, data_key)]
                writer = writer_info.writer
                if data:
                    await writer.write(data)
                if is_eof:
                    await writer.close()
                    finished_keys.append(data_key)
        async with self._lock:
            for data_key in finished_keys:
                event = self._writing_infos[(session_id, data_key)].event
                event.set()
                self._decref_writing_key(session_id, data_key)

    async def receive_part_data(
        self, data: list, session_id: str, data_keys: List[str], eof_marks: List[bool]
    ):
        write_task = asyncio.create_task(
            self.do_write(data, session_id, data_keys, eof_marks)
        )
        try:
            await asyncio.shield(write_task)
        except asyncio.CancelledError:
            async with self._lock:
                for data_key in data_keys:
                    if (session_id, data_key) in self._writing_infos:
                        if self._writing_infos[(session_id, data_key)].ref_counts == 1:
                            info = self._writing_infos[(session_id, data_key)]
                            await self._quota_refs[info.level].release_quota(info.size)
                            await self._storage_handler.delete(
                                session_id, data_key, error="ignore"
                            )
                            await info.writer.clean_up()
                            info.event.set()
                            self._decref_writing_key(session_id, data_key)
                            write_task.cancel()
                            await write_task
            raise

    async def wait_transfer_done(self, session_id, data_keys):
        await asyncio.gather(
            *[self._writing_infos[(session_id, key)].event.wait() for key in data_keys]
        )
        async with self._lock:
            for data_key in data_keys:
                self._decref_writing_key(session_id, data_key)

    def register_shuffle_task(self, main_key: str, mapper_keys: List):
        self._shuffle_infos[main_key] = ShuffleInfo(
            mapper_recorder=MapperRecorder(set(mapper_keys)), event=asyncio.Event()
        )
        for mapper_key in mapper_keys:
            self._sub_to_main_key[mapper_key] = main_key

    async def wait_writer_created(self, main_key: str):
        await self._shuffle_infos[main_key].event.wait()
