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

import os
import shutil
import sys
import tempfile
from typing import List

import numpy as np
import pytest

from .....lib.aio import AioFileObject
from .....tests.core import require_cupy
from .....utils import lazy_import, convert_to_cupy_ndarray
from .... import (
    Actor,
    BufferRef,
    buffer_ref,
    FileObjectRef,
    file_object_ref,
    ActorRefType,
    actor_ref,
    copyto_via_buffers,
    copyto_via_file_objects,
)
from ....context import get_context
from ...allocate_strategy import ProcessIndex
from ...pool import create_actor_pool
from ..pool import MainActorPool


rmm = lazy_import("rmm")
cupy = lazy_import("cupy")
ucp = lazy_import("ucp")


class BufferTransferActor(Actor):
    def __init__(self):
        self._buffers = []

    def create_buffers(self, sizes: List[int], cpu: bool = True) -> List[BufferRef]:
        if cpu:
            buffers = [np.empty(size, dtype="u1").data for size in sizes]
        else:
            buffers = [
                convert_to_cupy_ndarray(rmm.DeviceBuffer(size=size)) for size in sizes
            ]
        self._buffers.extend(buffers)
        return [buffer_ref(self.address, buf) for buf in buffers]

    def create_arrays_from_buffer_refs(
        self, buf_refs: List[BufferRef], cpu: bool = True
    ):
        if cpu:
            return [
                np.frombuffer(BufferRef.get_buffer(ref), dtype="u1") for ref in buf_refs
            ]
        else:
            return [
                convert_to_cupy_ndarray(BufferRef.get_buffer(ref)) for ref in buf_refs
            ]

    async def copy_data(
        self, ref: ActorRefType["BufferTransferActor"], sizes, cpu: bool = True
    ):
        xp = np if cpu else cupy
        arrays = [
            np.random.randint(2, dtype=bool, size=size).astype("u1") for size in sizes
        ]
        buffers = [a.data for a in arrays]
        if not cpu:
            arrays = [cupy.asarray(a) for a in arrays]
            buffers = arrays

        ref = await actor_ref(ref)
        buf_refs = await ref.create_buffers(sizes, cpu=cpu)
        await copyto_via_buffers(buffers, buf_refs)
        new_arrays = await ref.create_arrays_from_buffer_refs(buf_refs, cpu=cpu)
        assert len(arrays) == len(new_arrays)
        for a1, a2 in zip(arrays, new_arrays):
            xp.testing.assert_array_equal(a1, a2)


async def _copy_test(scheme: str, cpu: bool):
    start_method = (
        os.environ.get("POOL_START_METHOD", "forkserver")
        if sys.platform != "win32"
        else None
    )
    pool = await create_actor_pool(
        "127.0.0.1",
        pool_cls=MainActorPool,
        n_process=2,
        subprocess_start_method=start_method,
        external_address_schemes=[None, scheme, scheme],
    )

    async with pool:
        ctx = get_context()

        # actor on main pool
        actor_ref1 = await ctx.create_actor(
            BufferTransferActor,
            uid="test-1",
            address=pool.external_address,
            allocate_strategy=ProcessIndex(1),
        )
        actor_ref2 = await ctx.create_actor(
            BufferTransferActor,
            uid="test-2",
            address=pool.external_address,
            allocate_strategy=ProcessIndex(2),
        )
        sizes = [10 * 1024**2, 3 * 1024**2]
        await actor_ref1.copy_data(actor_ref2, sizes, cpu=cpu)


schemes = [None]
if ucp is not None:
    schemes.append("ucx")


@pytest.mark.asyncio
@pytest.mark.parametrize("scheme", schemes)
async def test_copy(scheme):
    await _copy_test(scheme, True)


@require_cupy
@pytest.mark.parametrize("scheme", schemes)
async def tests_gpu_copy(scheme):
    await _copy_test(scheme, False)


class FileobjTransferActor(Actor):
    def __init__(self):
        self._fileobjs = []

    async def create_file_objects(self, names: List[str]) -> List[FileObjectRef]:
        refs = []
        for name in names:
            fobj = open(name, "w+b")
            afobj = AioFileObject(fobj)
            self._fileobjs.append(afobj)
            refs.append(file_object_ref(self.address, afobj))
        return refs

    async def close(self):
        for fobj in self._fileobjs:
            assert await fobj.tell() > 0
            await fobj.close()

    async def copy_data(
        self,
        ref: ActorRefType["FileobjTransferActor"],
        names1: List[str],
        names2: List[str],
        sizes: List[int],
    ):
        fobjs = []
        for name, size in zip(names1, sizes):
            fobj = open(name, "w+b")
            fobj.write(np.random.bytes(size))
            fobj.seek(0)
            fobjs.append(AioFileObject(fobj))

        ref = await actor_ref(ref)
        file_obj_refs = await ref.create_file_objects(names2)
        await copyto_via_file_objects(fobjs, file_obj_refs)
        _ = [await f.close() for f in fobjs]
        await ref.close()

        for n1, n2 in zip(names1, names2):
            with open(n1, "rb") as f1, open(n2, "rb") as f2:
                b1 = f1.read()
                b2 = f2.read()
                assert b1 == b2


@pytest.mark.asyncio
async def test_copy_via_fileobjects():
    start_method = (
        os.environ.get("POOL_START_METHOD", "forkserver")
        if sys.platform != "win32"
        else None
    )
    pool = await create_actor_pool(
        "127.0.0.1",
        pool_cls=MainActorPool,
        n_process=2,
        subprocess_start_method=start_method,
    )

    d = tempfile.mkdtemp()
    async with pool:
        ctx = get_context()

        # actor on main pool
        actor_ref1 = await ctx.create_actor(
            FileobjTransferActor,
            uid="test-1",
            address=pool.external_address,
            allocate_strategy=ProcessIndex(1),
        )
        actor_ref2 = await ctx.create_actor(
            FileobjTransferActor,
            uid="test-2",
            address=pool.external_address,
            allocate_strategy=ProcessIndex(2),
        )
        sizes = [10 * 1024**2, 3 * 1024**2]
        names = []
        for _ in range(2 * len(sizes)):
            _, p = tempfile.mkstemp(dir=d)
            names.append(p)

        await actor_ref1.copy_data(actor_ref2, names[::2], names[1::2], sizes=sizes)
    try:
        shutil.rmtree(d)
    except PermissionError:
        pass
