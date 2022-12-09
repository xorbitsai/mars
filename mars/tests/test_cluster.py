# Copyright 1999-2022 Alibaba Group Holding Ltd.
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
import os
import sys
import subprocess
import tempfile

import psutil
import pytest

from .. import new_session
from .. import tensor as mt
from .. import remote as mr
from ..services.cluster import NodeRole, WebClusterAPI
from ..tests.core import flaky, lazy_import
from ..utils import get_next_port

ucp = lazy_import("ucp")


CONFIG_CONTENT = """\
"@inherits": "@mars/config.yml"
scheduling:
  mem_hard_limit: null"""

UCX_CONFIG_FILE = f"""
{CONFIG_CONTENT}
oscar:
  numa:
    external_addr_scheme: ucx
"""

configs = {
    "default": CONFIG_CONTENT,
}
if ucp is not None:
    configs["ucx"] = UCX_CONFIG_FILE


def _terminate(pid: int):
    proc = psutil.Process(pid)
    sub_pids = [p.pid for p in proc.children(recursive=True)]
    proc.terminate()
    proc.wait(5)
    for p in sub_pids:
        try:
            proc = psutil.Process(p)
            proc.kill()
        except psutil.NoSuchProcess:
            continue


@flaky(max_runs=3)
@pytest.mark.asyncio
@pytest.mark.parametrize("name", list(configs))
async def test_cluster(name):
    port = get_next_port()
    web_port = get_next_port()
    supervisor_addr = f"127.0.0.1:{port}"
    web_addr = f"http://127.0.0.1:{web_port}"

    # gen config file
    fd, path = tempfile.mkstemp()
    with os.fdopen(fd, mode="w") as f:
        f.write(configs[name])

    r = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "mars.supervisor",
            "-H",
            "127.0.0.1",
            "-p",
            str(port),
            "-w",
            str(web_port),
            "-f",
            path,
        ],
        stderr=subprocess.PIPE,
    )
    w = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "mars.worker",
            "-s",
            supervisor_addr,
            "--cuda-devices",
            "-1",
            "-f",
            path,
        ]
    )

    for p in [r, w]:
        try:
            retcode = p.wait(1)
        except subprocess.TimeoutExpired:
            # supervisor & worker will run forever,
            # timeout means everything goes well, at least looks well,
            continue
        else:
            if retcode:
                std_err = r.communicate()[1].decode()
                p.kill()
                raise RuntimeError("Start cluster failed, stderr: \n" + std_err)

    def _check(timeout: float = 1.0):
        for p in [r, w]:
            try:
                retcode = p.wait(timeout)
            except subprocess.TimeoutExpired:
                # supervisor & worker will run forever,
                # timeout means everything goes well, at least looks well,
                continue
            else:
                if retcode:
                    std_err = r.communicate()[1].decode()
                    p.kill()
                    raise RuntimeError("Start cluster failed, stderr: \n" + std_err)

    try:
        cluster_api = WebClusterAPI(web_addr)
        while True:
            try:
                jsn = await cluster_api.get_nodes_info(role=NodeRole.WORKER)
            except ConnectionError:
                await asyncio.to_thread(_check, timeout=0.5)
                continue
            if not jsn:
                await asyncio.sleep(0.5)
                continue
            if len(jsn) > 0:
                break

        def f():
            from ..oscar.backends.router import Router

            mapping = Router.get_instance()._mapping
            if name == "ucx":
                assert all(a.startswith("ucx") for a in mapping)
            else:
                assert all(not a.startswith("ucx") for a in mapping)

        sess = new_session(web_addr, default=True)
        a = mt.arange(10)
        assert a.sum().to_numpy(show_progress=False) == 45

        # no error should be raised
        mr.spawn(f).execute()

        sess2 = new_session(web_addr, session_id=sess.session_id)
        sess2.close()
    finally:
        _terminate(w.pid)
        _terminate(r.pid)

    # test stderr
    out = r.communicate()[1].decode()
    saddr = supervisor_addr if name == "default" else f"{name}://{supervisor_addr}"
    assert f"Supervisor started at {saddr}, web address: {web_addr}" in out
