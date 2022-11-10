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
import configparser
import logging
import os
import tempfile
from typing import Dict, List

from ... import oscar as mo
from ...constants import MARS_TMP_DIR_PREFIX, MARS_LOG_PREFIX, MARS_LOG_PATH_KEY
from ...resource import cuda_count, Resource

logger = logging.getLogger(__name__)


def _need_suspend_sigint() -> bool:
    try:
        from IPython import get_ipython

        return get_ipython() is not None
    except ImportError:
        return False


def _parse_file_logging_config(
    file_path: str,
    log_path: str,
    level: str,
    formatter: str = None,
    from_cmd: bool = False,
) -> configparser.RawConfigParser:
    config = configparser.RawConfigParser()
    config.read(file_path)
    logger_sections = [
        "logger_main",
        "logger_deploy",
        "logger_oscar",
        "logger_services",
        "logger_dataframe",
        "logger_learn",
        "logger_tensor",
        "handler_stream_handler",
        "handler_file_handler",
    ]
    all_sections = config.sections()
    for section in logger_sections:
        if section in all_sections:
            config[section]["level"] = level.upper() if level else "INFO"

    config["handler_file_handler"]["args"] = fr"('{log_path}',)"
    if formatter:
        format_section = "formatter_formatter"
        config[format_section]["format"] = formatter

    stream_handler_sec = "handler_stream_handler"
    # If not from cmd (like ipython) and user uses its own config file,
    # need to judge that whether handler_stream_handler section is in the config.
    if not from_cmd and stream_handler_sec in all_sections:
        # console log keeps the default level and formatter as before
        # file log on the web uses info level and the formatter in the config file
        config[stream_handler_sec]["level"] = "WARN"
        config[stream_handler_sec].pop("formatter")
    return config


def _config_logging(**kwargs):
    web: bool = kwargs.get("web", True)
    # web=False usually means it is a test environment.
    if not web:
        return
    if 'logging_conf' not in kwargs:
        return
    config = kwargs['logging_conf']
    from_cmd = config.get("from_cmd", False)
    log_dir = config.get("log_dir", None)
    log_conf_file = config.get("file", None)
    level = config.get("level", None)
    formatter = config.get("formatter", None)
    logging_config_path = log_conf_file or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "file-logging.conf"
    )
    # default config, then create a temp file
    if (os.environ.get(MARS_LOG_PATH_KEY, None)) is None:
        if log_dir is None:
            mars_tmp_dir = tempfile.mkdtemp(prefix=MARS_TMP_DIR_PREFIX)
        else:
            mars_tmp_dir = os.path.join(log_dir, MARS_TMP_DIR_PREFIX)
            os.makedirs(mars_tmp_dir, exist_ok=True)
        _, file_path = tempfile.mkstemp(prefix=MARS_LOG_PREFIX, dir=mars_tmp_dir)
        os.environ[MARS_LOG_PATH_KEY] = file_path
        logging_conf = _parse_file_logging_config(
            logging_config_path, file_path, level, formatter, from_cmd
        )
        # bind user's level and format when using default log conf
        logging.config.fileConfig(
            logging_conf,
            disable_existing_loggers=False,
        )
        logger.debug("Use logging config file at %s", logging_config_path)
        return logging_conf
    else:
        logging_conf = _parse_file_logging_config(
            logging_config_path,
            os.environ[MARS_LOG_PATH_KEY],
            level,
            formatter,
            from_cmd,
        )
        logging.config.fileConfig(
            logging_conf,
            os.environ[MARS_LOG_PATH_KEY],
            disable_existing_loggers=False,
        )
        logger.debug("Use logging config file at %s", logging_config_path)
        return logging_conf


async def create_supervisor_actor_pool(
    address: str,
    n_process: int,
    modules: List[str] = None,
    ports: List[int] = None,
    subprocess_start_method: str = None,
    **kwargs,
):
    logging_conf = _config_logging(**kwargs)
    kwargs["logging_conf"] = logging_conf
    return await mo.create_actor_pool(
        address,
        n_process=n_process,
        ports=ports,
        modules=modules,
        subprocess_start_method=subprocess_start_method,
        suspend_sigint=_need_suspend_sigint(),
        **kwargs,
    )


async def create_worker_actor_pool(
    address: str,
    band_to_resource: Dict[str, Resource],
    n_io_process: int = 1,
    modules: List[str] = None,
    ports: List[int] = None,
    cuda_devices: List[int] = None,
    subprocess_start_method: str = None,
    **kwargs,
):
    logging_conf = _config_logging(**kwargs)
    kwargs["logging_conf"] = logging_conf
    # TODO: support NUMA when ready
    n_process = sum(
        int(resource.num_cpus) or int(resource.num_gpus)
        for resource in band_to_resource.values()
    )
    envs = []
    labels = ["main"]

    if cuda_devices is None:  # pragma: no cover
        env_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        if not env_devices:
            cuda_devices = list(range(cuda_count()))
        else:
            cuda_devices = [int(i) for i in env_devices.split(",")]

    i_gpu = iter(sorted(cuda_devices))
    for band, resource in band_to_resource.items():
        if band.startswith("gpu"):  # pragma: no cover
            idx = str(next(i_gpu))
            envs.append({"CUDA_VISIBLE_DEVICES": idx})
            labels.append(f"gpu-{idx}")
        else:
            assert band.startswith("numa")
            num_cpus = int(resource.num_cpus)
            envs.extend([dict() for _ in range(num_cpus)])
            labels.extend([band] * num_cpus)

    return await mo.create_actor_pool(
        address,
        n_process=n_process,
        ports=ports,
        n_io_process=n_io_process,
        labels=labels,
        envs=envs,
        modules=modules,
        subprocess_start_method=subprocess_start_method,
        suspend_sigint=_need_suspend_sigint(),
        **kwargs,
    )
