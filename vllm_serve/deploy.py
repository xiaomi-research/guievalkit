import subprocess
import time
import itertools
import os
import signal

from vllm import SamplingParams
from utils import get_logger

# subsec internal
from vllm_serve.setup import ModelSetup
from vllm_serve.deployed import DeployedModel, ExtraInferenceSetup
from vllm_serve.utils.message_types import StrictMessages


class ModelDeploymentError(Exception):
    def __init__(self, msg: str):
        self._msg = msg

    def __repr__(self):
        return f'{self.__class__.__name__}({self._msg})'

    def __str__(self):
        return self._msg


class DeployModel:
    def __init__(self, setup: ModelSetup):
        self._setup = setup
        self._logger = get_logger(root_name='vLLM_serve',
                                  name=setup.model_alias,
                                  level='INFO',
                                  log_file=setup.deployment_log_path,
                                  always_ansi=False)
        # the instantiation of `DeployedModel` relies on a proper existing logger instance
        self._service = DeployedModel(setup=setup, logger=self._logger)

        self._deploy_proc: subprocess.Popen | None = None

    def _monitor(self, *,
                 steps: int | None = None):
        self._logger.info('Start monitoring service deployment...')
        counter = (itertools.count(start=1, step=1)
                   if steps is None else
                   range(steps))
        for i in counter:
            if self._service._is_alive(inform=(not (i % 7))):
                self._logger.info(f'Service has been online.\n'
                                  '<split_line>')
                return True
            elif self._deploy_proc is None:
                self._logger.info('No service deployment process has been initiated.')
            elif self._deploy_proc.poll() is not None:
                raise ModelDeploymentError('Service deployment failed. Check detailed log at:\n'
                                           f'\t{self._setup.log_path}')

            time.sleep(0.5)
        else:
            self._logger.info('Service currently offline.\n'
                              '<split_line>')
            return False

    def deploy(self, *,
               with_monitor: bool = True):
        if self._monitor(steps=1):
            return

        self._deploy_proc = subprocess.Popen([self._setup.serve_command, ], shell=True, preexec_fn=os.setsid,
                                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if with_monitor:
            self._monitor()

    def kill(self):
        os.killpg(os.getpgid(self._deploy_proc.pid), signal.SIGTERM)
        self._deploy_proc.wait()
        self._logger.info(f"Service safely killed. Exitcode: {self._deploy_proc.returncode}. ")

    def __call__(self,
                 messages: StrictMessages,
                 sampling_params: SamplingParams, *,
                 extra_params: ExtraInferenceSetup = ExtraInferenceSetup()) -> str:
        completion = self._service.chat(
            messages=messages,
            sampling_params=sampling_params,
            extra_params=extra_params
        )
        return completion.choices[0].message.content

    def __del__(self):
        self._logger.info(f"Service has been deprecated. Gonna kill it.")
        self.kill()
