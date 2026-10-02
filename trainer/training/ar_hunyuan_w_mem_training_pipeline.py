# SPDX-License-Identifier: Apache-2.0
from copy import deepcopy
import os
import sys
sys.path.append(os.path.abspath('.'))

from trainer.trainer_args import TrainerArgs, TrainingArgs
from trainer.logger import init_logger
from trainer.training.ar_hunyuan_mem_training_pipeline import TrainingPipeline
from trainer.utils import is_vsa_available

vsa_available = is_vsa_available()

logger = init_logger(__name__)


class HunyuanTrainingPipeline(TrainingPipeline):
    """
    A training pipeline for Hunyuan.
    """
    _required_config_modules = ["transformer"]

    def initialize_pipeline(self, trainer_args: TrainerArgs):
        pass
        # self.modules["scheduler"] = FlowUniPCMultistepScheduler(
        #     shift=trainer_args.pipeline_config.flow_shift)

    def create_training_stages(self, training_args: TrainingArgs):
        """
        May be used in future refactors.
        """
        pass

    def initialize_validation_pipeline(self, training_args: TrainingArgs):
        pass


def main(args) -> None:
    logger.info("Starting training pipeline...")

    pipeline = HunyuanTrainingPipeline.from_pretrained(
        args.pretrained_model_name_or_path, args=args)
    args = pipeline.training_args
    pipeline.train()
    logger.info("Training pipeline done")


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)

    argv = sys.argv
    from trainer.trainer_args import TrainingArgs
    from trainer.utils import FlexibleArgumentParser
    parser = FlexibleArgumentParser()
    parser = TrainingArgs.add_cli_args(parser)
    parser = TrainerArgs.add_cli_args(parser)
    args = parser.parse_args()
    args.dit_cpu_offload = False
    main(args)
