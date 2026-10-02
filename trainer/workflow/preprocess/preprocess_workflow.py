import os
from typing import cast

from datasets import load_dataset
from torch.utils.data import DataLoader

from trainer.configs.configs import PreprocessConfig
from trainer.dataset.dataloader.schema import (pyarrow_schema_i2v,
                                                 pyarrow_schema_t2v)
from trainer.trainer_args import TrainerArgs, WorkloadType
from trainer.logger import init_logger
from trainer.pipelines.pipeline_registry import PipelineType
from trainer.workflow.preprocess.components import (
    ParquetDatasetSaver, PreprocessingDataValidator, VideoForwardBatchBuilder)
from trainer.workflow.preprocess.record_schema import (
    basic_t2v_record_creator, i2v_record_creator)
from trainer.workflow.workflow_base import WorkflowBase

logger = init_logger(__name__)


class PreprocessWorkflow(WorkflowBase):

    def register_pipelines(self) -> None:
        self.add_pipeline_config("preprocess_pipeline",
                                 (PipelineType.PREPROCESS, self.trainer_args))

    def register_components(self) -> None:
        assert self.trainer_args.preprocess_config is not None
        preprocess_config: PreprocessConfig = self.trainer_args.preprocess_config

        # raw data validator
        raw_data_validator = PreprocessingDataValidator(
            max_height=preprocess_config.max_height,
            max_width=preprocess_config.max_width,
            num_frames=preprocess_config.num_frames,
            train_fps=preprocess_config.train_fps,
            speed_factor=preprocess_config.speed_factor,
            video_length_tolerance_range=preprocess_config.
            video_length_tolerance_range,
            drop_short_ratio=preprocess_config.drop_short_ratio,
        )
        self.add_component("raw_data_validator", raw_data_validator)

        # training dataset
        training_dataset = load_dataset(preprocess_config.dataset_path,
                                        split="train")
        # set load_from_cache_file to False to check filter stats
        training_dataset = training_dataset.filter(raw_data_validator)
        # we do not use collate_fn here because we use iterable-style Dataset
        # and want to keep the original type of the dataset
        training_dataloader = DataLoader(
            training_dataset,
            batch_size=preprocess_config.preprocess_video_batch_size,
            num_workers=preprocess_config.dataloader_num_workers,
            collate_fn=lambda x: x,
        )
        self.add_component("training_dataloader", training_dataloader)

        # try to load validation dataset if it exists
        try:
            validation_dataset = load_dataset(preprocess_config.dataset_path,
                                              split="validation")
            validation_dataset = validation_dataset.filter(raw_data_validator)
            validation_dataloader = DataLoader(
                validation_dataset,
                batch_size=preprocess_config.preprocess_video_batch_size,
                num_workers=preprocess_config.dataloader_num_workers,
                collate_fn=lambda x: x,
            )
        except ValueError:
            logger.warning(
                "Validation dataset not found, skipping validation dataset preprocessing."
            )
            validation_dataloader = None

        self.add_component("validation_dataloader", validation_dataloader)

        # forward batch builder
        video_forward_batch_builder = VideoForwardBatchBuilder(
            seed=self.trainer_args.preprocess_config.seed)
        self.add_component("video_forward_batch_builder",
                           video_forward_batch_builder)

        # record creator
        if self.trainer_args.workload_type == WorkloadType.I2V:
            record_creator = i2v_record_creator
            schema_fields = [f.name for f in pyarrow_schema_i2v]
        else:
            record_creator = basic_t2v_record_creator
            schema_fields = [f.name for f in pyarrow_schema_t2v]
        processed_dataset_saver = ParquetDatasetSaver(
            flush_frequency=self.trainer_args.preprocess_config.
            flush_frequency,
            samples_per_file=self.trainer_args.preprocess_config.
            samples_per_file,
            schema_fields=schema_fields,
            record_creator=record_creator,
        )
        self.add_component("processed_dataset_saver", processed_dataset_saver)

    def prepare_system_environment(self) -> None:
        assert self.trainer_args.preprocess_config is not None
        dataset_output_dir = self.trainer_args.preprocess_config.dataset_output_dir
        os.makedirs(dataset_output_dir, exist_ok=True)

        validation_dataset_output_dir = os.path.join(dataset_output_dir,
                                                     "validation_dataset")
        os.makedirs(validation_dataset_output_dir, exist_ok=True)
        self.validation_dataset_output_dir = validation_dataset_output_dir

        training_dataset_output_dir = os.path.join(dataset_output_dir,
                                                   "training_dataset")
        os.makedirs(training_dataset_output_dir, exist_ok=True)
        self.training_dataset_output_dir = training_dataset_output_dir

    @classmethod
    def get_workflow_cls(cls,
                         trainer_args: TrainerArgs) -> "PreprocessWorkflow":
        if trainer_args.workload_type == WorkloadType.T2V or trainer_args.workload_type == WorkloadType.I2V:
            from trainer.workflow.preprocess.preprocess_workflow_t2v import (
                PreprocessWorkflowT2V)
            return cast(PreprocessWorkflow, PreprocessWorkflowT2V)
        else:
            raise ValueError(
                f"Workload type: {trainer_args.workload_type} is not supported in preprocessing workflow."
            )
