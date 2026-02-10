import argparse
from typing import Union, Dict

from const_utils.arguments import Arguments
from const_utils.default_values import AppSettings
from const_utils.parser_help import HelpStrings
from file_operations.file_operation import FileOperation
from tools.stats.base_stats import BaseStats
from tools.stats.voc_stats import VOCStats
from tools.stats.yolo_stats import YoloStats


class StatsOperation(FileOperation):
    """
    This class show dataset information such as number of files object
    distribution, areas distribution etc. It can be used to get
    insights about the dataset before training a model.
    It can also be used to identify potential issues with the dataset,
    such as class or base features imbalance.
    """

    def __init__(self, settings: AppSettings, **kwargs):
        """Initializes the StatsOperation with settings and specific arguments."""
        super().__init__(settings, **kwargs)
        self.target_format: Union[str, None] = kwargs.get('target_format', self.settings.destination_type)
        self.stats_mapping: Dict[str, BaseStats.__subclasses__()] = {
            "yolo": YoloStats,
            "voc": VOCStats
        }
        self.stats_method: Union[BaseStats.__subclasses__()] = self.stats_mapping[self.target_format]()

    @staticmethod
    def add_arguments(settings: AppSettings, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            Arguments.destination_type,
            help=HelpStrings.destination_type,
        )
        parser.add_argument(
            Arguments.img_path,
            help=HelpStrings.img_path,
            default=None,
        )
        parser.add_argument(
            Arguments.n_jobs,
            help=HelpStrings.n_jobs,
            default=settings.n_jobs
        )

    def do_task(self):
        """
        This method should implement the logic to calculate and display the statistics of the dataset.
         It can include:
            - Counting the number of files in each class.
            - Calculating the distribution of object areas, positions etc.
            - Identifying any class imbalance issues.
            - Providing insights about the dataset that can help in model training and evaluation.
        """
        self.logger.info(f"Found {len(self.files_for_task)} annotations in {self.src}")
        pass