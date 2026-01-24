import argparse

from const_utils.arguments import Arguments
from const_utils.copmarer import Constants
from const_utils.default_values import DefaultValues
from const_utils.parser_help import HelpStrings
from file_operations.file_operation import FileOperation
from tools.comparer.img_comparer.img_comparer import ImageComparer


class DedupOperation(FileOperation):
    """

    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mapping = {
            Constants.image: ImageComparer
        }

        self.filetype = kwargs.get("filetype", DefaultValues.image)
        self.action = kwargs.get("action", DefaultValues.action)
        self.method = kwargs.get("method", DefaultValues.dhash)
        self.comparer: ImageComparer = self.mapping[self.filetype](
            method_name=self.method,
            log_path = self.log_path,
            threshold_percentage=float(kwargs.get("threshold", DefaultValues.hash_threshold))
        )

    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            Arguments.threshold,
            help=HelpStrings.threshold,
            default=DefaultValues.hash_threshold
        )
        parser.add_argument(
            Arguments.filetype,
            help=HelpStrings.filetype,
            default=DefaultValues.image
        )
        parser.add_argument(
            Arguments.method, Arguments.m,
            help=HelpStrings.method,
            default=DefaultValues.dhash
        )
        parser.add_argument(
            Arguments.action, Arguments.a,
            help=HelpStrings.action,
            default=DefaultValues.action
        )

    def do_task(self):
        duplicates = self.comparer.compare(self.files_for_task)
        self.logger.info(f"Found {len(duplicates)} duplicates in {len(self.files_for_task)} files")
        #TODO: допишу завтра
        pass





