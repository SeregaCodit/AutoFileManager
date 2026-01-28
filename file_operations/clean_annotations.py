import argparse

from const_utils.arguments import Arguments
from const_utils.default_values import AppSettings
from const_utils.parser_help import HelpStrings
from file_operations.file_operation import FileOperation
from tests.test_file_operation import settings


class CleanAnnotationsOperation(FileOperation):

    @staticmethod
    def add_arguments(settings: AppSettings, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            Arguments.a_suffix,
            help=HelpStrings.a_suffix,
            default=settings.a_suffix,
        )
        parser.add_argument(
            Arguments.a_source,
            help=HelpStrings.a_source,
            default=settings.a_source,
        )


    def do_task(self) -> None:
        annotations = self.get_files(source_directory=self.settings.a_source, pattern=self.settings.a_suffix)
        stems = set(a.stem for a in annotations)
        existed_files = set(self.files_for_task)
        stems_for_remove = stems - existed_files

