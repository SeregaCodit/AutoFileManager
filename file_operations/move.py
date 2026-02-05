import argparse
import shutil

from const_utils.arguments import Arguments
from const_utils.default_values import AppSettings
from const_utils.parser_help import HelpStrings
from file_operations.file_operation import FileOperation

class MoveOperation(FileOperation):
    """
    An operation to move files from a source directory to a target directory.

    This class identifies files that match specific patterns and relocates them
    to the destination directory. It includes a safety check to prevent moving a file
    into the same directory where it is already located.
    """
    @staticmethod
    def add_arguments(settings: AppSettings, parser: argparse.ArgumentParser) -> None:
        """
        Adds the destination argument to the command-line interface.

        Args:
            settings (AppSettings): The global settings object for default values.
            parser (argparse.ArgumentParser): The CLI parser to which arguments are added.
        """
        parser.add_argument(Arguments.dst, help=HelpStrings.dst)


    def do_task(self):
        """
        Iterates through the collected files and moves them to the target directory.

        The method checks if each item is a file and ensures the target path is
        different from the source path. It uses 'shutil.move' for the operation
        and logs the results or any errors that occur.
        """
        for file_path in self.files_for_task:
            if file_path.is_file() and file_path.parent.resolve() != self.target_directory.resolve():
                target_file_path = self.target_directory / file_path.name
                self.logger.info(f"{file_path} -> {self.target_directory}")

                try:
                    shutil.move(file_path, target_file_path)
                except Exception as e:
                    self.logger.error(e)
