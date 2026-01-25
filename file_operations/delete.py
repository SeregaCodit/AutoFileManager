import argparse

from file_operations.file_operation import FileOperation
from file_operations.file_remover import FileRemoverMixin


class DeleteOperation(FileOperation, FileRemoverMixin):
    """Delete files that match the pattern"""
    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser) -> None:
        """this class doesn't have unique arguments"""
        pass

    def do_task(self):
        """Delete files that match the pattern"""
        self._remove_all(self.files_for_task)
