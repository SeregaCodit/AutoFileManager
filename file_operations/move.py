import shutil
from pathlib import Path

from file_operations.file_operation import FileOperation


class MoveOperation(FileOperation):
    def run(self):
        source_directory = Path(self.src)
        target_directory = Path(self.dst)

        # Перевірка, чи існує джерело
        if not source_directory.exists():
            print(f"[ERROR] Source path '{self.src}' does not exist.")
            return

        target_directory.mkdir(parents=True, exist_ok=True)

        # Використовуємо rglob, якщо треба шукати і в підпапках, або glob для поточної
        for file_path in source_directory.glob(f"*{self.pattern}*"):
            # Переміщуємо тільки файли, ігноруємо папку призначення, якщо вона всередині джерела
            if file_path.is_file() and file_path.parent != target_directory:
                target_file_path = target_directory / file_path.name
                print(f"Moving: {file_path.name} -> {target_directory}", end=" ")

                try:
                    # shutil.move приймає Path об'єкти напряму
                    shutil.move(file_path, target_file_path)
                    print("[OK]")
                except Exception as e:
                    print(f"[ERROR] {e}")