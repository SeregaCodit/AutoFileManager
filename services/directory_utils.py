import os
from pathlib import Path
from typing import Union



def generate_directory_name(src: Union[Path, str]) -> Path:

    if not isinstance(src, Path):
        try:
            src = Path(src)
        except TypeError:
            raise TypeError(f"src must be a Path or str object, got {type(src)}")

    existing_directories = [p for p in os.listdir(src) if (src / p).is_dir()]
    dir_name = int(sorted(existing_directories)[-1]) + 1
    report_path = src / str(dir_name)
    return report_path.resolve()