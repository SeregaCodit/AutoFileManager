from dataclasses import dataclass
from const_utils.default_values import DefaultValues as defaults

@dataclass
class HelpStrings:
    move: str = "move file from source directory to target directory"
    src: str = "source directory"
    dst: str = "destination directory"
    pattern: str = r"Default - " + defaults.pattern + ". Do actions only with files that match pattern"
