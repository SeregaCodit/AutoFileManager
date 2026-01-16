from dataclasses import dataclass

@dataclass
class Arguments:
    src: str = "src"
    dst: str = "dst"
    pattern: str = "--pattern"
    p: str = "-p"