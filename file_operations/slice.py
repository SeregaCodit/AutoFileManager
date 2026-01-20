from file_operations.file_operation import FileOperation


class SliceOperation(FileOperation):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step_sec: float = kwargs.get('step_sec', 1)
        self.img_type: str = kwargs.get('img_type', '.jpg')


    def do_task(self):
        for file_path in self.files_for_task:
            if file_path.is_file():
                img_counter = 0
                target_file_path = self.target_directory / file_path.name.split(".")[-2] + "_" + str(img_counter) + ".jpg"
                print(f"Slicing {file_path.name}")
