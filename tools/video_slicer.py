from pathlib import Path
import cv2

class VideoSlicer:
    """A class to cut a video into many images.

    This class takes a video file and saves its frames as separate image files in a folder.
    """
    def __init__(self):
        """Initializes the VideoSlicer.

        It sets the initial state of the slicer.
        """
        self.__sliced: bool = False


    def slice(self, source_file: Path, target_dir: Path, suffix: str = ".jpg", step: float = 1) -> tuple:
        """Cuts the video into images and saves them to a folder.

        Args:
            source_file (Path): The path to the video file you want to cut.
            target_dir (Path): The folder where you want to save the images.
            suffix (str): The file extension for the images (for example, '.jpg'). Defaults to '.jpg'.
            step (float): How many seconds to wait between saving images. Defaults to 1.

        Returns:
            tuple: A tuple containing:
                - bool: True if the video was sliced successfully, False otherwise.
                - int: The total number of images saved.
        """
        cap = cv2.VideoCapture(str(source_file))

        if not cap.isOpened():
            return self.sliced, 0

        fps = cap.get(cv2.CAP_PROP_FPS)
        step_frames = int(fps * step)
        img_counter = 0
        frame_id = 0

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            if frame_id % step_frames == 0:
                new_filename = f"{source_file.stem}_{img_counter}{suffix}"
                file_path = target_dir / new_filename
                cv2.imwrite(str(file_path), frame)
                img_counter += 1

            frame_id += 1

        cap.release()
        self.__sliced = True
        return self.sliced, img_counter


    @property
    def sliced(self) -> bool:
        """Checks if the video has been sliced.

        Returns:
            bool: True if the slice method was called and finished, False otherwise.
        """
        return self.__sliced
