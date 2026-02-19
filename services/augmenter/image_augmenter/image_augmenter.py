from abc import ABC, abstractmethod

from services.augmenter.base_augmenter import BaseAugmenter


class ImageAugmenter(BaseAugmenter, ABC):

    def __init__(self, settings):
        super().__init__(settings)
        pass

