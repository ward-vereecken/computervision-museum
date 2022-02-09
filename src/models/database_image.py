# Shared class for working with images in dataset (info and descriptors)
from typing import Any

class DatabaseImage:
    room: int
    og_path: str
    paintingNmbr: int
    absolute_img_path: str
    descriptors: Any
    keypoints: Any

    def __init__(self, room, og_path, paintingNmbr, absolute_img_path, descriptors, keypoints) -> None:
        self.room = room
        self.og_path = og_path
        self.paintingNmbr = paintingNmbr
        self.absolute_img_path = absolute_img_path
        self.descriptors = descriptors
        self.keypoints = keypoints