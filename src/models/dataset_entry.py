from typing import List

from models.database_image import DatabaseImage


class DataSetEntry:
    dataset_img_abs_path: str
    matching_db_entries: List[DatabaseImage]

    def __init__(self, dataset_img_abs_path, matching_db_entries) -> None:
        self.dataset_img_abs_path = dataset_img_abs_path
        self.matching_db_entries = matching_db_entries