from models.database_image import DatabaseImage


class Match:
    matches_count: int
    image: DatabaseImage

    def __init__(self, matches_count,img):
        self.matches_count = matches_count
        self.image = img