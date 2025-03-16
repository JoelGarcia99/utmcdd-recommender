class Anime:
    def __init__(self, record):
        self.id = record['anime_id']
        self.name = record['name']
        self.genre = record['genre']
        self.type = record['type']
        self.episodes = record['episodes']
        self.rating = record['rating']
        self.members = record['members']

    def to_json(self):
        return {
            'id': self.id,
            'name': self.name,
            'genre': self.genre,
            'type': self.type,
            'episodes': self.episodes,
            'rating': self.rating,
            'members': self.members
        }
