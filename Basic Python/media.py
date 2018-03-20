import webbrowser

class Movie():
    # create __doc__
    """This class provides a way to store movie related information."""

    VALID_RATING = ["G", "PG", "PG13", "R"]     # Class Variable, using all cap

    def __init__(self, movie_title, movie_storyline, poster_image,
        trailer_youtube):     # Constructor
        self.title = movie_title
        self.storyline = movie_storyline
        self.poster_image_url = poster_image
        self.trailer_youtube_url = trailer_youtube

    def show_trailer(self):
        webbrowser.open(self.trailer_youtube_url)
