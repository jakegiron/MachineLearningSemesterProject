import datetime as dt


class Timer:

    def __init__(self):
        self.start_dt = None

    def start(self):
        self.start_dt = dt.datetime.now()

    def stop(self):
        return dt.datetime.now() - self.start_dt
