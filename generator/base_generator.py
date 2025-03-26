import time
from abc import ABC, abstractmethod
class BaseGenerator(ABC):
    def __init__(self, stream_period=0, timeout=30000):
        self.stream_period = stream_period
        self.timeout = timeout
        self.last_message_time = time.time()
        self._count = 0
    def __iter__(self):
        return self

    def __next__(self): # Python 2: def next(self)
        if self.stream_period > 0:
            if time.time() - self.last_message_time < self.stream_period / 1000:
                time.sleep(
                    self.stream_period / 1000
                    - (
                            (time.time() - self.last_message_time)
                            % (self.stream_period / 1000)
                    )
                )
        start_time = time.time()
        self.last_message_time = start_time
        try:
            return None
        except Exception as e:
            raise StopIteration

    def stop(self):
        """Function to be called when stream is finished."""
        pass


    @abstractmethod
    def get_message(self):
        """The function that contains the logic to generate a new message.
        It must return the message as an array.
        This function must be override by every custom stream.

        Raises:
            NotImplementedError: Abstract function has not been overrided.

        Returns:
            list: message
        """
        raise NotImplementedError("Abstract method")

    @abstractmethod
    def get_count(self):
        raise NotImplementedError("Abstract method")