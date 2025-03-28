"""Implements a wraper for river datasets."""

from generator.base_generator import BaseGenerator


class ListDatasetGenerator(BaseGenerator):

    def __init__(self, dataset, stream_period=0, timeout=30000,  n_instances=1000, **kwargs):
        """
        Args:
            stream_period (int): Delay between two consecutive messages, in ms.
            timeout (int): (Optional) Not used in this example, but included for completeness.
        """
        super().__init__(stream_period=stream_period, timeout=timeout)
        self.n_instances = n_instances
        self._iterator = iter(dataset)

    def __next__(self):
        """
        We override __next__ so that it:
          1. Respects the timing logic from the BaseGenerator (sleep if necessary).
          2. Calls get_message to fetch the next data point.
        """
        # Step 1: respect timing logic from the base class
        super().__next__()

        # Step 2: fetch the next message (x, y)
        return self.get_message()

    def get_message(self):
        """
        Retrieves the next item (x, y) from the Bikes dataset iterator.
        If the dataset is exhausted, it calls stop() and raises StopIteration.
        """
        try:
            item = next(self._iterator)
            # Check if 'item' is a single integer (or anything that's not a tuple).
            # If so, wrap it into a (x, None) tuple or whatever makes sense for your case.
            if isinstance(item, int):
                x, y = item, None
            else:
                # Otherwise assume it's already something like (x, y)
                x, y = item
            self._count += 1
            return x, y
        except StopIteration:
            self.stop()     # Optionally perform any cleanup here
            raise

    def get_count(self):
        return self._count