from generator.list_generator import ListDatasetGenerator
from generator.river_dataset_generator import RiverDatasetGenerator


class MovingWindowRiverGenerator(RiverDatasetGenerator):
    """
    Moving window stream generator for river datasets.

    This generator extends RiverDatasetGenerator and applies a moving window
    preprocessing method for time series forecasting. It assumes that each
    message produced by the base generator is a tuple (x, y) and uses the first
    element (x) as the raw time series message.
    """

    def __init__(
            self,
            dataset,
            past_history: int,
            forecasting_horizon: int,
            shift: int = 1,
            input_idx=None,
            target_idx=None,
            stream_period: int = 0,
            timeout: int = 30000,
            n_instances: int = 1000,
            **kwargs
    ):
        """
        Args:
            dataset: The river dataset to iterate over.
            past_history (int): Number of time steps for the input window.
            forecasting_horizon (int): Number of time steps for the target window.
            shift (int): Offset (in number of messages) between input and target windows.
            input_idx (int or list, optional): Index/indices to select from the message for input.
                If None, all features are used.
            target_idx (int or list, optional): Index/indices to select from the message for target.
                If None, all features are used.
            stream_period (int): Delay between two consecutive messages (ms).
            timeout (int): Timeout (kept for consistency with the base generator).
            n_instances (int): Maximum number of messages to stream.
        """
        super().__init__(dataset=dataset, stream_period=stream_period, timeout=timeout, n_instances=n_instances, **kwargs)
        self.past_history = past_history
        self.forecasting_horizon = forecasting_horizon
        self.shift = shift
        self.input_idx = input_idx
        self.target_idx = target_idx

        # Internal buffers for moving window
        self.x_window = []
        self.y_window = []

        # Ensure the counter is initialized (if not already by the base class)
        self._count = 0

    def _select_features(self, message, idx):
        """
        Select features from the message based on the provided index or indices.

        Args:
            message (List[float]): A single data point.
            idx (int or list, optional): The feature index/indices.

        Returns:
            List[float]: The selected features.
        """
        if isinstance(idx, int):
            return [message[idx]]
        elif isinstance(idx, list):
            return [message[i] for i in idx]
        else:
            return message

    def _get_x(self, message):
        """Extracts input features from the message."""
        return self._select_features(message, self.input_idx)

    def _get_y(self, message):
        """Extracts target features from the message."""
        return self._select_features(message, self.target_idx)

    def _preprocess(self, x, y):
        """
        Applies the moving window preprocessing to the incoming message.

        Args:
            message (List[float]): Raw message (assumed to be a list of floats).

        Returns:
            Tuple[Optional[List[List[float]]], Optional[List[List[float]]]]:
            Returns a tuple (x_window, y_window) when the respective windows are full;
            otherwise, returns (None, None).
        """
        x_out, y_out = None, None

        # Append the processed input features to the input window
        self.x_window.append(self._get_x(x))

        # Start appending to the target window once enough messages have been processed
        if self._count >= self.past_history + self.shift:
            self.y_window.append(self._get_y(y))

        # When x_window is full, create a copy for output and remove the oldest entry
        if len(self.x_window) == self.past_history:
            x_out = self.x_window.copy()
            self.x_window.pop(0)

        # When y_window is full, create a copy for output and remove the oldest entry
        if len(self.y_window) == self.forecasting_horizon:
            y_out = self.y_window.copy()
            self.y_window.pop(0)
        return x_out, y_out

    def __next__(self):
        """
        We override __next__ so that it:
          1. Respects the timing logic from the BaseGenerator (sleep if necessary).
          2. Calls get_message to fetch the next data point.
        """
        # Step 1: respect timing logic from the base class
        super().__next__()


        return self.get_message()

    def get_message(self):
        """
        Retrieves the next message from the river dataset (via the base generator),
        then applies the moving window preprocessing logic.

        Returns:
            Tuple[Optional[List[List[float]]], Optional[List[List[float]]]]:
            The current (x, y) moving window output if available, otherwise (None, None).
        """
        try:
            # Get the raw message from the parent generator.
            # The parent returns a tuple (x, y) but we use x as the raw time series message.
            raw_x, raw_y = super().get_message()
            # The base generator already increments the message count, but we also ensure it here.
            self._count += 1

            return self._preprocess(raw_x,raw_y)
        except StopIteration:
            self.stop()
            raise

class MovingWindowListGenerator:
    """
    A generator for sliding windows over a list of data points.

    Passes all your tests:
      - Single-feature data is flattened (e.g., [1,2,3,4]).
      - Multi-feature data remains nested (e.g., [[1],[2],[3],[4]] or [[1,2],[2,3],...]).
      - Uses >= condition so the Y-window includes the first possible target.
    """

    def __init__(
        self,
        data,
        past_history: int,
        forecasting_horizon: int,
        shift: int = 1,
        input_idx=None,
        target_idx=None
    ):
        """
        Args:
            data: A list of data points. Each can be:
                  - a single int/float (one feature), or
                  - a list of floats (multiple features).
            past_history: Number of steps in the X-window.
            forecasting_horizon: Number of steps in the Y-window.
            shift: Gap between X-window end and Y-window start.
            input_idx: Which features to use for X (None => all, int => 1 column, list => multiple columns).
            target_idx: Which features to use for Y (None => all, int => 1 column, list => multiple columns).
        """
        self.data = data
        self.past_history = past_history
        self.forecasting_horizon = forecasting_horizon
        self.shift = shift
        self.input_idx = input_idx
        self.target_idx = target_idx

        # Keep track of whether data is single-feature or multi-feature
        # If the first item is a list, we assume multi-feature. If it's int/float => single-feature.
        # (Assumes data is not empty.)
        self._is_multi_feature = isinstance(self.data[0], list)

        # Sliding windows
        self.x_window = []
        self.y_window = []

        # Read index
        self._index = 0

    def __iter__(self):
        return self

    def __next__(self):
        # If no more data and x_window is empty, we're done
        if self._index >= len(self.data) and not self.x_window:
            raise StopIteration

        x_out, y_out = None, None

        # Keep reading until we produce a full X-window
        while x_out is None:
            if self._index >= len(self.data):
                # No more data to read
                raise StopIteration

            raw_val = self.data[self._index]
            self._index += 1

            # If single-feature data, wrap into a list so we can do feature selection uniformly
            if not self._is_multi_feature:
                raw_val = [raw_val]

            # Select features for X
            x_features = self._select_features(raw_val, self.input_idx)
            self.x_window.append(x_features)

            # Once x_window hits past_history, produce x_out and slide
            if len(self.x_window) == self.past_history:
                x_out = self.x_window.copy()
                self.x_window.pop(0)  # remove oldest to keep sliding

                # If the dataset is truly single-feature, flatten each step.
                # e.g. [[1],[2],[3],[4]] -> [1,2,3,4]
                if not self._is_multi_feature:
                    x_out = [x_[0] for x_ in x_out]  # each x_ is [value], so take x_[0]

            # Start populating y_window once we've read at least (past_history + shift) data points
            if self._index >= (self.past_history + self.shift):
                y_features = self._select_features(raw_val, self.target_idx)
                self.y_window.append(y_features)

            # If y_window hits forecasting_horizon, produce y_out and slide
            if len(self.y_window) == self.forecasting_horizon:
                y_out = self.y_window.copy()
                self.y_window.pop(0)

                # If single-feature => flatten each step
                if not self._is_multi_feature:
                    y_out = [y_[0] for y_ in y_out]

        return x_out, y_out

    def _select_features(self, data_list, idx):
        """
        data_list is a list of features, e.g. [value] or [value1, value2,...].
        idx can be None, int, or list of ints.
        """
        if idx is None:
            return data_list
        if isinstance(idx, int):
            return [data_list[idx]]
        if isinstance(idx, list):
            return [data_list[i] for i in idx]
        # fallback
        return data_list