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
        super().__init__(dataset, stream_period=stream_period, timeout=timeout, n_instances=n_instances, **kwargs)
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
