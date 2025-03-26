from generator.base_generator import BaseGenerator


class MovingWindowListGenerator(BaseGenerator):
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
        target_idx=None,
        stream_period=0,
        timeout=30000
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
        super().__init__(stream_period=stream_period, timeout=timeout)
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
        self._count = 0

    def __iter__(self):
        return self

    def __next__(self):
        super().__next__()
        # If no more data and x_window is empty, we're done
        if self._count >= len(self.data) and not self.x_window:
            raise StopIteration
        return self.get_message()

    def get_count(self):
        return self._count

    def get_message(self):

        x_out, y_out = None, None

        # Keep reading until we produce a full X-window
        while x_out is None:
            if self._count >= len(self.data):
                # No more data to read
                raise StopIteration

            raw_val = self.data[self._count]
            self._count += 1

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
            if self._count >= (self.past_history + self.shift):
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