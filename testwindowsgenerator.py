from generator.movingwindow_river_generator import MovingWindowListGenerator


def _test_moving_window(
    data,
    expected_X,
    expected_y,
    past_history=4,
    forecasting_horizont=2,
    shift=1,
    input_idx=None,
    target_idx=None,
):
    # Instantiate the generator with your data and parameters
    generator = MovingWindowListGenerator(
        data=data,
        past_history=past_history,
        forecasting_horizon=forecasting_horizont,
        shift=shift,
        input_idx=input_idx,
        target_idx=target_idx,
    )

    collected_X = []
    collected_y = []

    # Drain the generator
    for x_window, y_window in generator:
        # x_window is guaranteed to have length = past_history once it's yielded
        collected_X.append(x_window)

        # y_window might be None if forecasting_horizon isn't ready
        if y_window is not None:
            collected_y.append(y_window)

    # After we've read everything, compare
    # The "expected_X" is all the X windows in order
    # The "expected_y" is only the times we have a full Y window
    assert collected_X == expected_X, f"\nGot X={collected_X}\nExpected X={expected_X}"
    assert collected_y == expected_y, f"\nGot y={collected_y}\nExpected y={expected_y}"
def test_one_variable():
    data = [x for x in range(1, 10)]  # Single-dimensional data
    expected_X = [
        [1, 2, 3, 4],
        [2, 3, 4, 5],
        [3, 4, 5, 6],
        [4, 5, 6, 7],
        [5, 6, 7, 8],
        [6, 7, 8, 9],
    ]
    expected_y = [[5, 6], [6, 7], [7, 8], [8, 9]]

    _test_moving_window(data, expected_X, expected_y)


def test_multivariable_variable():
    data = [[x, x] for x in range(1, 10)]
    expected_X = [
        [[1, 1], [2, 2], [3, 3], [4, 4]],
        [[2, 2], [3, 3], [4, 4], [5, 5]],
        [[3, 3], [4, 4], [5, 5], [6, 6]],
        [[4, 4], [5, 5], [6, 6], [7, 7]],
        [[5, 5], [6, 6], [7, 7], [8, 8]],
        [[6, 6], [7, 7], [8, 8], [9, 9]],
    ]
    expected_y = [
        [[5, 5], [6, 6]],
        [[6, 6], [7, 7]],
        [[7, 7], [8, 8]],
        [[8, 8], [9, 9]],
    ]

    _test_moving_window(data, expected_X, expected_y)

def test_multivariable_variable_one_input():
    data = [[x, x + 1] for x in range(1, 10)]
    expected_X = [
        [[1], [2], [3], [4]],
        [[2], [3], [4], [5]],
        [[3], [4], [5], [6]],
        [[4], [5], [6], [7]],
        [[5], [6], [7], [8]],
        [[6], [7], [8], [9]],
    ]
    expected_y = [
        [[5, 6], [6, 7]],
        [[6, 7], [7, 8]],
        [[7, 8], [8, 9]],
        [[8, 9], [9, 10]],
    ]
    _test_moving_window(data, expected_X, expected_y, input_idx=0)


def test_multivariable_variable_one_output():
    data = [[x, x + 1] for x in range(1, 10)]
    expected_X = [
        [[1, 2], [2, 3], [3, 4], [4, 5]],
        [[2, 3], [3, 4], [4, 5], [5, 6]],
        [[3, 4], [4, 5], [5, 6], [6, 7]],
        [[4, 5], [5, 6], [6, 7], [7, 8]],
        [[5, 6], [6, 7], [7, 8], [8, 9]],
        [[6, 7], [7, 8], [8, 9], [9, 10]],
    ]
    expected_y = [[[5], [6]], [[6], [7]], [[7], [8]], [[8], [9]]]
    _test_moving_window(data, expected_X, expected_y, target_idx=0)


def test_shift_one_variable():
    data = [x for x in range(1, 10)]
    expected_X = [
        [1, 2, 3, 4],
        [2, 3, 4, 5],
        [3, 4, 5, 6],
        [4, 5, 6, 7],
        [5, 6, 7, 8],
        [6, 7, 8, 9],
    ]
    expected_y = [[7, 8], [8, 9]]
    _test_moving_window(data, expected_X, expected_y, shift=3)


def test_shift_multivariable():
    data = [[x, x + 1] for x in range(1, 10)]
    expected_X = [
        [[1, 2], [2, 3], [3, 4], [4, 5]],
        [[2, 3], [3, 4], [4, 5], [5, 6]],
        [[3, 4], [4, 5], [5, 6], [6, 7]],
        [[4, 5], [5, 6], [6, 7], [7, 8]],
        [[5, 6], [6, 7], [7, 8], [8, 9]],
        [[6, 7], [7, 8], [8, 9], [9, 10]],
    ]
    expected_y = [[[7, 8], [8, 9]], [[8, 9], [9, 10]]]
    _test_moving_window(data, expected_X, expected_y, shift=3)


def test_shift_multivariable_one_output():
    data = [[x, x + 1] for x in range(1, 10)]
    expected_X = [
        [[1, 2], [2, 3], [3, 4], [4, 5]],
        [[2, 3], [3, 4], [4, 5], [5, 6]],
        [[3, 4], [4, 5], [5, 6], [6, 7]],
        [[4, 5], [5, 6], [6, 7], [7, 8]],
        [[5, 6], [6, 7], [7, 8], [8, 9]],
        [[6, 7], [7, 8], [8, 9], [9, 10]],
    ]
    expected_y = [[[7], [8]], [[8], [9]]]
    _test_moving_window(data, expected_X, expected_y, shift=3, target_idx=0)


if __name__ == "__main__":
    # Quick manual run of a single test
    test_one_variable()
    print("test_one_variable passed!")

    # You can do the same for the others:
    test_multivariable_variable()
    print("test_multivariable_variable passed!")

    test_multivariable_variable_one_input()
    print("test_multivariable_variable_one_input passed!")

    test_multivariable_variable_one_output()
    print("test_multivariable_variable_one_output passed!")

    test_shift_one_variable()
    print("test_shift_one_variable passed!")

    test_shift_multivariable()
    print("test_shift_multivariable passed!")

    test_shift_multivariable_one_output
    print("test_shift_multivariable_one_output passed!")
    # etc.
