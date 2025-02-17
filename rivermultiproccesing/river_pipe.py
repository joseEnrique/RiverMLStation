import multiprocessing
import pickle
import os
import uuid
import logging
from river import base  # For type annotation

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

class RiverModelProcess(multiprocessing.Process):
    """
    A background process that holds a River model and listens for commands
    over a Pipe (instead of a multiprocessing.Queue).
    """

    def __init__(
        self,
        model: 'base.Estimator',
        pipe_conn,
        stop_event: multiprocessing.Event,
        model_path: str = None
    ):
        super().__init__(daemon=True)
        self.pipe_conn = pipe_conn
        self.stop_event = stop_event
        self.model_path = model_path

        # Load the model from disk if it exists
        if model_path is not None and os.path.exists(model_path):
            logger.info(f"Loading existing model from {model_path}")
            with open(model_path, "rb") as f:
                self.model = pickle.load(f)
            logger.info("Model loaded.")
        else:
            logger.info("No existing model found (or no path). Using the provided model.")
            self.model = model

    def run(self):
        logger.info("RiverModelProcess started.")
        while not self.stop_event.is_set():
            # Check if there's data from the pipe
            if self.pipe_conn.poll(0.01):
                # Use a tiny timeout to avoid blocking the shutdown
                msg = self.pipe_conn.recv()  # Blocking read
                if not isinstance(msg, dict):
                    logger.warning(f"Unrecognized message format: {msg}")
                    continue

                command = msg.get("command")

                if command == "predict":
                    x_dict = msg["x_dict"]
                    request_id = msg["request_id"]
                    y_pred = self.model.predict_one(x_dict)
                    # Send response back
                    response = {
                        "type": "prediction",
                        "request_id": request_id,
                        "y_pred": y_pred
                    }
                    self.pipe_conn.send(response)

                elif command == "train":
                    x_dict = msg["x_dict"]
                    y_label = msg["y_label"]
                    self.model.learn_one(x_dict, y_label)

                else:
                    logger.warning(f"Unknown command {command}")
            # Loop again, checking stop_event

        # Save the model on shutdown
        if self.model_path:
            logger.info(f"Saving model to {self.model_path}")
            with open(self.model_path, "wb") as f:
                pickle.dump(self.model, f)
            logger.info("Model saved.")
        logger.info("RiverModelProcess stopped.")


class RiverModelManagerPipe:
    """
    Demonstrates a manager class that spawns a RiverModelProcess using a Pipe.
    """

    def __init__(self, model: 'base.Estimator', model_path: str = None):
        self.logger = logging.getLogger(self.__class__.__name__)

        # Create a Pipe (two connection objects, one for parent, one for child)
        parent_conn, child_conn = multiprocessing.Pipe(duplex=True)

        # Event for stopping the child process
        stop_event = multiprocessing.Event()

        # Create and start the child process
        self.proc = RiverModelProcess(
            model=model,
            pipe_conn=child_conn,
            stop_event=stop_event,
            model_path=model_path
        )
        self.proc.start()

        # Keep references for usage
        self.parent_conn = parent_conn
        self.stop_event = stop_event

    def predict_one(self, x_dict: dict):
        """
        Send a prediction request and wait for the result.
        """
        request_id = str(uuid.uuid4())

        # Send message to the child
        msg = {
            "command": "predict",
            "x_dict": x_dict,
            "request_id": request_id
        }
        self.parent_conn.send(msg)

        # Wait for the matching response
        while True:
            response = self.parent_conn.recv()
            if response.get("type") == "prediction":
                if response["request_id"] == request_id:
                    return response["y_pred"]
            else:
                self.logger.warning(f"Unexpected response: {response}")

    def learn_one(self, x_dict: dict, y_label):
        """
        Send a train request (non-blocking).
        """
        msg = {
            "command": "train",
            "x_dict": x_dict,
            "y_label": y_label
        }
        self.parent_conn.send(msg)

    def stop(self):
        """
        Signal the process to stop and wait for it to exit.
        """
        self.logger.info("Stopping the model process...")
        self.stop_event.set()
        self.proc.join()
        self.logger.info("Process stopped.")

# -------------------------------------------------------------------
# Usage example (uncomment to test as a script):
#
# if __name__ == "__main__":
#     from river.linear_model import LogisticRegression
#
#     # Instantiate any River model or pipeline
#     model = LogisticRegression()
#
#     # Create manager
#     manager = RiverModelManagerPipe(model=model, model_path="my_model.pkl")
#
#     # Train
#     for i in range(5):
#         x = {"feature": i}
#         y = i % 2  # Just a dummy label
#         manager.learn_one(x, y)
#
#     # Predict
#     x_test = {"feature": 5}
#     pred = manager.predict_one(x_test)
#     print("Prediction:", pred)
#
#     # Stop
#     manager.stop()
