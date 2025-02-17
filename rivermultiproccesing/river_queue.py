import os
import pickle
import uuid
import queue
import time
import multiprocessing
import logging
from river import base  # used for type annotation, optional

# -----------------------------
# Configure the root logger here or in main.py:
# Logging levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
logging.basicConfig(
    level=logging.ERROR,  # Set the minimum logging level you want
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
# -----------------------------

class RiverModelServer(multiprocessing.Process):
    """
    A background process that holds a single River model, listens for commands
    to predict or train, and optionally saves/loads the model to disk.
    """

    def __init__(
        self,
        model: 'base.Estimator',
        request_queue: multiprocessing.Queue,
        response_queue: multiprocessing.Queue,
        stop_event: multiprocessing.Event,
        model_path: str = None
    ):
        """
        :param model:         A River model or pipeline (e.g. compose.Pipeline(...))
        :parax< m request_queue:  Queue for ("predict", x, id) or ("train", x, y) commands
        :param response_queue: Queue for responses ("prediction", id, y_pred)
        :param stop_event:     Event to signal shutdown
        :param model_path:     Path to load/save the model (if not None)
        """
        super().__init__(daemon=True)
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.stop_event = stop_event
        self.model_path = model_path

        if model_path is not None and os.path.exists(model_path):
            logger.info(f"Loading existing model from {model_path}")
            with open(model_path, "rb") as f:
                self.model = pickle.load(f)
            logger.debug("Model successfully loaded from disk.")
        else:
            logger.info("No existing model found, using provided model instance.")
            self.model = model

    def run(self):
        logger.info("Starting model server process.")
        while not self.stop_event.is_set():
            try:
                msg = self.request_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if not isinstance(msg, tuple):
                logger.warning(f"Received an unexpected message format: {msg}")
                continue

            command = msg[0]

            if command == "predict":
                # ("predict", x_dict, request_id)
                _, x_dict, request_id = msg
                logger.debug(f"Received predict request for id={request_id} with x={x_dict}")
                y_pred = self.model.predict_one(x_dict)
                logger.debug(f"Prediction for id={request_id}: {y_pred}")
                self.response_queue.put(("prediction", request_id, y_pred))

            elif command == "train":
                # ("train", x_dict, y_label)
                _, x_dict, y_label = msg
                logger.debug(f"Received train request with x={x_dict}, y={y_label}")
                self.model.learn_one(x_dict, y_label)
                logger.debug("Model updated with one training example.")

            else:
                logger.warning(f"Unknown command: {command}")

        # Save the model on shutdown if a path was provided
        if self.model_path is not None:
            logger.info(f"Saving model to {self.model_path}")
            with open(self.model_path, "wb") as f:
                pickle.dump(self.model, f)
            logger.debug("Model saved successfully.")

        logger.info("Model server process stopped.")


class RiverModelManager:
    """
    Spawns the RiverModelServer in a separate process and provides methods to train/predict.
    Optionally persists the model to disk on stop, or loads it if it exists.
    """

    def __init__(self, model: 'base.Estimator', model_path: str = None):
        """
        :param model:      A River model or pipeline
        :param model_path: File path for saving/loading the model
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Initializing RiverModelManager.")

        self.request_queue = multiprocessing.Queue()
        self.response_queue = multiprocessing.Queue()
        self.stop_event = multiprocessing.Event()

        self.server = RiverModelServer(
            model=model,
            request_queue=self.request_queue,
            response_queue=self.response_queue,
            stop_event=self.stop_event,
            model_path=model_path
        )
        self.server.start()
        self.logger.info("RiverModelServer process started.")

    def predict_one(self, x_dict: dict):
        """
        Send a predict request and wait for the server's response.
        :param x_dict: features dict
        :return: prediction from the model
        """
        request_id = str(uuid.uuid4())
        self.logger.debug(f"Sending predict request {request_id} with x={x_dict}")
        self.request_queue.put(("predict", x_dict, request_id))

        while True:
            msg = self.response_queue.get()
            if msg[0] == "prediction":
                _, resp_id, y_pred = msg
                if resp_id == request_id:
                    self.logger.debug(f"Received prediction for request {request_id}: {y_pred}")
                    return y_pred
            else:
                self.logger.warning(f"Unexpected message in response queue: {msg}")

    def learn_one(self, x_dict: dict, y_label):
        """
        Send a train request to the server.
        Note: This is non-blocking; it just sends the request.
        """
        self.logger.debug(f"Sending train request with x={x_dict}, y={y_label}")
        self.request_queue.put(("train", x_dict, y_label))

    def stop(self):
        """
        Signal the server to stop and wait for it to exit.
        If model_path was provided, it saves the model to disk.
        """
        self.logger.info("Stopping the model server...")
        self.stop_event.set()
        self.server.join()
        self.logger.info("Model server has stopped.")