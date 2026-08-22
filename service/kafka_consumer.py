"""Background consumer for the user-feedback topic: reads feedback events and calls
the active policy's update() -- this is what closes the online learning loop.
"""

import json
import logging
import threading

from kafka import KafkaConsumer

from service.kafka_producer import BOOTSTRAP_SERVERS, USER_FEEDBACK_TOPIC

logger = logging.getLogger(__name__)


class FeedbackConsumer:
    """Runs consume_loop() in a background thread; on_feedback(event) is called for
    every user-feedback message so the caller decides how to route it to the policy."""

    def __init__(self, on_feedback, bootstrap_servers: str = BOOTSTRAP_SERVERS):
        self._on_feedback = on_feedback
        self._bootstrap_servers = bootstrap_servers
        self._stop_event = threading.Event()
        self._thread = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._consume_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()

    def _consume_loop(self) -> None:
        try:
            consumer = KafkaConsumer(
                USER_FEEDBACK_TOPIC,
                bootstrap_servers=self._bootstrap_servers,
                value_deserializer=lambda v: json.loads(v.decode("utf-8")),
                consumer_timeout_ms=1000,
                auto_offset_reset="latest",
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("FeedbackConsumer could not connect to Kafka: %s", exc)
            return

        while not self._stop_event.is_set():
            for message in consumer:
                if self._stop_event.is_set():
                    break
                try:
                    self._on_feedback(message.value)
                except Exception:  # noqa: BLE001
                    logger.exception("error handling feedback event: %s", message.value)
