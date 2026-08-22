"""Kafka producer wrapper for the two topics frozen in contracts/kafka_topics.md.

Fails soft: if Kafka isn't reachable (e.g. local dev without docker-compose up), the
service should still serve /recommend and /feedback -- /health is what reports the
Kafka-down state, not a hard crash on every request.
"""

import json
import logging
import os

from kafka import KafkaProducer
from kafka.errors import KafkaError

logger = logging.getLogger(__name__)

BOOTSTRAP_SERVERS = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
RECOMMENDATION_EVENTS_TOPIC = "recommendation-events"
USER_FEEDBACK_TOPIC = "user-feedback"


class RecommendationEventProducer:
    def __init__(self, bootstrap_servers: str = BOOTSTRAP_SERVERS):
        self._producer = None
        self._bootstrap_servers = bootstrap_servers

    def _get_producer(self) -> KafkaProducer:
        if self._producer is None:
            self._producer = KafkaProducer(
                bootstrap_servers=self._bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                request_timeout_ms=3000,
            )
        return self._producer

    def is_reachable(self) -> bool:
        try:
            producer = self._get_producer()
            return producer.bootstrap_connected()
        except Exception:  # noqa: BLE001
            return False

    def send_recommendation_event(
        self, user_id: str, track_id: str, policy: str, timestamp: str
    ) -> None:
        payload = {
            "user_id": user_id,
            "track_id": track_id,
            "policy": policy,
            "timestamp": timestamp,
        }
        self._send(RECOMMENDATION_EVENTS_TOPIC, payload)

    def send_user_feedback(
        self, user_id: str, track_id: str, action: str, reward: float, timestamp: str
    ) -> None:
        payload = {
            "user_id": user_id,
            "track_id": track_id,
            "action": action,
            "reward": reward,
            "timestamp": timestamp,
        }
        self._send(USER_FEEDBACK_TOPIC, payload)

    def _send(self, topic: str, payload: dict) -> None:
        try:
            producer = self._get_producer()
            producer.send(topic, payload)
            producer.flush(timeout=3)
        except KafkaError as exc:
            logger.warning("Kafka send to %s failed (continuing without it): %s", topic, exc)
