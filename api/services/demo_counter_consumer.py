"""
File: api/services/demo_counter_consumer.py
Purpose: A tiny example consumer that proves the idempotent-consumer helper works. For every NEW event it adds 1 to a
         counter in the demo_event_counts table. A duplicated event is skipped by the helper, so the counter ends up
         equal to the number of DISTINCT events, not the number of messages.

It is run on demand, not as a permanent compose service:
    python -m api.services.demo_counter_consumer --group-id some-fresh-group --idle-timeout 5
A fresh --group-id makes it read the topic from the very beginning.
"""

import argparse
import asyncio
import uuid

import asyncpg

from api.services.idempotent_consumer import run_consumer
from core.config import config
from core.logging import configure_logging

CONSUMER_NAME = "demo"


async def count_one_event(conn: asyncpg.Connection) -> None:
    """The example's real work: add 1 to this consumer's counter (creating the row the first time)."""
    await conn.execute(
        """
        INSERT INTO demo_event_counts (consumer_name, events_counted) VALUES ($1, 1)
        ON CONFLICT (consumer_name) DO UPDATE SET events_counted = demo_event_counts.events_counted + 1
        """,
        CONSUMER_NAME,
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Demo consumer: counts distinct transaction events")
    parser.add_argument("--group-id", default=f"demo-{uuid.uuid4()}",
                        help="Kafka consumer group (default: a fresh one)")
    parser.add_argument("--idle-timeout", type=float, default=None,
                        help="stop after this many seconds without a new message (default: run until stopped)")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    configure_logging(config.log_level)
    try:
        asyncio.run(run_consumer(CONSUMER_NAME, config.kafka_outbox_topic, args.group_id, count_one_event,
                                 idle_timeout_seconds=args.idle_timeout))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
