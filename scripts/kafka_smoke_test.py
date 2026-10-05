# File: scripts/kafka_smoke_test.py
# Purpose: A tiny end-to-end check that the Kafka-compatible broker (Redpanda) is reachable and works: send ONE
#          message to a throwaway topic, read it back, and print PASS or FAIL.
#
# It only touches the throwaway topic "smoke_test". No business data, no outbox, no /score, no e-commerce store.
#
# Run from the repo root (the broker must be up: docker compose up -d redpanda):
#     PYTHONPATH=. python scripts/kafka_smoke_test.py
#
# Which broker: it uses the same config object as the rest of the app (config.kafka_bootstrap_servers, read from the
# KAFKA_BOOTSTRAP_SERVERS environment variable, default localhost:19092, the address that works from the host).
import asyncio
import sys
import uuid
from pathlib import Path

from aiokafka import AIOKafkaConsumer, AIOKafkaProducer

# Make the repo root importable, so "from core.config import config" works when this file is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import config  # noqa: E402

TOPIC = "smoke_test"
WAIT_SECONDS = 20  # how long we wait to read our message back before calling it a failure


async def send_one_message(message_text: str) -> str:
    """
    Send one message and wait for the broker to confirm it was stored. Returns where it landed.

    Why "send_and_wait": it does not return until the broker acknowledges the message. If the broker is down or
    refuses the message, it raises an error instead of silently losing it.
    """
    producer = AIOKafkaProducer(bootstrap_servers=config.kafka_bootstrap_servers)
    try:
        await producer.start()  # inside the try, so a failed start is still cleaned up by the finally below
        metadata = await producer.send_and_wait(TOPIC, message_text.encode("utf-8"))
        return f"partition {metadata.partition}, offset {metadata.offset}"
    finally:
        await producer.stop()


async def read_message_back(message_text: str) -> bool:
    """
    Read the topic from the beginning and look for our exact message. Returns True if it shows up in time.

    Why look for our own unique text: the topic may already hold messages from earlier runs, so "the first
    message I see" would prove nothing. Finding OUR unique text proves the round trip.
    """
    consumer = AIOKafkaConsumer(
        TOPIC,
        bootstrap_servers=config.kafka_bootstrap_servers,
        auto_offset_reset="earliest",  # start from the oldest message, not only new ones
        enable_auto_commit=False,  # a smoke test should not leave consumer-group bookkeeping behind
    )
    await consumer.start()
    try:
        async for record in consumer:
            if record.value.decode("utf-8") == message_text:
                print(f"read back : {message_text} (partition {record.partition}, offset {record.offset})")
                return True
    finally:
        await consumer.stop()
    return False


async def run_smoke_test() -> bool:
    """Send one unique message, then read it back, with an overall time limit."""
    message_text = f"smoke-test-{uuid.uuid4()}"
    print(f"broker    : {config.kafka_bootstrap_servers}")
    print(f"sending   : {message_text}")
    where = await send_one_message(message_text)
    print(f"acked     : stored at {where}")
    try:
        return await asyncio.wait_for(read_message_back(message_text), timeout=WAIT_SECONDS)
    except asyncio.TimeoutError:
        return False


def main() -> int:
    try:
        passed = asyncio.run(run_smoke_test())
    except Exception as error:  # noqa: BLE001 - any failure (broker down, wrong address) is a clear FAIL
        print(f"FAIL: {type(error).__name__}: {error}")
        return 1
    print("PASS: message went to the broker and came back." if passed else "FAIL: message did not come back in time.")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
