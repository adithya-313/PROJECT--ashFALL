"""
File: core/tracing.py
Purpose: Sets up OpenTelemetry distributed tracing and injects trace_id into structlog.
Connects: Called from api/main.py after the FastAPI app is created.
          Exports spans to Tempo via the OTLP HTTP protocol.
          trace_id_ctx_var is read by core/logging.py to attach trace_id to every log line.
"""

from contextvars import ContextVar
from typing import Optional

import fastapi
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Context variable that holds the active trace_id for the current async call stack.
# Set to None when no trace is active (e.g. outside an HTTP request).
# core/logging.py reads this to attach trace_id to every JSON log line.
trace_id_ctx_var: ContextVar[Optional[str]] = ContextVar("trace_id", default=None)


def setup_tracing(app: fastapi.FastAPI) -> None:
    """
    Wires OpenTelemetry tracing into the FastAPI application.

    What this does:
    1. Creates a TracerProvider that labels all spans with the service name.
    2. Attaches an OTLP exporter that sends completed spans to Tempo over HTTP.
    3. Instruments FastAPI so every HTTP request automatically becomes a root span.
    4. Registers a request hook that copies the active trace_id into a ContextVar
       so structlog can include it in every log line for that request.
    """
    # Read the OTLP endpoint from the validated app config.
    # Defaults to Jaeger's address inside Docker; override OTLP_ENDPOINT in .env for local dev.
    from core.config import config

    otlp_endpoint = config.otlp_endpoint

    # A Resource labels every span with the name of the service that produced it.
    # Grafana uses this name to group spans in the trace view.
    resource = Resource.create({"service.name": "ashfall-api"})

    # TracerProvider is the factory that creates individual tracers.
    # BatchSpanProcessor buffers spans and sends them in batches to reduce overhead.
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint=otlp_endpoint))
    )

    # Register the provider globally so any opentelemetry.trace.get_tracer() call
    # in any module returns a tracer backed by this provider.
    trace.set_tracer_provider(provider)

    def _set_trace_id_in_context(span: trace.Span, scope: dict) -> None:
        """
        Called by FastAPIInstrumentor at the start of each request.
        Copies the active trace_id into the ContextVar so structlog can read it.
        """
        ctx = span.get_span_context()
        if ctx and ctx.is_valid:
            # Format the 128-bit integer trace_id as a 32-character hex string,
            # which is the standard representation used by Grafana and Tempo.
            trace_id_ctx_var.set(format(ctx.trace_id, "032x"))

    # FastAPIInstrumentor patches FastAPI's routing layer to create a span for
    # every incoming HTTP request. The server_request_hook fires at request start.
    FastAPIInstrumentor.instrument_app(
        app,
        server_request_hook=_set_trace_id_in_context,
    )
