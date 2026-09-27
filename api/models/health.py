"""
File: api/models/health.py
Purpose: Pydantic response model for the GET /health endpoint.
         Pins the exact shape the endpoint must return so changes that break
         the contract are caught at the model layer, not in production.
"""

from pydantic import BaseModel, ConfigDict, model_validator


class HealthResponse(BaseModel):
    status: str
    environment: str

    # frozen=True makes every instance read-only after it is created.
    # This prevents accidental mutation of a response object before it is sent.
    model_config = ConfigDict(frozen=True)

    @model_validator(mode="after")
    def status_must_not_be_empty(self) -> "HealthResponse":
        # If the health handler is misconfigured and returns an empty status string,
        # we want to catch that here before the response leaves the service,
        # rather than returning a silent empty string to callers.
        if not self.status:
            raise ValueError("status must not be an empty string")
        return self
