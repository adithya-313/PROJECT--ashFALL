# National systems: what is real and what is mocked

**What this file is for**
It separates what is publicly known about India's shared fraud systems from the mock shapes this project invents.
The mock shapes let ashFALL be tested end to end without any real connection.
Nothing here is an official interface.

## Section 1. What is real

"Hand checked" is no unless stated.

### DPIP (Digital Payments Intelligence Platform)
- Source: RBIH documentation, https://docs.rbihub.in/digital-payments-intelligence-platform
- Five services are named: Ingestion, Screening, Feedback, Revoke, Dispute.
- Identifiers are hashed inside the reporting bank before they are sent.
- Screening returns, at a high level, whether an identifier matches, its status and a risk score.
- Transaction scoring is planned for phase 2.
- Status: **confirmed**.
- "Live on 1 Sep 2026 with 8 banks" and "salted" come only from MediaNama, 11 Sep 2026. Status: **reported**.
- The API reference with endpoint paths and field names could not be opened. Status: **unverified**.

### I4C Suspect Registry
- Launched 10 Sep 2024 (I4C page).
- MHA Lok Sabha answers: more than 23.05 lakh suspect identifiers and 27.37 lakh Layer 1 mule accounts (data to 31 Jan 2026); more than 30.48 lakh and 32.08 lakh (data to 30 Jun 2026). (1 lakh = 100,000.)
- Status: **confirmed**.
- Number of banks on the registry itself: **unverified**. A 62-banks figure is for the cybercrime portal, not the registry.

### CFCFRMS hold
- Source: MHA SOP, 2 Jan 2026. Hand checked: **yes**.
- Sentence: "The beneficiary bank or FI will put on hold an amount to the extent of the amount reported and update the same on CFCFRMS."
- The message fields are not public. Status: **unverified**.

## Section 2. Mock shapes (OUR DESIGN, not official)

None of these field names come from RBIH, I4C or MHA.
They are invented for testing. Values are examples only.
If the real field names are ever published, they replace these.

Mock DPIP screening request and response:

```json
{
  "request": {"identifier_hash": "example_hash_001", "identifier_type": "example_type", "requesting_bank_id": "example_bank_1"},
  "response": {"match": true, "status": "example_status", "risk_score": 0.0}
}
```

Mock DPIP ingestion, feedback, revoke and dispute (one example each, same fields):

```json
{
  "ingestion": {"identifier_hash": "example_hash_001", "bank_id": "example_bank_1", "reason_code": "example_code", "reported_at": "2026-01-01T00:00:00Z"},
  "feedback":  {"identifier_hash": "example_hash_001", "bank_id": "example_bank_1", "reason_code": "example_code", "reported_at": "2026-01-01T00:00:00Z"},
  "revoke":    {"identifier_hash": "example_hash_001", "bank_id": "example_bank_1", "reason_code": "example_code", "reported_at": "2026-01-01T00:00:00Z"},
  "dispute":   {"identifier_hash": "example_hash_001", "bank_id": "example_bank_1", "reason_code": "example_code", "reported_at": "2026-01-01T00:00:00Z"}
}
```

Mock registry feed entry:

```json
{"identifier_hash": "example_hash_001", "layer": 1, "listed_at": "2026-01-01T00:00:00Z"}
```

Mock hold request and response. The `action` is one of the names in `config/legal_actions.yaml`:

```json
{
  "request": {"request_id": "example_req_1", "account_ref": "example_account_1", "amount_inr": 1000, "action": "hold_reported_amount", "issued_at": "2026-01-01T00:00:00Z"},
  "response": {"request_id": "example_req_1", "status": "example_status", "held_amount_inr": 1000, "updated_at": "2026-01-01T00:00:00Z"}
}
```
