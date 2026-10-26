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
- The DPIP API reference requires a Pratirupa login and offers no public signup, so the field names are not public.
- A sixth DPIP service, augmentation, is under consideration and subject to approval. It is NOT live.
- The mocks in Section 2 keep their own design names.

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

### Fields added for Level 11 (N1, N2, N3)

None of these field names are official; they are our design.

N1, identifier reuse: add one field `distinct_accounts_linked` (a whole number: how many different accounts this hashed identifier has been reported against) to the mock screening response example. The same identifier can appear on several mule accounts, and a gang may switch to a fresh identifier after one is flagged.
```json
{
  "request": {"identifier_hash": "example_hash_001", "identifier_type": "example_type", "requesting_bank_id": "example_bank_1"},
  "response": {"match": true, "status": "example_status", "risk_score": 0.0, "distinct_accounts_linked": 4}
}
```

N2, per-bank participation: a small mock bank settings example. Each of the three banks has this `dpip_participation` flag, so some banks do not receive registry matches.
```json
{
  "bank_id": "example_bank_1",
  "dpip_participation": false
}
```

N3, hold latency: in the mock hold response add `requested_at`, `applied_at`, `latency_minutes` and `outcome`, where outcome is one of "applied" (the hold succeeded), "too_late" (the money left just before the hold landed) or "missed" (the hold completely failed). Money may already have left the account before a hold lands.
```json
{
  "request": {"request_id": "example_req_1", "account_ref": "example_account_1", "amount_inr": 1000, "action": "hold_reported_amount", "issued_at": "2026-01-01T00:00:00Z"},
  "response": {"request_id": "example_req_1", "status": "example_status", "held_amount_inr": 1000, "updated_at": "2026-01-01T00:00:00Z", "requested_at": "2026-01-01T00:00:00Z", "applied_at": "2026-01-01T00:15:00Z", "latency_minutes": 15, "outcome": "applied"}
}
```

## Level 12e: registry-report events emitted by the world (OUR DESIGN)

The simulated world emits one event each time a bank reports an account to the national registry. Level 13 reads
these events and turns them into mock registry entries (hashing identifiers on the bank side, as Section 1
describes). Times are world minutes, IST wall time with no offset (ground rule 30). The event carries no truth: it
never says whether the account really is a mule.

```json
{
  "report_id": "<run_id>-R000001",
  "kind": "registry_report",
  "reported_minute": 4321,
  "reporting_bank_index": 1,
  "account_id": "acc_person_0042",
  "device_fingerprint": "example_fingerprint_or_null",
  "confirmed_minute": 3900,
  "complaint_ids": ["<run_id>-L000003"]
}
```

- `reported_minute`: when the report reaches the registry (`confirmed_minute` + the drawn delay).
- `confirmed_minute`: arrival minute of the first complaint naming the account.
- `device_fingerprint`: the account holder's device fingerprint when the holder is a Level 11 person; null for shops
  and accounts opened mid-run.
- `complaint_ids`: every complaint about the account that arrived at or before `confirmed_minute`.
- Every wrong registry entry in the simulation comes from these events. Level 13 adds no second error rate.

## Revision 2 (Level 13a) (OUR DESIGN, not official)

National systems mock contract additions and updates for Level 13a:

- **Amounts in paise**: All monetary amounts are integer paise. Mocks no longer use floating or rupee units.
- **Hold requests carry complaint and payment references**: Every hold request carries `complaint_id` and the victim's `payment_id`, and uses `amount_paise`. Amended hold example:
```json
{
  "request": {
    "request_id": "example_req_1",
    "account_ref": "example_account_1",
    "amount_paise": 100000,
    "complaint_id": "example_complaint_001",
    "payment_id": "example_payment_001",
    "action": "hold_reported_amount",
    "issued_at": "2026-09-01T10:00:00+05:30"
  },
  "response": {
    "request_id": "example_req_1",
    "status": "example_status",
    "held_amount_paise": 100000,
    "updated_at": "2026-09-01T10:15:00+05:30",
    "requested_at": "2026-09-01T10:00:00+05:30",
    "applied_at": "2026-09-01T10:15:00+05:30",
    "latency_minutes": 15,
    "outcome": "applied"
  }
}
```
- **Timestamps follow docs/interfaces/time.md**: Timestamps are ISO 8601 strings with timezone offset `+05:30` (aware IST), conforming to Ground Rule 30.
- **Keyed hashing for identifiers**: Identifiers are hashed using HMAC-SHA256 with a key held strictly by DPIP participants. The message format is `<run_id>|<identifier_type>|<namespaced raw value>`, where for account identifiers the raw value is namespaced as `<bank_index>:<account_id>` and for devices the raw value is the device fingerprint string.
- **DPIP Phase-2 transaction scores out of scope**: Transaction scoring is out of scope and is not built. Mock risk scores are fixed at `None` or `0.0`.
- **Listing delay distinct from report delay**: The registry listing delay (`listing_delay_minutes`) represents the registry's internal ingestion processing time before an entry becomes visible to screening, separate and distinct from Level 12's `registry_report_delay_minutes` (bank reporting latency).
