# Time in ashFALL: the time contract and Ground Rule 30

## Ground Rule 30: Event Time vs Transport Time

1. **IST World Time**: Simulated time in ashFALL is strictly in Indian Standard Time (IST, UTC+05:30), with no offset written. The simulation start date is fixed at `2026-09-01` (`WORLD_START_DATE = date(2026, 9, 1)`).
2. **Explicit Event Time**: Every payment event, failure, outside intervention, label, and registry entry records its event time explicitly as elapsed world minutes from simulation start (`minute: int`), or as an aware IST datetime.
3. **Event Time Drives Logic**: Event time drives every temporal window, simulation step, delay calculation, hold duration, and label cutoff.
4. **Transport Time Only for Real Infrastructure**: Wall-clock time (real system time) is permitted strictly for transport-layer mechanics: HTTP request signing, transport headers, outbox dispatch timestamps, and in-memory cache expirations. Business logic and simulation state transitions must never observe real host wall-clock time.
5. **Conversion Helpers**:
   - `world_minute_to_ist(minute: int, world_start: date) -> datetime`: converts an integer minute offset from midnight of `world_start` into a timezone-aware `datetime` in `Asia/Kolkata` (+05:30). Negative minutes, booleans, and non-integer types are rejected with a `ValueError`.
   - `ist_to_world_minute(moment: datetime, world_start: date) -> int`: converts a timezone-aware `datetime` into integer minutes elapsed from midnight of `world_start`. Naive datetimes (without timezone info) are rejected with a `ValueError`.
6. **Hour-of-Day Features**: Diurnal rhythms, peak transaction hours, and feature bucketings are evaluated strictly in IST wall hours (0 to 23).
