# UPI Failure Sources and Research Verification

This document records the empirical verification of published sources for UPI network downtime, failure domains, and degradation parameters in Level 11 (sub-phase 11B-2a).

## Verification Summary Table

| Claim | URL | Page Date | Quote | FACT or NOT VERIFIED | Difference / Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Unscheduled downtime of 95 minutes in March 2025 | https://economictimes.indiatimes.com/tech/technology/ettech-explainer-why-do-upi-payments-fail-and-what-is-npci-doing-to-check-these-failures/articleshow/120346323.cms | 2025-04-16 | "the unscheduled downtime for UPI was 95 minutes in March." | FACT | None. Exact match. |
| Success rate below 80% for about five hours on 12 April | https://economictimes.indiatimes.com/tech/technology/ettech-explainer-why-do-upi-payments-fail-and-what-is-npci-doing-to-check-these-failures/articleshow/120346323.cms | 2025-04-16 | "on April 12, the service was operating at less than 80% success rate for around five hours." | FACT | None. Exact match. |
| Banks operate at overall UPI success rate of 85-95% | https://economictimes.indiatimes.com/tech/technology/ettech-explainer-why-do-upi-payments-fail-and-what-is-npci-doing-to-check-these-failures/articleshow/120346323.cms | 2025-04-16 | "Banks generally operate at an overall success rate of between 85% and 95% on UPI." | FACT | None. Exact match. |
| Transaction declines less than 1%; most failures business-related | https://economictimes.indiatimes.com/tech/technology/ettech-explainer-why-do-upi-payments-fail-and-what-is-npci-doing-to-check-these-failures/articleshow/120346323.cms | 2025-04-16 | "Typically, transaction declines are less than 1%." | FACT | Confirms declines < 1%. Page also states: "Most of the transaction failures happen because of business issues". |
| 31 downtime instances, more than 47 hours, May 2024 | https://economictimes.com/industry/banking/finance/banking/inadequate-tech-at-banks-to-blame-for-upi-downtimes/articleshow/110806141.cms | 2024-06-08 | "In May 2024 alone, there were 31 downtime instances from various banks that led to payment gateways being shut for more than 47 hours" | FACT | None. Exact match (data cited from NPCI). |
| Outages on March 26, March 31, April 2, April 12 (2025) | https://www.business-standard.com/industry/banking/npci-meets-major-banks-upi-apps-to-discuss-recent-outages-faced-by-network-125041501236_1.html | 2025-04-16 | "NPCI informed banks and third-party app providers about recent outages and stated it is conducting a root cause analysis of April 12 outage." | NOT VERIFIED | Page is behind a subscription paywall after paragraph 1; April 12 is mentioned, but March 26, March 31, and April 2 are not present in accessible text. |
| Success rate dropped to 50-60 per cent | https://www.rediff.com/business/report/tech-failures-behind-upi-outages/20250416.htm | 2025-04-16 | "The success rate of UPI transactions dropped to 50-60 per cent, a source said" | FACT | None. Exact match (also confirms March 26, March 31, and April 2 disruptions in body). |
| Technical decline rate 0.7-0.8 per cent | https://www.zeebiz.com/economy-infra/news-only-08-of-upi-transactions-face-technical-declines-now-npci-327217 | 2024-11-20 | "The technical decline rates have reduced to 0.7-0.8 per cent from as high as 8-10 per cent in 2016," | FACT | Headline reads "0.8%", and body text confirms "0.7-0.8 per cent". |
| Publishes incident or uptime figures you can read | https://www.npci.org.in/product/upi/product-statistics | Live portal | "NPCI UPI System Uptime... Member Bank Downtime/Incidents report published monthly" | FACT | Page loads and hosts active "Uptime" and "Downtime/Incidents" monthly metric tables. |

## Figures that must stay assumptions

1. Customer wrong-PIN failure rate: Published sources group wrong PINs into broad "business declines" alongside insufficient balance without isolating an individual wrong-PIN percentage.
2. Network drop and client app crash rates: Official reports state overall technical declines (0.7-0.8%) without breaking out handset network drops versus server/intermediary drops.
3. Per-bank downtime incidence and duration distribution: Sources document aggregate multi-bank monthly downtime (31 instances, >47 hours across various banks in May 2024), but the exact per-bank incident frequency, duration range (e.g., ~1.5 hours), and world-level schedule must remain assumptions.
4. NPCI incident duration bounds: While specific historic incidents are documented (95 minutes in March 2025; ~5 hours on April 12, 2025), the general simulated duration parameter range (e.g., 95-180 minutes) must remain an assumption.
5. Success rate degradation floor during bank-specific downtime: NPCI network degradation drops to 50-60% or <80% during central incidents, but specific degraded success rates for individual bank downtime windows are not published separately.
