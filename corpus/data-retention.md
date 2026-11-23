# Data Retention

## Application logs

Application logs are retained for 30 days in hot storage and then archived to cold storage for a total of 400 days. Logs must never contain passwords, API tokens or full payment card numbers; the log pipeline redacts known token formats automatically.

## Customer data deletion

When a customer closes their account, their personal data is deleted from primary databases within 30 days. Backups containing that data age out within a further 35 days, so complete erasure takes at most 65 days.

## Metrics and traces

Metrics are retained for 13 months at one-minute resolution. Distributed traces are sampled at 10% and retained for 14 days.
