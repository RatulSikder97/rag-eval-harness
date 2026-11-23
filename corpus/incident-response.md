# Incident Response

Atlas classifies incidents by customer impact, not by the component that failed.

## Severity levels

- SEV1: a full outage of checkout, login or the public API, or any confirmed data loss. SEV1 pages the incident commander and the on-call engineer immediately.
- SEV2: a degraded core flow, such as p95 API latency above 2 seconds for more than 10 minutes, or an outage of a single region.
- SEV3: a non-core feature is broken or a workaround exists. SEV3 incidents are handled in business hours.

## Roles

Every SEV1 and SEV2 incident has exactly one incident commander (IC). The IC coordinates and makes decisions; the IC does not debug. A separate communications lead posts status updates to the #atlas-status channel every 30 minutes until the incident is resolved.

## Postmortems

A blameless postmortem is required for every SEV1 and SEV2 incident. The draft is due within 5 business days of resolution and is reviewed in the weekly reliability review. Action items must have an owner and a due date, and are tracked with the `postmortem` label.
