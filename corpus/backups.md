# Backups and Restore

## Backup schedule

Primary Postgres databases take a full snapshot every 24 hours and stream write-ahead logs continuously, which allows point-in-time recovery to any second within the last 35 days. Snapshots are copied to a second cloud region.

## Recovery objectives

The recovery point objective (RPO) for primary databases is 5 minutes and the recovery time objective (RTO) is 4 hours.

## Restore drills

A restore drill is run every quarter: a randomly chosen production database is restored into an isolated environment and verified with row-count and checksum comparisons. A drill that misses the RTO is treated as a SEV3 incident.
