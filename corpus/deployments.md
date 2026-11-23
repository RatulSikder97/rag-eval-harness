# Deployments

## Pipeline

Every merge to `main` builds an immutable container image tagged with the git commit SHA. The image is deployed to staging automatically. Production deploys are triggered from the release dashboard and require a green staging smoke test.

## Canary releases

Production rollouts start as a canary that receives 5% of traffic for 15 minutes. The canary is promoted automatically if its error rate stays within 0.5 percentage points of the baseline fleet; otherwise it is rolled back without human intervention.

## Deploy freezes

Production deploys are frozen from Friday 16:00 UTC until Monday 08:00 UTC, and during company-wide freeze windows announced in #atlas-releases. Emergency fixes during a freeze require approval from the on-call incident commander.

## Rollbacks

Any engineer may roll back a production deploy at any time with `atlas deploy rollback <service>`. Rolling back never requires approval; rolling forward during an incident does.
