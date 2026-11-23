# Feature Flags

## When to use a flag

Any user-visible change that cannot be safely rolled back by a deploy must ship behind a feature flag. Flags are managed in the Atlas flag service and evaluated server-side.

## Flag lifecycle

Every flag has an owner and an expiry date. Release flags expire 30 days after they reach 100% rollout, and the flag service opens a cleanup ticket for the owner on expiry. Permanent operational flags, such as kill switches, must be tagged `ops` and are exempt from expiry.

## Rollout percentages

Percentage rollouts are sticky per account: an account that sees a feature at 10% keeps seeing it as the percentage increases. The recommended ramp is 1%, 10%, 50%, then 100%, waiting at least one hour between steps while watching error rates.
