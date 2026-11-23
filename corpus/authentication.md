# Authentication

## API keys

API keys are created in the Atlas console and are scoped to a single workspace. Keys are shown once at creation time and stored only as a salted hash. Keys that have not been used for 90 days are automatically disabled.

## Key rotation

Customers should rotate API keys at least every 180 days. Rotation is zero-downtime: a workspace can hold two active keys at once, so the new key can be deployed before the old key is revoked.

## Service-to-service authentication

Internal services authenticate to each other with short-lived mTLS certificates issued by the Atlas internal CA. Certificates are valid for 24 hours and are renewed automatically by the sidecar 6 hours before expiry.
