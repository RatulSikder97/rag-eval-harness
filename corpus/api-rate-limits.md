# API Rate Limits

## Default limits

The public Atlas API allows 600 requests per minute per API key on the standard plan and 3,000 requests per minute on the enterprise plan. Limits are enforced with a token bucket that refills continuously.

## Rate limit responses

When a client exceeds its limit, the API responds with HTTP 429 and a `Retry-After` header giving the number of seconds to wait. Clients should back off exponentially with jitter and must not retry immediately.

## Raising limits

Enterprise customers can request a higher limit through their account manager. Temporary increases for planned events, such as a data migration, are granted for up to 7 days.
