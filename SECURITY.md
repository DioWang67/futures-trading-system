# Security Notes

## Secrets

- Keep `SHIOAJI_API_KEY`, `SHIOAJI_SECRET_KEY`, `WEBHOOK_SECRET`, and `ADMIN_SECRET` only in `.env`.
- Do not hardcode secrets in Python files, notebooks, screenshots, or chat logs.
- On Windows, restrict `.env` so only your user account, `Administrators`, and `SYSTEM` can read or modify it.

## Webhook Authentication

- Production webhook requests should include both `X-Timestamp` and `X-Signature`.
- The signature format is `HMAC-SHA256(secret, "<timestamp>.<raw_body>")`.
- `X-Webhook-Secret` is a legacy fallback and should remain disabled unless you explicitly set `WEBHOOK_ALLOW_LEGACY_SECRET_HEADER=true`.

## Control Plane

- `/risk`, `/risk/resume`, `/trades/*`, and `/positions` require `X-Admin-Secret`.
- Set `ADMIN_SECRET` in `.env` before starting the service.

## Operational Hygiene

- Rotate keys immediately if they were pasted into chat, screenshots, or shared documents.
- Use IP restrictions in the broker/API portal whenever available.
- Keep test keys and live-trading keys separated.
