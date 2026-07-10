# Security Policy

## Supported versions

The project is pre-1.0 and evolves on the latest release. Security fixes are applied
to the most recent tagged release.

| Version | Supported |
|---|---|
| latest `0.x` | yes |
| older `0.x` | no |

## Reporting a vulnerability

Please do not open a public issue for a security vulnerability.

Report it privately by email to felipe.santibanez@accenture.com, or use GitHub's
private "Report a vulnerability" flow under the repository Security tab. Include a
description, reproduction steps, affected version, and any relevant logs or a proof of
concept.

You can expect an acknowledgement within a few business days and a coordinated
disclosure once a fix is available.

## Handling secrets

Never commit secrets (tokens, keys, credentials) to this repository. Publishing to
PyPI uses OIDC Trusted Publishing, so no long-lived token is stored anywhere in the
repository or its CI configuration.
