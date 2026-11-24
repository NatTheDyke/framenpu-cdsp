# Security and hardware boundary

framenpu-cdsp runs trusted native supervisor code on a firmware-specific DSP.
FJOB checksums detect corruption, not authenticity. Only run native jobs from
trusted authors. The library does not isolate jobs from other DSP clients.

HMX uses root /dev/mem access. Exact firmware hashes are checked before writes.
Unknown firmware is unsupported. There is no automatic service restart or
production cancellation; the stock lease may expire if the host is delayed.

## Reporting

Use GitHub's private "Report a vulnerability" feature if enabled. If unavailable,
open an issue requesting a private contact without posting sensitive details.

For ordinary bugs, include package/Python versions, compatibility results and a
minimal sanitized reproducer. Omit credentials, serial numbers, raw captures
and firmware dumps.

The supported research line is 0.1.x on the documented exact firmware profile.
The project has not undergone an independent security audit or sustained
eye-tracking coexistence assessment.
