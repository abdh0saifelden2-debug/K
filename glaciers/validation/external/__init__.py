"""Real-data loaders for the §V validation pipeline.

These are **honest stubs**: the development VM has no outbound network access to
the NSIDC / Earthdata / tide-model archives, and the datasets require
authentication plus multi-GB downloads.  Each loader documents its data source,
authentication requirement, expected on-disk layout, and the exact fields it must
return so the §V.1 / §V.2 validators can run unchanged once real data are
provided locally.

Calling a loader without the data present raises ``DataUnavailableError`` with a
copy-pasteable provisioning hint rather than silently fabricating data.
"""


class DataUnavailableError(RuntimeError):
    """Raised when a real dataset is not present / not reachable locally."""
