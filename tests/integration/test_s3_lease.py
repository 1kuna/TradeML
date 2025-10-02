import os
import time
import uuid
import pytest

pytestmark = pytest.mark.skipif(
    not os.getenv("S3_ENDPOINT"),
    reason="Requires S3/MinIO endpoint in environment",
)


def test_s3_lease_acquire_renew_release():
    from data_layer.storage.s3_client import get_s3_client
    from data_layer.storage.lease_manager import LeaseManager

    s3 = get_s3_client()
    lease_name = f"it-{uuid.uuid4().hex}"

    l1 = LeaseManager(s3_client=s3, lease_seconds=2, renew_seconds=1)
    assert l1.acquire(lease_name, force=True) is True
    assert l1.is_held_by_me(lease_name) is True

    # Second manager should not acquire immediately
    l2 = LeaseManager(s3_client=s3, lease_seconds=2, renew_seconds=1)
    assert l2.acquire(lease_name) is False

    # Wait for expiration and force-steal
    time.sleep(3)
    assert l2.acquire(lease_name, force=True) is True
    assert l2.is_held_by_me(lease_name) is True

    # First manager can no longer release (not holder)
    assert l1.release(lease_name) is False

    # Cleanup
    assert l2.release(lease_name) is True

