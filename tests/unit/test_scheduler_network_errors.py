from scripts.scheduler.per_vendor import _is_network_error


def test_is_network_error_matches_ssl_variants():
    assert _is_network_error("SSLError(SSLZeroReturnError(6, 'TLS/SSL connection has been closed (EOF) (_ssl.c:992)'))")
    assert _is_network_error("SSLEOFError(8, 'EOF occurred in violation of protocol (_ssl.c:2393)')")
    assert _is_network_error("Connection broken: IncompleteRead(259 bytes read, 366 more expected)")
    assert _is_network_error("no route to host")


def test_is_network_error_ignores_non_network_failures():
    assert _is_network_error("Temporary failure in name resolution")
    assert not _is_network_error("HTTP 401 unauthorized")
