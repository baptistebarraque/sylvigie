import hashlib

import pytest
import requests

from sylvigie.ingest import ancillary
from sylvigie.ingest.ancillary import download_file, fetch_wfs

L93 = {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::2154"}}


class FakeResponse:
    def __init__(self, payload=None, content=b"", status_code=200, text=""):
        self._payload = payload
        self._content = content
        self.status_code = status_code
        self.text = text

    @property
    def ok(self):
        return self.status_code < 400

    def raise_for_status(self):
        if not self.ok:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload

    def iter_content(self, chunk_size):
        yield self._content

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeWfs:
    """Serves `n` point features in pages, like the Géoplateforme WFS."""

    def __init__(self, n, crs=L93):
        self.features = [
            {
                "type": "Feature",
                "properties": {"id": i},
                "geometry": {"type": "Point", "coordinates": [780000 + i, 6680000]},
            }
            for i in range(n)
        ]
        self.crs = crs
        self.calls = []

    def get(self, url, params, timeout):
        self.calls.append(dict(params))
        start, count = params["STARTINDEX"], params["COUNT"]
        return FakeResponse(
            {
                "type": "FeatureCollection",
                "features": self.features[start : start + count],
                "numberMatched": len(self.features),
                "crs": self.crs,
            }
        )


def test_fetch_wfs_pages_through_all_features():
    server = FakeWfs(n=12)

    gdf = fetch_wfs(
        "layer", "EPSG:2154", "id", bbox=(0, 0, 1, 1), session=server, page_size=5
    )

    assert list(gdf["id"]) == list(range(12))
    assert gdf.crs.to_epsg() == 2154
    assert [c["STARTINDEX"] for c in server.calls] == [0, 5, 10]
    assert server.calls[0]["BBOX"] == "0,0,1,1,EPSG:2154"


def test_fetch_wfs_rejects_unexpected_crs():
    server = FakeWfs(n=3, crs={"type": "name", "properties": {"name": "EPSG:3857"}})

    with pytest.raises(RuntimeError, match="EPSG:3857"):
        fetch_wfs("layer", "EPSG:2154", "id", session=server)


class FlakyWfs(FakeWfs):
    """Fails with the Géoplateforme 'pool exhausted' error before answering."""

    def __init__(self, n, failures, status_code=400, text=None):
        super().__init__(n)
        self.failures = failures
        self.status_code = status_code
        self.text = text or "Unable to obtain connection: pool error"

    def get(self, url, params, timeout):
        if self.failures:
            self.failures -= 1
            return FakeResponse(status_code=self.status_code, text=self.text)
        return super().get(url, params, timeout)


@pytest.fixture
def no_sleep(monkeypatch):
    waits = []
    monkeypatch.setattr(ancillary, "_sleep", waits.append)
    return waits


def test_fetch_wfs_retries_transient_errors(no_sleep):
    server = FlakyWfs(n=3, failures=2)

    gdf = fetch_wfs("layer", "EPSG:2154", "id", session=server)

    assert len(gdf) == 3
    assert no_sleep == [5.0, 10.0]


def test_fetch_wfs_gives_up_after_max_attempts(no_sleep):
    server = FlakyWfs(n=3, failures=100, status_code=503, text="")

    with pytest.raises(RuntimeError, match="still failing"):
        fetch_wfs("layer", "EPSG:2154", "id", session=server)
    assert len(no_sleep) == ancillary.WFS_ATTEMPTS - 1


def test_fetch_wfs_does_not_retry_real_client_errors(no_sleep):
    server = FlakyWfs(n=3, failures=1, text="Unknown property 'foo'")

    with pytest.raises(requests.HTTPError):
        fetch_wfs("layer", "EPSG:2154", "id", session=server)
    assert no_sleep == []


def test_fetch_wfs_rejects_bbox_with_filter():
    with pytest.raises(ValueError):
        fetch_wfs("layer", "EPSG:2154", "id", bbox=(0, 0, 1, 1), cql_filter="id=1")


class FakeDownload:
    def __init__(self, content):
        self.content = content
        self.calls = 0

    def get(self, url, stream, timeout):
        self.calls += 1
        return FakeResponse(content=self.content)


def test_download_file_checks_md5_and_caches(tmp_path):
    content = b"clear-cut dates"
    md5 = hashlib.md5(content).hexdigest()
    session = FakeDownload(content)
    dest = tmp_path / "raw" / "file.tif"

    download_file("https://example.org/file", dest, md5, session=session)
    download_file("https://example.org/file", dest, md5, session=session)

    assert dest.read_bytes() == content
    assert session.calls == 1


def test_download_file_rejects_corrupted_file(tmp_path):
    session = FakeDownload(b"truncated")
    dest = tmp_path / "file.tif"

    with pytest.raises(ValueError, match="md5 mismatch"):
        download_file("https://example.org/file", dest, "0" * 32, session=session)
    assert not dest.exists()
    assert not dest.with_name("file.tif.part").exists()
