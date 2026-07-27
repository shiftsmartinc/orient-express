import pytest

from orient_express.utils.image_processor import UnsafeUrlError, validate_url

PUBLIC_ADDR_INFO = [(2, 1, 6, "", ("142.250.68.78", 0))]


@pytest.mark.parametrize(
    "url",
    [
        "http://metadata.google.internal/computeMetadata/v1/",
        "http://169.254.169.254/computeMetadata/v1/",
        "http://127.0.0.1:8080/v1/models",
        "http://localhost/foo.jpg",
        "http://10.0.0.5/foo.jpg",
        "http://192.168.1.1/foo.jpg",
        "http://[::1]/foo.jpg",
    ],
)
def test_validate_url_blocks_internal_addresses(url):
    with pytest.raises(UnsafeUrlError):
        validate_url(url)


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "gopher://example.com/",
        "ftp://example.com/foo.jpg",
        "http://",
    ],
)
def test_validate_url_blocks_bad_schemes_and_hosts(url):
    with pytest.raises(UnsafeUrlError):
        validate_url(url)


def test_validate_url_allows_public_host(monkeypatch):
    monkeypatch.setattr(
        "orient_express.utils.image_processor.socket.getaddrinfo",
        lambda host, port: PUBLIC_ADDR_INFO,
    )
    validate_url("https://storage.googleapis.com/some-bucket/image.jpg")


def test_validate_url_blocks_host_resolving_to_metadata_ip(monkeypatch):
    # DNS-level bypass: a public-looking hostname pointing at the metadata server
    monkeypatch.setattr(
        "orient_express.utils.image_processor.socket.getaddrinfo",
        lambda host, port: [(2, 1, 6, "", ("169.254.169.254", 0))],
    )
    with pytest.raises(UnsafeUrlError):
        validate_url("https://innocent-looking-domain.example.com/image.jpg")


def _oriented_image(orientation):
    from PIL import Image

    img = Image.new("RGB", (2, 1))
    img.putpixel((0, 0), (255, 0, 0))  # red left, black right
    img.getexif()[0x0112] = orientation
    return img


def test_fix_rotation_rotates_orientation_6():
    from orient_express.utils.image_processor import fix_rotation

    fixed = fix_rotation(_oriented_image(6))
    assert fixed.size == (1, 2)  # upright: 90-degree turn swaps dimensions


def test_fix_rotation_mirrors_orientation_2():
    # mirrored orientations (2/4/5/7) were passed through untouched by the
    # old 3/6/8-only rotation; the full transform must flip them
    from orient_express.utils.image_processor import fix_rotation

    fixed = fix_rotation(_oriented_image(2))
    assert fixed.getpixel((1, 0)) == (255, 0, 0)  # horizontally flipped


def test_fix_rotation_noop_without_orientation():
    from PIL import Image

    from orient_express.utils.image_processor import fix_rotation

    img = Image.new("RGB", (2, 1))
    img.putpixel((0, 0), (255, 0, 0))
    fixed = fix_rotation(img)
    assert fixed.size == (2, 1)
    assert fixed.getpixel((0, 0)) == (255, 0, 0)
