import io
import typing
from pathlib import Path

import pytest

import lcc.io

if typing.TYPE_CHECKING:  # pragma: no cover
    from pytest_mock import MockerFixture

# ---------------------------------------------------------------------------


try:
    import warcio
    import warcio.recordbuilder
    import warcio.warcwriter
except ImportError:
    pytest.skip(
        "skipping warcio-only tests since 'warcio' not installed",
        allow_module_level=True,
    )


def test_WARC_SUPPORTED():
    assert lcc.io.WARC_SUPPORTED is True

    # TODO: can we even test this?, do some monkeypatching?
    # https://stackoverflow.com/a/51048604/9360161 seems to fail spectacularly


# ---------------------------------------------------------------------------


# from: https://github.com/webrecorder/pywb/blob/main/sample_archive/warcs/example.warc [9b8b4d83882b126dd061492069695535b789aad3]
EXAMPLE_WARC10_CONTENT = (
    # warcinfo = len=260
    b"WARC/1.0\r\nWARC-Type: warcinfo\r\nWARC-Record-ID: <urn:uuid:fbd6cf0a-6160-4550-b343-12188dc05234>\r\nWARC-Date: 2014-01-03T03:03:22Z\r\nContent-Length: 196\r\nContent-Type: application/warc-fields\r\nWARC-Filename: live-20140103030321-wwwb-app5.us.archive.org.warc.gz\r\n\r\n"
    # warcinfo->content = len=196
    b"software: LiveWeb Warc Writer 1.0\r\nhost: wwwb-app5.us.archive.org\r\nisPartOf: liveweb\r\nformat: WARC file version 1.0\r\nconformsTo: http://bibnum.bnf.fr/WARC/WARC_ISO_28500_version1_latestdraft.pdf\r\n"
    # len=456
    b"\r\n\r\n"
    # response = len=377 (totallen=1991)
    b"WARC/1.0\r\nWARC-Type: response\r\nWARC-Record-ID: <urn:uuid:6d058047-ede2-4a13-be79-90c17c631dd4>\r\nWARC-Date: 2014-01-03T03:03:21Z\r\nContent-Length: 1610\r\nContent-Type: application/http; msgtype=response\r\nWARC-Payload-Digest: sha1:B2LTWWPUOYAH7UIPQ7ZUPQ4VMBSVC36A\r\nWARC-Target-URI: http://example.com?example=1\r\nWARC-Warcinfo-ID: <urn:uuid:fbd6cf0a-6160-4550-b343-12188dc05234>\r\n\r\n"
    # len=340
    b'HTTP/1.1 200 OK\r\nAccept-Ranges: bytes\r\nCache-Control: max-age=604800\r\nContent-Type: text/html\r\nDate: Fri, 03 Jan 2014 03:03:21 GMT\r\nEtag: "359670651"\r\nExpires: Fri, 10 Jan 2014 03:03:21 GMT\r\nLast-Modified: Fri, 09 Aug 2013 23:54:35 GMT\r\nServer: ECS (sjc/4FCE)\r\nX-Cache: HIT\r\nx-ec-custom-error: 1\r\nContent-Length: 1270\r\nConnection: close\r\n\r\n'
    # len=1270
    b'<!doctype html>\n<html>\n<head>\n    <title>Example Domain</title>\n\n    <meta charset="utf-8" />\n    <meta http-equiv="Content-type" content="text/html; charset=utf-8" />\n    <meta name="viewport" content="width=device-width, initial-scale=1" />\n    <style type="text/css">\n    body {\n        background-color: #f0f0f2;\n        margin: 0;\n        padding: 0;\n        font-family: "Open Sans", "Helvetica Neue", Helvetica, Arial, sans-serif;\n        \n    }\n    div {\n        width: 600px;\n        margin: 5em auto;\n        padding: 50px;\n        background-color: #fff;\n        border-radius: 1em;\n    }\n    a:link, a:visited {\n        color: #38488f;\n        text-decoration: none;\n    }\n    @media (max-width: 700px) {\n        body {\n            background-color: #fff;\n        }\n        div {\n            width: auto;\n            margin: 0 auto;\n            border-radius: 0;\n            padding: 1em;\n        }\n    }\n    </style>    \n</head>\n\n<body>\n<div>\n    <h1>Example Domain</h1>\n    <p>This domain is established to be used for illustrative examples in documents. You may use this\n    domain in examples without prior coordination or asking for permission.</p>\n    <p><a href="http://www.iana.org/domains/example">More information...</a></p>\n</div>\n</body>\n</html>\n'
    b"\r\n\r\n"
    # request
    b"WARC/1.0\r\nWARC-Type: request\r\nWARC-Record-ID: <urn:uuid:9a3ffea5-9556-4790-a6bf-c15231fd6b97>\r\nWARC-Date: 2014-01-03T03:03:21Z\r\nContent-Length: 323\r\nContent-Type: application/http; msgtype=request\r\nWARC-Concurrent-To: <urn:uuid:6d058047-ede2-4a13-be79-90c17c631dd4>\r\nWARC-Target-URI: http://example.com?example=1\r\nWARC-Warcinfo-ID: <urn:uuid:fbd6cf0a-6160-4550-b343-12188dc05234>\r\n\r\n"
    b"GET /?example=1 HTTP/1.1\r\nConnection: close\r\nAccept: text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8\r\nAccept-Language: en-US,en;q=0.8\r\nUser-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/31.0.1650.57 Safari/537.36 (via Wayback Save Page)\r\nHost: example.com\r\n\r\n\r\n"
    # revisit
    b"WARC/1.0\r\nWARC-Type: revisit\r\nWARC-Record-ID: <urn:uuid:3619f5b0-d967-44be-8f24-762098d427c4>\r\nWARC-Date: 2014-01-03T03:03:41Z\r\nContent-Length: 340\r\nContent-Type: application/http; msgtype=response\r\nWARC-Payload-Digest: sha1:B2LTWWPUOYAH7UIPQ7ZUPQ4VMBSVC36A\r\nWARC-Target-URI: http://example.com?example=1\r\nWARC-Warcinfo-ID: <urn:uuid:fbd6cf0a-6160-4550-b343-12188dc05234>\r\nWARC-Profile: http://netpreserve.org/warc/0.18/revisit/identical-payload-digest\r\nWARC-Refers-To-Target-URI: http://example.com?example=1\r\nWARC-Refers-To-Date: 2014-01-03T03:03:21Z\r\n\r\n"
    b'HTTP/1.1 200 OK\r\nAccept-Ranges: bytes\r\nCache-Control: max-age=604800\r\nContent-Type: text/html\r\nDate: Fri, 03 Jan 2014 03:03:41 GMT\r\nEtag: "359670651"\r\nExpires: Fri, 10 Jan 2014 03:03:41 GMT\r\nLast-Modified: Fri, 09 Aug 2013 23:54:35 GMT\r\nServer: ECS (sjc/4FCE)\r\nX-Cache: HIT\r\nx-ec-custom-error: 1\r\nContent-Length: 1270\r\nConnection: close\r\n\r\n'
    b"\r\n\r\n"
    # request
    b"WARC/1.0\r\nWARC-Type: request\r\nWARC-Record-ID: <urn:uuid:c59f3330-b241-4fca-8513-d687cd85bcfb>\r\nWARC-Date: 2014-01-03T03:03:41Z\r\nContent-Length: 320\r\nContent-Type: application/http; msgtype=request\r\nWARC-Concurrent-To: <urn:uuid:3619f5b0-d967-44be-8f24-762098d427c4>\r\nWARC-Target-URI: http://example.com?example=1\r\nWARC-Warcinfo-ID: <urn:uuid:fbd6cf0a-6160-4550-b343-12188dc05234>\r\n\r\n"
    b"GET /?example=1 HTTP/1.1\r\nConnection: close\r\nAccept: text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8\r\nAccept-Language: en-US,en;q=0.8\r\nUser-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/31.0.1650.57 Safari/537.36 (via Wayback Save Page)\r\nHost: example.com\r\n\r\n\r\n"
    # response
    b"WARC/1.0\r\nWARC-Type: response\r\nWARC-Record-ID: <urn:uuid:1d673b2a-c593-402e-8973-3950d0bc6163>\r\nWARC-Date: 2014-01-28T05:15:39Z\r\nContent-Length: 471\r\nContent-Type: application/http; msgtype=response\r\nWARC-Payload-Digest: sha1:JZ622UA23G5ZU6Y3XAKH4LINONUEICEG\r\nWARC-Target-URI: http://www.iana.org/domains/example\r\nWARC-Warcinfo-ID: <urn:uuid:e9f7f74b-0280-47fd-99bc-f00f1a570a46>\r\n\r\n"
    b"HTTP/1.1 302 Found\r\nServer: Apache\r\nLocation: /domains/reserved\r\nContent-Type: text/html; charset=iso-8859-1\r\nContent-Length: 201\r\nAccept-Ranges: bytes\r\nDate: Tue, 28 Jan 2014 05:15:39 GMT\r\nX-Varnish: 774901408 774900872\r\nAge: 80\r\nVia: 1.1 varnish\r\nConnection: close\r\n\r\n"
    b'<!DOCTYPE HTML PUBLIC "-//IETF//DTD HTML 2.0//EN">\n<html><head>\n<title>302 Found</title>\n</head><body>\n<h1>Found</h1>\n<p>The document has moved <a href="/domains/reserved">here</a>.</p>\n</body></html>\n'
    b"\r\n\r\n"
)


# ---------------------------------------------------------------------------


def test_WARCDocMetadata_from_warcio_headers():
    headers = warcio.StatusAndHeaders("TEST/1.0", [])
    headers.add_header("WARC-Target-URI", "target")
    headers.add_header("WARC-Date", "0000-00-00Trest...")

    meta = lcc.io.WARCDocMetadata.from_warcio_headers(headers)
    assert meta.date == "0000-00-00"
    assert meta.location == "target"
    with pytest.raises(AttributeError):
        meta.recordID

    headers.add_header("WARC-Record-ID", "id")
    meta = lcc.io.WARCDocMetadata.from_warcio_headers(headers)
    assert meta.recordID == "id"


def test_parse_warc_docs_iter(tmp_path: Path):
    fn_warc = tmp_path / "data.warc"
    fn_warc.write_bytes(EXAMPLE_WARC10_CONTENT)

    doc_iter = lcc.io.parse_warc_docs_iter(fn_warc, add_content=True)
    docs = list(doc_iter)
    assert len(docs) == 2
    doc = docs[0]
    assert isinstance(doc.meta, lcc.io.WARCDocMetadata)
    assert doc.meta.date == "2014-01-03"
    assert doc.meta.location == "http://example.com?example=1"
    assert doc.meta.recordID == "<urn:uuid:6d058047-ede2-4a13-be79-90c17c631dd4>"
    assert doc.offset == 460
    assert doc.length == 1987
    assert isinstance(doc.content, bytes)
    assert len(doc.content) == 340 + 1270

    doc_iter = lcc.io.parse_warc_docs_iter(
        fn_warc, add_content=False, record_types=("request", "response")
    )
    assert len(list(doc_iter)) == 4
    doc_iter = lcc.io.parse_warc_docs_iter(
        fn_warc, add_content=False, record_types=None
    )
    assert len(list(doc_iter)) == 6


def test__build_warc_record():
    buf = io.BytesIO()
    writer = warcio.WARCWriter(buf, warc_version="WARC/1.1", gzip=False)

    meta = lcc.io.DocMetadata(location="url", date="2020-02-20")
    doc = lcc.io.DocAndMeta(meta=meta, content=b"12\n3")

    record = lcc.io._build_warc_record(writer, doc, "test", allow_empty=True)
    assert record is not None
    writer.write_record(record)

    content = buf.getvalue()
    assert content[0:43] == b"WARC/1.1\r\n" b"WARC-Date: 2020-02-20T00:00:00Z\r\n"
    assert content[43:108].startswith(b"WARC-Record-ID: <urn:uuid:")
    assert content[108:147] == b"WARC-Type: test\r\n" b"WARC-Target-URI: url\r\n"
    assert b"WARC-Payload-Digest: sha1:" in content[147:265]
    assert b"WARC-Block-Digest: sha1:" in content[147:265]
    assert (
        content[265:312] == b"Content-Type: text/plain\r\n" b"Content-Length: 4\r\n\r\n"
    )
    assert content[312:316] == b"12\n3"
    assert content[316:320] == b"\r\n\r\n"

    builder = warcio.recordbuilder.RecordBuilder(warc_version="WARC/1.1")
    meta = lcc.io.DocMetadata(location="url", date="2020-02-20")
    doc = lcc.io.DocAndMeta(meta=meta, content=None)
    record = lcc.io._build_warc_record(writer, doc, "test", allow_empty=True)
    assert record is not None
    assert record.content_stream().read() == b""
    record = lcc.io._build_warc_record(writer, doc, "test", allow_empty=False)
    assert record is None


def test_write_warc_docs_iter(tmp_path: Path, mocker: "MockerFixture"):
    fn_out = tmp_path / "out.warc"

    meta = lcc.io.DocMetadata(location="url", date="2020-02-20")
    doc1 = lcc.io.DocAndMeta(meta=meta, content=b"12\n3")
    doc2 = lcc.io.DocAndMeta(meta=meta, content=b"")
    doc3 = lcc.io.DocAndMeta(meta=meta, content=None)

    mock_write = mocker.patch("lcc.io.warcio.WARCWriter.write_record")

    lcc.io.write_warc_docs_iter(
        str(fn_out), [doc1, doc2, doc3], record_type="test", write_empty_records=False
    )
    assert mock_write.call_count == 2
    mock_write.reset_mock()

    lcc.io.write_warc_docs_iter(
        str(fn_out), [doc1, doc2, doc3], record_type="test", write_empty_records=True
    )
    assert mock_write.call_count == 3


# ---------------------------------------------------------------------------
