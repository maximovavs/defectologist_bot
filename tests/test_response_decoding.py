import unittest

import requests

from src.publisher.run_publisher import _decode_response_text


def _response(body: bytes, content_type: str = "") -> requests.Response:
    response = requests.Response()
    response._content = body
    response.status_code = 200
    if content_type:
        response.headers["Content-Type"] = content_type
    return response


class ResponseDecodingTest(unittest.TestCase):
    def test_utf8_without_charset_is_decoded_as_utf8(self):
        html = "<html><body>Ребенок повторяет слово</body></html>"
        response = _response(html.encode("utf-8"), "text/html")

        self.assertEqual(_decode_response_text(response), html)

    def test_windows_1251_without_charset_is_decoded_readably(self):
        html = "<html><body>Логопедическая игра</body></html>"
        response = _response(html.encode("cp1251"), "text/html")

        decoded = _decode_response_text(response)

        self.assertIn("Логопедическая игра", decoded)
        self.assertNotIn("Ëîãî", decoded)

    def test_implicit_iso8859_does_not_win_over_utf8(self):
        html = "<html><body>Занятие: ребенок выбирает картинку</body></html>"
        response = _response(html.encode("utf-8"), "text/html")
        response.encoding = "iso-8859-1"

        decoded = _decode_response_text(response)

        self.assertIn("ребенок", decoded)
        self.assertNotIn("Ð", decoded)

    def test_explicit_windows_1251_charset_is_honored(self):
        html = "<html><body>Занятие по артикуляции</body></html>"
        response = _response(html.encode("cp1251"), "text/html; charset=windows-1251")

        self.assertEqual(_decode_response_text(response), html)

    def test_plain_english_text_has_no_mojibake(self):
        html = "<html><body>Use picture cards and ask the child to repeat the word.</body></html>"
        response = _response(html.encode("utf-8"), "text/html")

        self.assertEqual(_decode_response_text(response), html)


if __name__ == "__main__":
    unittest.main()
