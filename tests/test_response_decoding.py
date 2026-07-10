import unittest

import requests

from src.publisher.run_publisher import _decode_response_text


def _response(body: bytes, content_type: str = "") -> requests.Response:
    r = requests.Response()
    r._content = body
    r.status_code = 200
    if content_type:
        r.headers["Content-Type"] = content_type
    return r


class ResponseDecodingTest(unittest.TestCase):
    def test_utf8_html_with_correct_header_remains_correct(self):
        html = "<html><body>Конспект занятия</body></html>"
        r = _response(html.encode("utf-8"), "text/html; charset=utf-8")

        self.assertIn("Конспект", _decode_response_text(r))

    def test_utf8_html_without_charset_remains_correct(self):
        html = "<html><body>Ребенок повторяет слово</body></html>"
        r = _response(html.encode("utf-8"), "text/html")

        self.assertIn("повторяет", _decode_response_text(r))

    def test_windows_1251_html_decodes_to_readable_russian(self):
        html = "<html><body>Логопедическая игра</body></html>"
        r = _response(html.encode("cp1251"), "text/html; charset=windows-1251")

        self.assertIn("Логопедическая", _decode_response_text(r))

    def test_windows_1251_html_without_charset_decodes_to_readable_russian(self):
        html = "<html><body>Логопедическая игра</body></html>"
        r = _response(html.encode("cp1251"), "text/html")

        decoded = _decode_response_text(r)

        self.assertIn("Логопедическая игра", decoded)
        self.assertNotIn("Ëîãî", decoded)

    def test_incorrect_declared_charset_does_not_choose_mojibake(self):
        html = "<html><body>Занятие: ребенок выбирает картинку</body></html>"
        r = _response(html.encode("utf-8"), "text/html; charset=iso-8859-1")

        decoded = _decode_response_text(r)

        self.assertIn("ребенок", decoded)
        self.assertNotIn("Ð", decoded)

    def test_normal_english_html_remains_unchanged(self):
        html = "<html><body>Use picture cards and ask the child to repeat the word.</body></html>"
        r = _response(html.encode("utf-8"), "text/html")

        self.assertEqual(_decode_response_text(r), html)


if __name__ == "__main__":
    unittest.main()
