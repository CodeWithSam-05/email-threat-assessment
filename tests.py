from django.test import TestCase

from views import analyze_pdf_indicators, normalize_pdf_text


class PdfIndicatorTests(TestCase):
    def test_normalizes_obfuscated_urls(self):
        self.assertEqual(
            normalize_pdf_text("Visit hxxps://secure[dot]example[.]com"),
            "Visit https://secure.example.com",
        )

    def test_extracts_structured_indicators(self):
        report = analyze_pdf_indicators(
            "Email security@example.com and visit hxxp://login[dot]example.com. "
            "Open invoice.zip immediately and pay the payment."
        )

        indicator_types = {indicator["type"] for indicator in report["indicators"]}
        self.assertIn("url", indicator_types)
        self.assertIn("domain", indicator_types)
        self.assertIn("email_address", indicator_types)
        self.assertIn("attachment", indicator_types)
        self.assertIn("urgency", indicator_types)
        self.assertIn("payment_request", indicator_types)
        self.assertEqual(report["verdict"], "suspicious")

    def test_detects_obfuscated_commands_as_malicious(self):
        report = analyze_pdf_indicators(
            "Run p o w e r s h e l l and disable anti virus, then execute payload."
        )

        self.assertEqual(report["verdict"], "malicious")
        self.assertGreater(report["indicator_counts"]["critical"], 0)
        self.assertTrue(
            any(
                indicator["type"] == "obfuscated_command"
                for indicator in report["indicators"]
            )
        )

    def test_safe_text_has_no_threat_signals(self):
        report = analyze_pdf_indicators("Meeting notes for the engineering team.")

        self.assertEqual(report["verdict"], "safe")
        self.assertEqual(report["indicator_counts"]["total"], 0)
