"""Static scope wording only; no browser, image review or model execution."""
import importlib.util
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]


class ExpansionPageTests(unittest.TestCase):
    def tool(self):
        path=ROOT/'tools/cleaning_expansion_page.py'
        spec=importlib.util.spec_from_file_location('expansion_page',path)
        m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

    def test_scope_text_updated_without_changing_embedded_candidate_data(self):
        m=self.tool()
        template=(ROOT/'tools/oof_label_cleaning_review.template.html').read_text(encoding='utf-8')
        payload='{"candidates": [], "reviews": []}'
        page=template.replace('__DATA__',payload)
        updated=m.correct_scope(page,6188)
        self.assertIn('6188',updated);self.assertIn('D085',updated)
        self.assertNotIn('744',updated);self.assertNotIn('<title>OOF ≥0.85',updated)
        self.assertNotIn('GT＝原始标签',updated)
        self.assertIn('const data='+payload+';',updated)
        self.assertIn('原图',updated)

    def test_unknown_legacy_page_fails_closed(self):
        m=self.tool()
        with self.assertRaises(ValueError):m.correct_scope('<html>unknown</html>',6188)


if __name__=='__main__':unittest.main()
