"""Receipt collection is serial, provenance-bound, and safe to resume."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ReceiptCollectionTests(unittest.TestCase):
    def tool(self):
        path = ROOT/'tools/cleaning_expansion_collect.py'
        self.assertTrue(path.exists(), 'serial receipt collector not implemented')
        spec = importlib.util.spec_from_file_location('receipt_collect', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_completed_event_is_not_reimported_but_hash_is_still_checked(self):
        m = self.tool()
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            root = Path(temp)
            store = root/'review'
            raw = root/'raw.json'
            raw.write_text('{"opinion":"original"}', encoding='utf-8')
            receipt = dict(batch_id='b', attempt_id='a', reviewer_session_id='ses_test',
                           requested_model=m.review.MODEL, reported_model=None,
                           raw_path='raw.json', raw_sha256=m.review.sha(raw))
            event = dict(feedback=dict(batch_id='b', attempt_id='a'),
                         transport=dict(session_id='ses_test', raw_sha256=m.review.sha(raw)),
                         feedback_sha256=m.review.sha(raw))
            m.review.save(store/'accepted/a.json', event)
            self.assertEqual(m.collect_receipt(root, store, [], receipt), 'already_accepted')
            with self.assertRaises(ValueError):
                m.collect_receipt(root, store, [], dict(receipt, batch_id='wrong-batch'))
            raw.write_text('changed', encoding='utf-8')
            with self.assertRaises(ValueError): m.collect_receipt(root, store, [], receipt)

    def test_receipt_cannot_claim_another_session_or_model(self):
        m = self.tool()
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            root = Path(temp)
            store = root/'review'
            raw = root/'raw.json'
            raw.write_text('{}', encoding='utf-8')
            receipt = dict(batch_id='b', attempt_id='a', reviewer_session_id='ses_wrong',
                           requested_model=m.review.MODEL, reported_model=None,
                           raw_path='raw.json', raw_sha256=m.review.sha(raw))
            m.review.save(store/'accepted/a.json', dict(transport=dict(session_id='ses_original',
                raw_sha256=m.review.sha(raw)), feedback_sha256=m.review.sha(raw)))
            with self.assertRaises(ValueError): m.collect_receipt(root, store, [], receipt)

    def test_receipt_rejects_traversal_and_unrequested_model_before_read(self):
        m = self.tool()
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            root = Path(temp)
            receipt = dict(batch_id='../escape', attempt_id='a', reviewer_session_id='ses_test',
                           requested_model=m.review.MODEL, reported_model=None,
                           raw_path='does-not-exist.json', raw_sha256='0'*64)
            with self.assertRaises(ValueError): m.collect_receipt(root, root/'store', [], receipt)
            receipt.update(batch_id='safe', requested_model='wrong-model')
            with self.assertRaises(ValueError): m.collect_receipt(root, root/'store', [], receipt)


if __name__ == '__main__': unittest.main()
