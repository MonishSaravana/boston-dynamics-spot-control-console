import unittest

import numpy as np

from scope.interaction import TextCommand
from scope.semantic_query import SemanticQuery, ObjectCandidate, QueryState, QueryResult, decide


class SemanticQueryTests(unittest.TestCase):
    def test_phrase_trace_and_arbitrary_vocabulary(self):
        q = SemanticQuery.from_command(TextCommand("  Gray COUCH!  ", request_id="q1"))
        self.assertEqual(q.raw_phrase, "  Gray COUCH!  ")
        self.assertEqual(q.alternatives, ("gray couch", "gray sofa"))
        self.assertEqual(q.request_id, "q1")
        self.assertEqual(q.entity_label,"sofa")
        self.assertEqual(SemanticQuery.from_command(TextCommand("robot charger")).alternatives, ("robot charger",))
        self.assertEqual(SemanticQuery.from_command(TextCommand("bagpipe")).alternatives, ("bagpipe",))

    def test_absent_low_and_ambiguous_are_distinct(self):
        def c(box, score): return ObjectCandidate(box, score, "object", "model")
        self.assertEqual(decide([])[0], QueryState.NOT_FOUND)
        self.assertEqual(decide([c((0,0,10,10),.25)])[0], QueryState.LOW_CONFIDENCE)
        self.assertEqual(decide([c((0,0,10,10),.8),c((20,0,30,10),.7)])[0], QueryState.AMBIGUOUS)
        state,_,candidates = decide([c((0,0,10,10),.8),c((0,0,10,10),.7)])
        self.assertEqual(state, QueryState.FOUND)
        self.assertEqual(len(candidates),1)

    def test_empty_phrases_rejected(self):
        for phrase in ("", "  ", ".", "x"*181):
            with self.assertRaises(ValueError):SemanticQuery.from_command(TextCommand(phrase))

    def test_disagreement_abstains_and_keeps_raw_proposal(self):
        from scope.open_vocab import corroborate
        q=SemanticQuery.from_command(TextCommand("toolbox"))
        raw=QueryResult(q,QueryState.FOUND,"proposal",[ObjectCandidate((0,0,10,10),.8,"toolbox","dino")],0.,"f")
        other=QueryResult(q,QueryState.NOT_FOUND,"none",[],0.,"f",{"model":"owl"})
        result=corroborate(raw,other)
        self.assertEqual(result.state,QueryState.LOW_CONFIDENCE)
        self.assertFalse(result.candidates[0].evidence["verification"]["agreed"])
        self.assertEqual(len(result.metadata["rejected_proposals"]),1)


if __name__ == "__main__": unittest.main()
