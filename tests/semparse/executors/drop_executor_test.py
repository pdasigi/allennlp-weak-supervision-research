# pylint: disable=no-self-use,invalid-name,too-many-public-methods
from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.tokenizers import WordTokenizer
from allennlp.semparse.worlds.world import ExecutionError

from weak_supervision.semparse.contexts import ParagraphQuestionContext
from weak_supervision.semparse.executors import DropExecutor


class TestDropExecutor(AllenNlpTestCase):
    def setUp(self):
        super().setUp()
        question_tokens = WordTokenizer().tokenize("""How many yards was the field goal made by David
                                                   Akers?""")
        self.context_file = "fixtures/data/drop/sample_table.tagged"
        context = ParagraphQuestionContext.read_from_file(self.context_file,
                                                          question_tokens)
        self.executor = DropExecutor(context.paragraph_data)

    def _get_executor_with_question(self, question: str) -> DropExecutor:
        question_tokens = WordTokenizer().tokenize(question)
        context = ParagraphQuestionContext.read_from_file(self.context_file,
                                                          question_tokens)
        return DropExecutor(context.paragraph_data)

    def test_execute_fails_with_unknown_function(self):
        logical_form = "(unknown_function all_structures relation:verb)"
        with self.assertRaises(ExecutionError):
            self.executor.execute(logical_form)

    def test_execute_works_with_select(self):
        executor = self._get_executor_with_question("Who made touchdown points?")
        logical_form = """(select (filter_in all_structures relation:arg1 string:touchdown)
                           relation:arg0)"""
        cell_list = executor.execute(logical_form)
        assert set(cell_list) == {'he', 'kaepernick', 'campbell'}

    def test_execute_works_with_extract_entity(self):
        executor = self._get_executor_with_question("Who made the longest field goal?")
        logical_form = """(extract_entity (argmax (filter_in all_structures relation:arg1 string:field_goal)
                                   relation:arg1) relation:arg0)"""
        cell_list = executor.execute(logical_form)
        assert cell_list == ['Kendall Hunter']

    def test_execute_works_with_argmax(self):
        executor = self._get_executor_with_question("Who made the longest field goal?")
        logical_form = """(select (argmax (filter_in all_structures relation:arg1 string:field_goal)
                                   relation:arg1) relation:arg0)"""
        cell_list = executor.execute(logical_form)
        assert cell_list == ['49ers_running_back_kendall_hunter']

    def test_execute_works_with_argmin(self):
        executor = self._get_executor_with_question("Who made the shortest field goal?")
        logical_form = """(select (argmin (filter_in all_structures relation:arg1 string:field_goal)
                                   relation:arg1) relation:arg0)"""
        cell_list = executor.execute(logical_form)
        assert cell_list == ['akers']

    def test_execute_works_with_filter_number_greater(self):
        executor = self._get_executor_with_question("""Whose field goal was longer than 33 yards?""")
        logical_form = """(extract_entity (filter_number_greater all_structures relation:arg1 33)
                           relation:arg0)"""
        cell_value_list = executor.execute(logical_form)
        assert cell_value_list == ['Kaepernick', 'Kyle Williams', 'Kendall Hunter']

    def test_execute_works_with_filter_date_greater(self):
        executor = self._get_executor_with_question("""Who scored after the first quarter?""")
        logical_form = """(extract_entity (filter_date_greater (filter_in all_structures
                                           relation:verb string:scored) relation:arg2 (date -1 -1 -1 1))
                           relation:arg0)"""
        cell_value_list = executor.execute(logical_form)
        assert cell_value_list == ['Kendall Hunter']

    def test_execute_works_with_filter_number_greater_equals(self):
        executor = self._get_executor_with_question("""Who scored during or after the first quarter?""")
        logical_form = """(extract_entity (filter_date_greater_equals (filter_in all_structures
                                           relation:verb string:scored) relation:arg2 (date -1 -1 -1 1))
                           relation:arg0)"""
        cell_value_list = executor.execute(logical_form)
        assert cell_value_list == ['Kendall Hunter']

    def test_execute_works_with_filter_number_equals(self):
        executor = self._get_executor_with_question("""Whose field goal was 33 yards long?""")
        logical_form = """(extract_entity (filter_number_equals all_structures relation:arg1 33)
                           relation:arg0)"""
        cell_value_list = executor.execute(logical_form)
        assert cell_value_list == []

    def test_execute_works_with_first(self):
        executor = self._get_executor_with_question("""Who scored the first field goal?""")
        logical_form = """(extract_entity (first (filter_in all_structures relation:arg1
                                                  string:field_goal))
                           relation:arg0)"""
        cell_list = executor.execute(logical_form)
        assert cell_list == ["David Akers"]

    def test_execute_works_with_last(self):
        executor = self._get_executor_with_question("""Who scored the last touchdown?""")
        logical_form = """(extract_entity (last (filter_in all_structures relation:arg1
                                                  string:touchdown))
                           relation:arg0)"""
        cell_list = executor.execute(logical_form)
        assert cell_list == ["Kaepernick"]

    def test_execute_works_with_previous(self):
        executor = self._get_executor_with_question("""Who scored first touchdown?""")
        # First row has "He" as the ARG0. The previous one has the person's name.
        logical_form = """(extract_entity (previous (first (filter_in all_structures relation:arg1
                                                           string:touchdown)))
                           relation:arg0)"""
        cell_list = executor.execute(logical_form)
        assert cell_list == ["Vince Ferragamo"]

    def test_execute_works_with_average(self):
        executor = self._get_executor_with_question("""What as the average field goal distance?""")
        logical_form = """(average (filter_in all_structures relation:arg1 string:field_goal) relation:arg1)"""
        avg_value = executor.execute(logical_form)
        assert avg_value == 34.5  # average of 32, 32 and 37

    def test_execute_works_with_diff(self):
        executor = self._get_executor_with_question("""How many points did the Bears score after the
                                                    first quarter7""")
        logical_form = """(diff (extract_number (filter_in all_structures relation:arg1
                                                 string:final_score) relation:arg1)
                                (extract_number (filter_in all_structures relation:arg2
                                                 string:first_quarter) relation:arg2))"""
        avg_value = executor.execute(logical_form)
        assert avg_value == -17.0
