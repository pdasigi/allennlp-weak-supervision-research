import re
import csv
from typing import Dict, List, Tuple, Union

from unidecode import unidecode
from allennlp.data.tokenizers import Token
from allennlp.semparse.contexts.knowledge_graph import KnowledgeGraph

# == stop words that will be omitted by ContextGenerator
STOP_WORDS = {"", "", "all", "being", "-", "over", "through", "yourselves", "its", "before",
              "hadn", "with", "had", ",", "should", "to", "only", "under", "ours", "has", "ought", "do",
              "them", "his", "than", "very", "cannot", "they", "not", "during", "yourself", "him",
              "nor", "did", "didn", "'ve", "this", "she", "each", "where", "because", "doing", "some", "we", "are",
              "further", "ourselves", "out", "what", "for", "weren", "does", "above", "between", "mustn", "?",
              "be", "hasn", "who", "were", "here", "shouldn", "let", "hers", "by", "both", "about", "couldn",
              "of", "could", "against", "isn", "or", "own", "into", "while", "whom", "down", "wasn", "your",
              "from", "her", "their", "aren", "there", "been", ".", "few", "too", "wouldn", "themselves",
              ":", "was", "until", "more", "himself", "on", "but", "don", "herself", "haven", "those", "he",
              "me", "myself", "these", "up", ";", "below", "'re", "can", "theirs", "my", "and", "would", "then",
              "is", "am", "it", "doesn", "an", "as", "itself", "at", "have", "in", "any", "if", "!",
              "again", "'ll", "no", "that", "when", "same", "how", "other", "which", "you", "many", "shan",
              "'t", "'s", "our", "after", "most", "'d", "such", "'m", "why", "a", "off", "i", "yours", "so",
              "the", "having", "once"}

NUMBER_CHARACTERS = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.', '-'}
MONTH_NUMBERS = {
        'january': 1,
        'jan': 1,
        'february': 2,
        'feb': 2,
        'march': 3,
        'mar': 3,
        'april': 4,
        'apr': 4,
        'may': 5,
        'june': 6,
        'jun': 6,
        'july': 7,
        'jul': 7,
        'august': 8,
        'aug': 8,
        'september': 9,
        'sep': 9,
        'october': 10,
        'oct': 10,
        'november': 11,
        'nov': 11,
        'december': 12,
        'dec': 12,
        }
ORDER_OF_MAGNITUDE_WORDS = {'hundred': 100, 'thousand': 1000, 'million': 1000000}
NUMBER_WORDS = {
        'zero': 0,
        'one': 1,
        'two': 2,
        'three': 3,
        'four': 4,
        'five': 5,
        'six': 6,
        'seven': 7,
        'eight': 8,
        'nine': 9,
        'ten': 10,
        'first': 1,
        'second': 2,
        'third': 3,
        'fourth': 4,
        'fifth': 5,
        'sixth': 6,
        'seventh': 7,
        'eighth': 8,
        'ninth': 9,
        'tenth': 10,
        **MONTH_NUMBERS,
        }


class Date:
    def __init__(self, year: int, month: int, day: int) -> None:
        self.year = year
        self.month = month
        self.day = day

    def __eq__(self, other) -> bool:
        # Note that the logic below renders equality to be non-transitive. That is,
        # Date(2018, -1, -1) == Date(2018, 2, 3) and Date(2018, -1, -1) == Date(2018, 4, 5)
        # but Date(2018, 2, 3) != Date(2018, 4, 5).
        if not isinstance(other, Date):
            return False
        year_is_same = self.year == -1 or other.year == -1 or self.year == other.year
        month_is_same = self.month == -1 or other.month == -1 or self.month == other.month
        day_is_same = self.day == -1 or other.day == -1 or self.day == other.day
        return year_is_same and month_is_same and day_is_same

    def __gt__(self, other) -> bool:
        # pylint: disable=too-many-return-statements
        # The logic below is tricky, and is based on some assumptions we make about date comparison.
        # Year, month or day being -1 means that we do not know its value. In those cases, the
        # we consider the comparison to be undefined, and return False if all the fields that are
        # more significant than the field being compared are equal. However, when year is -1 for both
        # dates being compared, it is safe to assume that the year is not specified because it is
        # the same. So we make an exception just in that case. That is, we deem the comparison
        # undefined only when one of the year values is -1, but not both.
        if not isinstance(other, Date):
            return False  # comparison undefined
        # We're doing an exclusive or below.
        if (self.year == -1) != (other.year == -1):
            return False  # comparison undefined
        # If both years are -1, we proceed.
        if self.year != other.year:
            return self.year > other.year
        # The years are equal and not -1, or both are -1.
        if self.month == -1 or other.month == -1:
            return False
        if self.month != other.month:
            return self.month > other.month
        # The months and years are equal and not -1
        if self.day == -1 or other.day == -1:
            return False
        return self.day > other.day

    def __ge__(self, other) -> bool:
        if not isinstance(other, Date):
            return False
        return self > other or self == other

    def __str__(self):
        return f"{self.year}-{self.month}-{self.day}"

    @classmethod
    def make_date(cls, string: str) -> 'Date':
        string_parts = string.split("_")
        year = -1
        month = -1
        day = -1
        for part in string_parts:
            if part.isdigit():
                if len(part) == 4:
                    year = int(part)
                else:
                    day = int(part)
            elif part in MONTH_NUMBERS:
                month = MONTH_NUMBERS[part]
        return Date(year, month, day)


class Argument:
    def __init__(self,
                 argument_string: str = None,
                 numbers: List[int] = None,
                 dates: List[Date] = None,
                 entities: List[str] = None) -> None:
        self. argument_string = argument_string
        self.numbers = numbers or []
        self.dates = dates or []
        self.entities = entities or []

    def __eq__(self, other):
        if not isinstance(other, Argument):
            return False
        return self.argument_string == other.argument_string and self.numbers == other.numbers \
                and self.dates == other.dates and self.entities == other.entities


class ParagraphQuestionContext:
    def __init__(self,
                 paragraph_data: List[Dict[str, Argument]],
                 question_tokens: List[Token]) -> None:
        self.paragraph_data = paragraph_data
        self.question_tokens = question_tokens
        self._paragraph_strings = []
        for structure in paragraph_data:
            for argument in structure.values():
                self._paragraph_strings.append(argument.argument_string)
        self._table_knowledge_graph: KnowledgeGraph = None

    def __eq__(self, other):
        if not isinstance(other, ParagraphQuestionContext):
            return False
        return self.paragraph_data == other.paragraph_data

    def get_table_knowledge_graph(self) -> KnowledgeGraph:
        raise NotImplementedError

    @classmethod
    def read_from_lines(cls,
                        lines: List[List[str]],
                        question_tokens: List[Token]) -> 'TableQuestionContext':
        column_index_to_name = {}

        header = lines[0] # the first line is the header
        index = 1
        paragraph_data: List[Dict[str, Argument]] = []
        while lines[index][0] == '-1':
            # column names start with relation:.
            current_line = lines[index]
            column_name = current_line[2]
            column_index = int(current_line[1])
            column_index_to_name[column_index] = column_name
            index += 1
        last_row_index = -1
        for current_line in lines[1:]:
            row_index = int(current_line[0])
            if row_index == -1:
                continue  # header row
            column_index = int(current_line[1])
            if row_index != last_row_index:
                paragraph_data.append({})
            node_info = dict(zip(header, current_line))
            cell_value = cls.normalize_string(node_info['content'])
            column_name = column_index_to_name[column_index]
            number_values = None
            date_values = None
            ner_values = None
            if node_info['date']:
                date_values = [Date.make_date(string) for string in node_info['date'].split('|')]
            if node_info['number']:
                number_values = []
                for string in node_info['number'].split('|'):
                    try:
                        number_values.append(float(string))
                    except ValueError:
                        continue
            if node_info['nerValues']:
                ner_values = node_info['nerValues'].split('|')
            paragraph_data[-1][column_name] = Argument(argument_string=cell_value,
                                                       numbers=number_values,
                                                       dates=date_values,
                                                       entities=ner_values)
            last_row_index = row_index
        return cls(paragraph_data, question_tokens)

    @classmethod
    def read_from_file(cls, filename: str, question_tokens: List[Token]) -> 'ParagraphQuestionContext':
        with open(filename, 'r') as file_pointer:
            reader = csv.reader(file_pointer, delimiter='\t', quoting=csv.QUOTE_NONE)
            lines = [line for line in reader]
            return cls.read_from_lines(lines, question_tokens)

    def get_entities_from_question(self) -> Tuple[List[str], List[Tuple[str, int]]]:
        entity_data = []
        for i, token in enumerate(self.question_tokens):
            token_text = token.text
            if token_text in STOP_WORDS:
                continue
            normalized_token_text = self.normalize_string(token_text)
            if not normalized_token_text:
                continue
            if self._string_in_paragraph(normalized_token_text):
                entity_data.append({'value': normalized_token_text,
                                    'token_start': i,
                                    'token_end': i+1})

        extracted_numbers = self._get_numbers_from_tokens(self.question_tokens)
        # filter out number entities to avoid repetition
        expanded_entities = []
        for entity in self._expand_entities(self.question_tokens, entity_data):
            expanded_entities.append(f"string:{entity['value']}")
        return expanded_entities, extracted_numbers

    @staticmethod
    def _get_numbers_from_tokens(tokens: List[Token]) -> List[Tuple[str, int]]:
        """
        Finds numbers in the input tokens and returns them as strings.  We do some simple heuristic
        number recognition, finding ordinals and cardinals expressed as text ("one", "first",
        etc.), as well as numerals ("7th", "3rd"), months (mapping "july" to 7), and units
        ("1ghz").

        We also handle year ranges expressed as decade or centuries ("1800s" or "1950s"), adding
        the endpoints of the range as possible numbers to generate.

        We return a list of tuples, where each tuple is the (number_string, token_index) for a
        number found in the input tokens.
        """
        numbers = []
        for i, token in enumerate(tokens):
            number: Union[int, float] = None
            token_text = token.text
            text = token.text.replace(',', '').lower()
            if text in NUMBER_WORDS:
                number = NUMBER_WORDS[text]

            magnitude = 1
            if i < len(tokens) - 1:
                next_token = tokens[i + 1].text.lower()
                if next_token in ORDER_OF_MAGNITUDE_WORDS:
                    magnitude = ORDER_OF_MAGNITUDE_WORDS[next_token]
                    token_text += ' ' + tokens[i + 1].text

            is_range = False
            if len(text) > 1 and text[-1] == 's' and text[-2] == '0':
                is_range = True
                text = text[:-1]

            # We strip out any non-digit characters, to capture things like '7th', or '1ghz'.  The
            # way we're doing this could lead to false positives for something like '1e2', but
            # we'll take that risk.  It shouldn't be a big deal.
            text = ''.join(text[i] for i, char in enumerate(text) if char in NUMBER_CHARACTERS)

            try:
                # We'll use a check for float(text) to find numbers, because text.isdigit() doesn't
                # catch things like "-3" or "0.07".
                number = float(text)
            except ValueError:
                pass

            if number is not None:
                number = number * magnitude
                if '.' in text:
                    number_string = '%.3f' % number
                else:
                    number_string = '%d' % number
                numbers.append((number_string, i))
                if is_range:
                    # TODO(mattg): both numbers in the range will have the same text, and so the
                    # linking score won't have any way to differentiate them...  We should figure
                    # out a better way to handle this.
                    num_zeros = 1
                    while text[-(num_zeros + 1)] == '0':
                        num_zeros += 1
                    numbers.append((str(int(number + 10 ** num_zeros)), i))
        return numbers

    def _string_in_paragraph(self, candidate: str) -> bool:
        """
        Checks if the string occurs in the paragraph.
        """
        for string in self._paragraph_strings:
            if candidate in string:
                return True
        return False

    def _expand_entities(self, question, entity_data):
        new_entities = []
        for entity in entity_data:
            # to ensure the same strings are not used over and over
            if new_entities and entity['token_end'] <= new_entities[-1]['token_end']:
                continue
            current_start = entity['token_start']
            current_end = entity['token_end']
            current_token = entity['value']

            while current_end < len(question):
                next_token = question[current_end].text
                next_token_normalized = self.normalize_string(next_token)
                if next_token_normalized == "":
                    current_end += 1
                    continue
                candidate = "%s_%s" %(current_token, next_token_normalized)
                if self._string_in_paragraph(candidate):
                    current_end += 1
                    current_token = candidate
                else:
                    break

            new_entities.append({'token_start' : current_start,
                                 'token_end' : current_end,
                                 'value' : current_token})
        return new_entities

    @staticmethod
    def normalize_string(string: str) -> str:
        """
        These are the transformation rules used to normalize cell in column names in Sempre.  See
        ``edu.stanford.nlp.sempre.tables.StringNormalizationUtils.characterNormalize`` and
        ``edu.stanford.nlp.sempre.tables.TableTypeSystem.canonicalizeName``.  We reproduce those
        rules here to normalize and canonicalize cells and columns in the same way so that we can
        match them against constants in logical forms appropriately.
        """
        # Normalization rules from Sempre
        # \u201A -> ,
        string = re.sub("‚", ",", string)
        string = re.sub("„", ",,", string)
        string = re.sub("[·・]", ".", string)
        string = re.sub("…", "...", string)
        string = re.sub("ˆ", "^", string)
        string = re.sub("˜", "~", string)
        string = re.sub("‹", "<", string)
        string = re.sub("›", ">", string)
        string = re.sub("[‘’´`]", "'", string)
        string = re.sub("[“”«»]", "\"", string)
        string = re.sub("[•†‡²³]", "", string)
        string = re.sub("[‐‑–—−]", "-", string)
        # Oddly, some unicode characters get converted to _ instead of being stripped.  Not really
        # sure how sempre decides what to do with these...  TODO(mattg): can we just get rid of the
        # need for this function somehow?  It's causing a whole lot of headaches.
        string = re.sub("[ðø′″€⁄ªΣ]", "_", string)
        # This is such a mess.  There isn't just a block of unicode that we can strip out, because
        # sometimes sempre just strips diacritics...  We'll try stripping out a few separate
        # blocks, skipping the ones that sempre skips...
        string = re.sub("[\\u0180-\\u0210]", "", string).strip()
        string = re.sub("[\\u0220-\\uFFFF]", "", string).strip()
        string = string.replace("\\n", "_")
        string = re.sub("\\s+", " ", string)
        # Canonicalization rules from Sempre.
        string = re.sub("[^\\w]", "_", string)
        string = re.sub("_+", "_", string)
        string = re.sub("_$", "", string)
        return unidecode(string.lower())
