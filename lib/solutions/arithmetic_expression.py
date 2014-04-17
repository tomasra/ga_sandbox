from lib.chromosomes.binary import BinaryChromosome
from lib.solution import Solution, SolutionFactory
from lib.helpers import Helpers


class ArithExpSolution(Solution):
    _na_char = '?'
    _digits = "0123456789"
    _operators = "+-*/"
    _coding = _digits + _operators + "??"
    _char_length = 4

    def __init__(self, target, length):
        self._target = target
        self._length = length

    @property
    def target(self):
        return self._target

    @property
    def length(self):
        return self._length

    @property
    def expression(self):
        return self._expression

    @expression.setter
    def expression(self, value):
        self._expression = value

    def __repr__(self):
        return "Expression: {0}, value: {1}".format(
            self.expression, self._evaluate())

    def encode(self):
        """
        Encodes expression as binary chromosome
        """
        binary_content = "".join([
            Helpers.char_index_to_bin(self._coding, [char, self._na_char], 4)
            for char
            in self.expression
        ])
        return BinaryChromosome(content=binary_content)

    def decode(self, chromosome):
        """
        Decodes expression from binary chromosome
        """
        self.expression = "".join([
            self._coding[Helpers.bin_to_int(chunk)]
            for chunk
            in Helpers.enumerate_chunks(chromosome.content, 4)
        ])
        return self

    @property
    def fitness(self):
        diff = abs(self._target - self._evaluate())
        return 1.0 / (1.0 + diff)

    def initialize_chromosome(self):
        char_length = ArithExpSolution._char_length
        return BinaryChromosome(length=self._length * char_length)

    def _evaluate(self):
        """
        Evaluates current expression string.
        Operators have no precedence and are applied from left to right
        If two or more characters of same type go in sequence,
        only the first one is evaluated while the rest are skipped
        I.e: "22+?-72" is evaluated as "2+7"
        """
        digits = "0123456789"
        operators = "+-*/"
        result = 0

        # it's ok to just add the first digit to the result (currently zero)
        current_operator = '+'

        # assuming the expression starts with a digit
        digit_next, operator_next = True, False

        for char in self.expression:
            if digit_next and char in digits:
                # digit needed and found
                # the current operator can now be applied
                digit = int(char)
                if current_operator == '+':
                    result += digit
                elif current_operator == '-':
                    result -= digit
                elif current_operator == '*':
                    result *= digit
                elif current_operator == '/' and digit != 0:
                    result /= digit
                digit_next, operator_next = False, True

            elif operator_next and char in operators:
                # operator needed and found
                # can be saved to be applied for the next digit
                current_operator = char
                digit_next, operator_next = True, False

        return result


class ArithExpSolutionFactory(SolutionFactory):
    def __init__(self, target, length):
        self._target = target
        self._length = length

    def create(self):
        return ArithExpSolution(
            target=self._target,
            length=self._length
        )
