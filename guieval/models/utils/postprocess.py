import re
import inspect
import json
from pydantic import BaseModel, AfterValidator, model_validator
from typing import (Annotated, Sequence, Callable, TypedDict, Any)
from typing_extensions import Self
from string import Template


class ParserTools:
    @staticmethod
    def fetch_matched_exterior_braces(dstr: str) -> list[tuple[int, int]]:
        '''
        Fetch the matched exterior braces in the string.
        The returned list contains the start and end indices of the braces.

        Args:
            dstr (str): The string to fetch the matched exterior braces from.

        Returns:
            list: The list of matched exterior braces.

        Example:
            >>> ParserTools.fetch_matched_exterior_braces('{1,2,3}')
            [(0, 6)]
            >>> ParserTools.fetch_matched_exterior_braces('{1,2,3}, {4, 5}')
            [(0, 6), (8, 14)]
        '''
        pairs = list()
        box = list()
        left_braces_count = dstr.count(r'{')
        right_braces_count = dstr.count(r'}')
        if left_braces_count * right_braces_count == 0:
            return list()
        else:
            _start = 0
            for i, element in enumerate(dstr):
                if element == '{':
                    if not box:
                        _start = i
                    box.append(element)
                elif element == '}':
                    if box:
                        box.pop()
                        if not box and i > _start:
                            pairs.append((_start, i))
        return pairs

    @staticmethod
    def parse_json_dict_block(dstr: str) -> dict:
        '''
        Parse the JSON dictionary block in the string.
        The returned dictionary is the parsed JSON dictionary.

        Raise AttributeError if no parseable braced json block is found.

        Args:
            dstr (str): The string to parse the JSON dictionary block from.

        Returns:
            dict: The parsed JSON dictionary.

        Example:
            >>> ParserTools.parse_json_dict_block('foobar{"a": 1, "b": 2} 42 shall be the answer.')
            {"a": 1, "b": 2}
        '''
        for pair in ParserTools.fetch_matched_exterior_braces(dstr):
            try:
                return json.loads(dstr[pair[0]:(pair[1] + 1)])
            except json.JSONDecodeError:
                pass
        else:
            raise AttributeError('No parseable braced json block')

    @staticmethod
    def enhanced_strip(dstr: str, *, extra: list[str] = list()) -> str:
        '''
        Enhanced strip the string.
        The returned string is the stripped string.
        If the string does not match the pattern, the returned string is the original string.

        Args:
            dstr (str): The string to strip.

        Returns:
            str: The stripped string.

        Example:
            >>> ParserTools.enhanced_strip('\\t\\n\\s\\t 42 shall be the answer.  ')
            "42 shall be the answer."
        '''
        try:
            pattern = r"'\"\n\s\t" + ''.join(extra)
            stripped = re.search(f"^[{pattern}]*(.*?)[{pattern}]*$", dstr, re.DOTALL).group(1)
            return stripped if stripped else dstr
        except Exception:
            return dstr


def _unite_flags(flags: list[re.RegexFlag] | None) -> re.RegexFlag | None:
    if flags is None:
        return None
    elif isinstance(flags, Sequence):
        union = 0
        for _flag in flags:
            union |= _flag
        return union
    else:
        return flags


def _tolerantly_compile(pattern: re.Pattern | str | None) -> re.Pattern | None:
    if pattern is None:
        return None
    else:
        return re.compile(pattern)


class ModelPatterns(BaseModel):
    answer_pattern: Annotated[str | re.Pattern | None,
                              AfterValidator(_tolerantly_compile)] = re.compile(r'<tool_call>.*({.*}).*</tool_call>',
                                                                                re.DOTALL)
    answer_flags: Annotated[re.RegexFlag | list[re.RegexFlag] | None,
                            AfterValidator(_unite_flags)] = None

    thinking_pattern: Annotated[str | re.Pattern | None,
                                AfterValidator(_tolerantly_compile)] = re.compile(r'<thinking>(.*)</thinking>',
                                                                                  re.DOTALL)
    thinking_flags: Annotated[re.RegexFlag | list[re.RegexFlag] | None, AfterValidator(_unite_flags)] = None

    conclusion_pattern: Annotated[str | re.Pattern | None,
                                  AfterValidator(_tolerantly_compile)] = re.compile(r'<conclusion>(.*)</conclusion>',
                                                                                    re.DOTALL)
    conclusion_flags: Annotated[re.RegexFlag | list[re.RegexFlag] | None,
                                AfterValidator(_unite_flags)] = None

    @model_validator(mode='after')
    def _compile_with_flags(self) -> Self:
        if self.answer_pattern is not None and self.answer_flags is not None:
            raw_pattern = self.answer_pattern.pattern
            self.answer_pattern = re.compile(raw_pattern, self.answer_flags)
        if self.thinking_pattern is not None and self.thinking_flags is not None:
            raw_pattern = self.thinking_pattern.pattern
            self.thinking_pattern = re.compile(raw_pattern, self.thinking_flags)
        if self.conclusion_pattern is not None and self.conclusion_flags is not None:
            raw_pattern = self.conclusion_pattern.pattern
            self.conclusion_pattern = re.compile(raw_pattern, self.conclusion_flags)
        return self

    def search_answer(self, resp: str) -> re.Match | None:
        if self.answer_pattern is None:
            return None
        else:
            return self.answer_pattern.search(resp)

    def search_thinking(self, resp: str) -> re.Match | None:
        if self.thinking_pattern is None:
            return None
        else:
            return self.thinking_pattern.search(resp)

    def search_conclusion(self, resp: str) -> re.Match | None:
        if self.conclusion_pattern is None:
            return None
        else:
            return self.conclusion_pattern.search(resp)


class ModelPatternExtractionError(Exception):
    '''
    raises when no proper match found for corresponding `ModelPatterns` instance.
    '''
    DEFAULT_MESSAGE_TEMPLATE = Template('No proper `ModelPatterns` match found for text:\n\t${text}')

    def __init__(self, text: str | None = None, *,
                 _message_template: Template = DEFAULT_MESSAGE_TEMPLATE,
                 _template_key: str = 'text'):
        self._text = text
        if f'${{{_template_key}}}' not in _message_template.template:
            raise ValueError(f'No matched template key `{_template_key}` in provided template')
        self._template = _message_template
        self._template_key = _template_key

    def __str__(self):
        return self._template.safe_substitute(**{self._template_key: self._text})

    def __repr__(self):
        return f'{self.__class__.__name__}("No proper `ModelPatterns` match found")'


class PARSED_MATCHES(TypedDict):
    answer: re.Match | None
    thinking: re.Match | None
    conclusion: re.Match | None

    error: ModelPatternExtractionError


class first_level_parser(ModelPatterns):
    '''
    A instantiatable decorator for method or function to parse the response of the model.
    The returned dictionary is the parsed `re.Match` objects for `answer`, `thinking`, and `conclusion`.

    Args:
        patterns (ModelPatterns): The patterns to parse the response.

    Returns:
        Callable: The decorated function or method.

    Example:
        >>> @first_level_parser(patterns=ModelPatterns())
        >>> def parse_response(parsed_matches: PARSED_MATCHES):
        >>>     ...  # continue to process the parsed matches
        >>>     return processed_parsed_matches
    '''

    def __call__(self, func: Callable[[PARSED_MATCHES], Any]):
        sig = inspect.signature(func)
        params = list(sig.parameters.keys())  # check the function signature to determine the parameter structure <save>

        def method_wrapper(instance, resp: str):
            parsed_matches = dict(
                answer=self.search_answer(resp=resp),
                thinking=self.search_thinking(resp=resp),
                conclusion=self.search_conclusion(resp=resp),
                error=ModelPatternExtractionError(text=resp)
            )
            return func(instance, parsed_matches)

        def class_method_wrapper(cls, resp: str):
            parsed_matches = dict(
                answer=self.search_answer(resp=resp),
                thinking=self.search_thinking(resp=resp),
                conclusion=self.search_conclusion(resp=resp),
                error=ModelPatternExtractionError(text=resp)
            )
            return func(cls, parsed_matches)

        def function_wrapper(resp: str):
            parsed_matches = dict(
                answer=self.search_answer(resp=resp),
                thinking=self.search_thinking(resp=resp),
                conclusion=self.search_conclusion(resp=resp),
                error=ModelPatternExtractionError(text=resp)
            )
            return func(parsed_matches)

        if not callable(func):
            raise ValueError("Decorated object must be callable")

        if params and params[0] == 'self':
            return method_wrapper
        elif params and params[0] == 'cls':
            return class_method_wrapper
        else:
            return function_wrapper

    @classmethod
    def validate_patterns(cls, patterns: ModelPatterns | None):
        if isinstance(patterns, ModelPatterns):
            return cls.model_validate(patterns.model_dump())
        else:
            return lambda func: func
