import pytest
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.preprocessing import clean_text, is_english, limit_review_length


class TestCleanText:

    def test_lowercase_text(self):
        assert clean_text('GREAT FooD AMAziNG SERvice') == 'great food amazing service'

    def test_removes_url(self):
        result = clean_text("Check out this URL https://www.google.com for more information")
        assert "https" not in result
        assert "google.com" not in result


    def test_removes_html_tags(self):
        result = clean_text("<p> Check out <i> this URL for more </i> information </p>")
        assert '<p>' and '</p>' not in result
        assert '<i>' and '</i>' not in result
        assert 'check out' in result
        assert 'check out this url for more information'

    def test_removes_punctuation(self):
        result = clean_text("Check!!! out thi?s?? URL: for more information!?")
        assert '?' not in result
        assert "!" not in result
        assert ':' not in result
        assert 'this' in result


    def test_preserves_numbers(self):
        result = clean_text('We waited 45 minutes for our table of 9')
        assert '45' in result
        assert '9' in result

    def test_collapses_extra_whitespace(self):
        result = clean_text('    We waited       45 minutes for     our table of 9        ')
        assert '    ' not in result
        assert 'we waited 45 minutes for our table of 9' in result


    def test_strips_leading_trailing_whitespace(self):
        result = clean_text('    great food        ')
        assert result == 'great food'

    def test_handles_empty_string(self):
        assert clean_text("") == ''

    def test_handles_only_punctuation(self):
        assert clean_text('!.?!!!..... ') == ''

    def test_returns_string(self):
        assert isinstance(clean_text('Great food here'), str)


class TestIsEnglish:

    def test_english_review_returns_true(self):
        assert is_english('the restaurant we went to last night was great') == True

    def test_clearly_non_english_returns_false(self):
        assert is_english('el restaurante es mala') == False
        
    def test_returns_bool(self):
        assert isinstance(is_english('the restaurant we went to last night was fabulous'), bool)


class TestLimitReviewLength:

    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            'review': ['a', 'b', 'c', 'd', 'e'],
            'review_length': [5, 10, 50, 200, 600],
            'review_rating': [1, 2, 3, 4, 5]
        })

    def test_filters_below_min(self, sample_df):
        result = limit_review_length(sample_df, 10, 500)
        assert 5 not in result['review_length'].values

    def test_filters_above_max(self, sample_df):
        result = limit_review_length(sample_df, 10, 500)
        assert 600 not in result['review_length'].values

    def test_keeps_values_within_range(self, sample_df):
        result = limit_review_length(sample_df, 10, 500)
        assert set(result['review_length'])== {10,50,200}

    def test_includes_boundary_min(self, sample_df):
        result = limit_review_length(sample_df, 10, 500)
        assert 10 in result['review_length'].values

    def test_returns_dataframe(self, sample_df):
        assert isinstance(limit_review_length(sample_df, 10, 500), pd.DataFrame)

    def test_preserves_all_columns(self, sample_df):
        result = limit_review_length(sample_df, 10, 500)
        assert list(result.columns)== list(sample_df.columns)

    def test_empty_result_when_no_match(self, sample_df):
        result = limit_review_length(sample_df, 201, 500)
        assert len(result)==0

    def test_all_pass_when_wide_range(self, sample_df):
        result = limit_review_length(sample_df, 1, 900)
        assert len(result)==len(sample_df)

    def test_does_not_modify_original(self, sample_df):
        original_len = len(sample_df)
        limit_review_length(sample_df, 11, 300)
        assert original_len == len(sample_df)
