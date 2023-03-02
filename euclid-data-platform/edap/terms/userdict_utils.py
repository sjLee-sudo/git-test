import re

one_word_regex_word = re.compile(r"^[가-힣a-zA-Z]$", re.UNICODE)
only_number_word = re.compile(r"^[0-9]+$", re.UNICODE)
only_consonant_vowel_word = re.compile(r"(^[ㄱ-ㅎㅏ-ㅣ]+$)", re.UNICODE)
back_number_word = re.compile(r"([0-9]+$)", re.UNICODE)
front_or_back_consonant_vowel_word = re.compile(r"(^[ㄱ-ㅎㅏ-ㅣ]+[^자형][\w]+)|([\w]+[ㄱ-ㅎㅏ-ㅣ]+$)", re.UNICODE)
ending_word = re.compile(r"[를|는]$", re.UNICODE)
english_only_pattern = re.compile(r'^([(a-zA-Z)+\-*(a-zA-Z)*]*)$')

word_only_pattern = re.compile("[^a-zA-Z0-9ㄱ-ㅎ가-힣\s]")

delete_word_list = [one_word_regex_word, only_number_word, only_consonant_vowel_word, back_number_word,
                    front_or_back_consonant_vowel_word, ending_word]


def delete_word_after_making_userdict(word):
    for delete_word in delete_word_list:
        if delete_word.search(word):
            return True
    return False