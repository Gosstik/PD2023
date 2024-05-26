#!/usr/bin/env python3

import sys

cur_perm = ''
cur_word = ''
perm_cnt = 0
word_cnt = 0
word_dict = {}


def get_5_most_frequent():
    res = sorted(word_dict.items(), key=lambda item: item[1], reverse=True)
    if len(res) > 5:
        res = res[:5]

    word_info = ["%s:%d" % (key_word, val) for key_word, val in res]

    return ';'.join(word_info)


def handle_last_word():
    global perm_cnt, word_cnt, cur_word

    perm_cnt += word_cnt

    if cur_word in word_dict:
        word_dict[cur_word] += word_cnt
    else:
        word_dict[cur_word] = word_cnt

    cur_word = word
    word_cnt = 0


def handle_perm_res():
    global cur_perm, perm_cnt, word_dict

    word_info = get_5_most_frequent()

    print("{}\t{}\t{};".format(cur_perm, perm_cnt, word_info))

    cur_perm = perm
    perm_cnt = 0
    word_dict = {}


for line in sys.stdin:
    try:
        perm, word, cnt = line.strip().split('\t', 2)
        cnt = int(cnt)
    except ValueError as e:
        continue

    if not perm:
        cur_perm = perm
        cur_word = word

    if word != cur_word:
        handle_last_word()

    if perm != cur_perm:
        handle_perm_res()

    word_cnt += cnt

if cur_perm:
    handle_perm_res()
