import sys

def search_duplicates(file_dir, verbose=False):
    # type: (str, bool) -> (tuple[dict, dict])
    words = {}
    duplicate_indexes = {}

    with open(file_dir, 'r', encoding='utf-8') as f:
        if verbose:
            i = 0
        for line in f:
            for word in line.split():
                if verbose:
                    if (i % 1000 == 0):
                        print('Processing line: {}'.format(i))
                    i += 1

                found_word = words.get(word, None)
                if found_word is None:
                    words[word] = 1
                else:
                    duplicate_indexes[word] = 1

    return words, duplicate_indexes

if __name__ == '__main__':
    file_dir = sys.argv[1]
    words, duplicate_indexes = search_duplicates(file_dir)
    print('Duplicate indexes: {duplicate_indexes}'.format(
        duplicate_indexes=list(duplicate_indexes.keys())))
    with open(file_dir + '.nodup', 'w', encoding='utf-8') as f:
        for idx, word in enumerate(words.keys()):
            if duplicate_indexes.get(word, None) is None:
                f.write(word + '\n')