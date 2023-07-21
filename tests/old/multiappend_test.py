import multiprocessing as mp

def append_to_file(file: str, text: str):
    with open(file, 'a', encoding='utf-8') as f:
        f.write(text)

PROCESS_NUM = 500

def multiappend_to_file(file: str, text: str):
    with mp.Pool(PROCESS_NUM) as p:
        p.starmap(append_to_file, [(file, text)] * PROCESS_NUM)

if __name__ == '__main__':
    multiappend_to_file('test.txt', 'This is a test message :). Feel free to ruin this file\n')
