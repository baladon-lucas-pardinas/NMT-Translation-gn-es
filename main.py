from tests import minimal as tests

if __name__ == '__main__':
    import sys
    test_method = sys.argv[1]
    test_methods = {
        'minimal_train_cpu': tests.minimal_train_cpu,
        'minimal_evaluation_cpu': tests.minimal_evaluation_cpu,
        'minimal_train_gpu': tests.minimal_train_gpu,
    }

    marian_dir = '../marianmt/build'
    corpus_dir = 'artifacts/data'
    try:
        test_methods[test_method](marian_dir, corpus_dir)
    except KeyError:
        print('Unknown test method: {test_method}')
        print('Available test methods: {test_method_keys}'.format(test_method_keys=test_methods.keys()))