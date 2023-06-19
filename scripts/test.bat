set TEST_ONLY=%1

if "%TEST_ONLY%"=="" (
    python -m unittest tests.regression.ingestion_test
    python -m unittest tests.regression.valid_script_test
    python -m unittest tests.regression.train_test
) else if "%TEST_ONLY%"=="test1" (
    python -m unittest tests.regression.ingestion_test
) else if "%TEST_ONLY%"=="test2" (
    python -m unittest tests.regression.valid_script_test
) else if "%TEST_ONLY%"=="test3" (
    python -m unittest tests.regression.train_test
)