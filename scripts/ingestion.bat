@echo off
set INGEST_AUGMENTED_DATA=%1
set VOCABS=artifacts\\data\\vocabulary\\gn_unique_tokens.txt artifacts\\data\\vocabulary\\es_unique_tokens.txt
set TRAIN_SETS=artifacts\\data\\train\\train_gn.txt artifacts\\data\\train\\train_es.txt
set VALID_SETS=artifacts/data/validation/valid_gn.txt artifacts/data/validation/valid_es.txt

if NOT "%INGEST_AUGMENTED_DATA%" == "" (
    python main.py --command-path marian --ingest --ingest-augmented-data --flags "--vocabs %VOCABS% --train-sets %TRAIN_SETS% --valid-sets %VALID_SETS%"
) else (
    python main.py --command-path marian --ingest --flags "--vocabs %VOCABS% --train-sets %TRAIN_SETS%--valid-sets %VALID_SETS%"
)