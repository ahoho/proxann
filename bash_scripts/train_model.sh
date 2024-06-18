#!/bin/bash

cd "$(dirname "$0")/.."

pwd

# Train Mallet model
: <<'END_COMMENT'
python3 main.py train_tm \
    --corpus_file "/export/usuarios_ml4ds/lbartolome/Repos/umd/theta-evaluation/data/train.metadata.jsonl" \
    --model_path "/export/usuarios_ml4ds/lbartolome/Repos/umd/theta-evaluation/data/models/trained/mallet_wiki_50_v2" \
    --trainer_type "MalletLda" \
    --num_topics 50 \
    --text_col "tokenized_text"


# Train TomotopyLda model
python3 main.py train_tm \
    --corpus_file "/export/usuarios_ml4ds/lbartolome/Repos/umd/theta-evaluation/data/train.metadata.jsonl" \
    --model_path "/export/usuarios_ml4ds/lbartolome/Repos/umd/theta-evaluation/data/models/trained/tomotopy_wiki_50_v2" \
    --trainer_type "TomotopyLda" \
    --num_topics 50 \
    --text_col "tokenized_text"
END_COMMENT

# Train Bertopic model (CTM embeddings and preprocessed text)
python3 main.py train_tm \
    --corpus_file "/export/usuarios_ml4ds/lbartolome/Repos/umd/theta-evaluation/data/train.metadata.enriched.parquet" \
    --model_path "/export/usuarios_ml4ds/lbartolome/Repos/umd/theta-evaluation/data/models/trained/bertopic_wiki_50_ctm_emb_pre2" \
    --trainer_type "BERTopic" \
    --num_topics 50 \
    --text_col "tokenized_text" \
    --vocab_path "data/models/mallet/vocab.json"


: <<'END_COMMENT'
# Train Bertopic model (BERTopic default embeddings and preprocessed text)
python3 main.py train_tm \
    --corpus_file "/export/usuarios_ml4ds/lbartolome/Repos/umd/theta-evaluation/data/train.metadata.enriched.parquet.all-MiniLM-L6-v2.parquet" \
    --model_path "/export/usuarios_ml4ds/lbartolome/Repos/umd/theta-evaluation/data/models/trained/bertopic_wiki_50_bert_dft_emb_pre" \
    --trainer_type "BERTopic" \
    --num_topics 50 \
    --text_col "tokenized_text"

# Train Bertopic model (CTM embeddings and raw text)
python3 main.py train_tm \
    --corpus_file "/export/usuarios_ml4ds/lbartolome/Repos/umd/theta-evaluation/data/train.metadata.enriched.parquet" \
    --model_path "/export/usuarios_ml4ds/lbartolome/Repos/umd/theta-evaluation/data/models/trained/bertopic_wiki_50_ctm_emb" \
    --trainer_type "BERTopic" \
    --num_topics 50 \
    --text_col "text"

# Train Bertopic model (BERTopic default embeddings and raw text)
python3 main.py train_tm \
    --corpus_file "/export/usuarios_ml4ds/lbartolome/Repos/umd/theta-evaluation/data/train.metadata.enriched.parquet.all-MiniLM-L6-v2.parquet" \
    --model_path "/export/usuarios_ml4ds/lbartolome/Repos/umd/theta-evaluation/data/models/trained/bertopic_wiki_50_bert_dft_emb" \
    --trainer_type "BERTopic" \
    --num_topics 50 \
    --text_col "text"
    END_COMMENT
