
## Get embeddings
```
python3 main.py generate_embeddings --source_file /export/usuarios_ml4ds/lbartolome/Repos/umd/theta-evaluation/data/training_data/bills/train.metadata.jsonl --output_file /export/usuarios_ml4ds/lbartolome/Repos/umd/theta-evaluation/data/training_data/bills/train.metadata.embeddings.jsonl --batch_size 512 --sbert_model all-MiniLM-L6-v2 --calculate_on summary
```

```
python3 main.py generate_embeddings --source_file /export/usuarios_ml4ds/lbartolome/Repos/umd/theta-evaluation/data/training_data/wiki/train.metadata.jsonl --output_file /export/usuarios_ml4ds/lbartolome/Repos/umd/theta-evaluation/data/training_data/wiki/train.metadata.embeddings.jsonl --batch_size 512 --sbert_model all-MiniLM-L6-v2 --calculate_on text
```

