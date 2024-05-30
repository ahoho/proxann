# THETA-EVALUATION

## Modules

### Embedder

Calculates embeddings using `SentenceTransformers`, with or without aggregation, and saves them as strings in a new dataframe column.

### TopicJsonFormatter

Generates JSON output for topic models, including:

- **Representative Documents** (methods: `'thetas'`, `'thetas_sample'`, `'thetas_thr'`, `'sall'`, `'spart'`, `'s3'`)
- **Top Words for Each Topic**
- **Evaluation Documents and Their Probabilities**

Example structure:

```json
{
    "<topic_id>": {
        "exemplar_docs": ["<top doc>", "<2nd top doc>", "<nth top doc>"],
        "topic_words": ["<top word>", "<2nd top word>", "<nth top word>"],
        "eval_docs": ["<high prob doc>", "<lower prob doc>", "<nth lower prob doc>"],
        "eval_docs_probs": [0.99, 0.87, 0.73, 0.61, 0.22, 0.0]
    }
    ...
}
```

### DocSelector

Implements methods to select representative documents for each topic.

### TMTrainer

Trains topic models with wrappers for LDA-Mallet, LDA-Tomotopy, and BERTopic.

## Notebooks

## TODO

- [ ] Revise S3.
- [ ] Add saving of JSON to file.
- [ ] Check ids of eval_probs are calculated in the proper order
