import sys
import argparse


def generate_embeddings(args):
    from src.embeddings.embedder import main as embedder_main

    arguments = [f"--{key}={value}" for key, value in args.items() if key != 'func']
    sys.argv = [sys.argv[0]] + arguments
    embedder_main()


def train_tm(args):
    from src.train.tm_trainer import main as tm_trainer_main

    arguments = [f"--{key}={value}" for key, value in args.items() if key != 'func']
    sys.argv = [sys.argv[0]] + arguments
    tm_trainer_main()


def get_top_docs(args):
    from src.top_docs_selection.doc_selector import main as doc_selector_main

    arguments = [f"--{key}={value}" for key, value in args.items() if key != 'func']
    sys.argv = [sys.argv[0]] + arguments
    doc_selector_main()


def jsonfy(args):
    from src.jsonfy.topic_json_formatter import main as jsonfy_main

    arguments = [f"--{key}={value}" for key, value in args.items() if key != 'func']
    sys.argv = [sys.argv[0]] + arguments
    jsonfy_main()


def main():
    argparser = argparse.ArgumentParser()
    subparsers = argparser.add_subparsers()

    ########################################################################
    #                           EMBEDDINGS                                 #
    ########################################################################
    parser_embeddings = subparsers.add_parser('generate_embeddings')
    parser_embeddings.add_argument("--source_file", type=str, required=True)
    parser_embeddings.add_argument("--output_file", type=str, required=True)
    parser_embeddings.add_argument("--batch_size", type=int, default=128)
    parser_embeddings.add_argument("--sbert_model", type=str, default="all-MiniLM-L6-v2")
    parser_embeddings.add_argument("--aggregate_embeddings", type=bool, default=False)
    parser_embeddings.add_argument("--calculate_on", type=str, default="text")
    parser_embeddings.set_defaults(func=generate_embeddings)

    ########################################################################
    #                          TM   TRAINING                               #
    ########################################################################
    parser_tm_training = subparsers.add_parser('train_tm')
    parser_tm_training.add_argument("--corpus_file", type=str, default="/export/usuarios_ml4ds/lbartolome/Repos/umd/fluffy-train/data/train.metadata.enriched.parquet", required=False)
    parser_tm_training.add_argument("--model_path", type=str, default="/export/usuarios_ml4ds/lbartolome/Repos/umd/theta-evaluation/data/modeltest", required=False)
    parser_tm_training.add_argument("--trainer_type", type=str, default="MalletLda", required=False)
    parser_tm_training.add_argument("--num_topics", type=int, default=50, required=False)
    parser_tm_training.add_argument("--text_col", type=str, default="tokenized_text", required=False)
    parser_tm_training.set_defaults(func=train_tm)

    ########################################################################
    #                              TOP DOCS                                #
    ########################################################################
    parser_top_docs = subparsers.add_parser('get_top_docs')
    parser_top_docs.add_argument('--method', type=str, required=True, help="Method to use for selecting top documents. Available methods: 'thetas', 'thetas_sample', 'thetas_thr', 'sall', 'spart', 's3'")
    parser_top_docs.add_argument('--thetas_path', type=str, required=False, help='Path to the thetas numpy file.')
    parser_top_docs.add_argument('--bow_path', type=str, required=False, help='Path to the bag-of-words numpy file.')
    parser_top_docs.add_argument('--betas_path', type=str, required=False, help='Path to the betas numpy file.')
    parser_top_docs.add_argument('--corpus_path', type=str, required=False, help='Path to the corpus file.')
    parser_top_docs.add_argument('--vocab_path', type=str, required=False, help='Path to the vocabulary file (word to index mapping).')
    parser_top_docs.add_argument('--model_path', type=str, required=False, help='Path to the model directory.')
    parser_top_docs.add_argument('--top_words', type=int, required=False, help='Number of top words to keep in the betas matrix when using S3.')
    parser_top_docs.add_argument('--thr', type=str, required=False, default="0.1,0.8", help='Threshold values for the thetas_thr method.')
    parser_top_docs.add_argument('--ntop', type=int, default=5, help='Number of top documents to select.')
    parser_top_docs.add_argument('--trained_with_thetas_eval', type=bool, default=True, help="Whether the model given by model_path was trained using this code")
    parser_top_docs.add_argument('--text_column', type=str, required=False, default="tokenized_text", help='Column of corpus_path that was used for training the model.')
    parser_top_docs.set_defaults(func=get_top_docs)

    ########################################################################
    #                               JSONFY                                 #
    ########################################################################
    parser_jsonfy = subparsers.add_parser('jsonfy')
    parser_jsonfy.add_argument('--method', type=str, required=True, help="Method to use for selecting top documents. Available methods: 'thetas', 'thetas_sample', 'thetas_thr', 'sall', 'spart', 's3'")
    parser_jsonfy.add_argument('--thetas_path', type=str, required=False, help='Path to the thetas numpy file.')
    parser_jsonfy.add_argument('--bow_path', type=str, required=False, help='Path to the bag-of-words numpy file.')
    parser_jsonfy.add_argument('--betas_path', type=str, required=False, help='Path to the betas numpy file.')
    parser_jsonfy.add_argument('--corpus_path', type=str, required=False, help='Path to the corpus file.')
    parser_jsonfy.add_argument('--vocab_path', type=str, required=False, help='Path to the vocabulary file (word to index mapping).')
    parser_jsonfy.add_argument('--model_path', type=str, required=False, help='Path to the model directory.')
    parser_jsonfy.add_argument('--top_words', type=int, required=False, help='Number of top words to keep in the betas matrix when using S3.')
    parser_jsonfy.add_argument('--thr', type=str, required=False, default="0.1,0.8", help='Threshold values for the thetas_thr method.')
    parser_jsonfy.add_argument('--ntop', type=int, default=5, help='Number of top documents to select.')
    parser_jsonfy.add_argument('--trained_with_thetas_eval', type=bool, default=True, help="Whether the model given by model_path was trained using this code")
    parser_jsonfy.add_argument('--text_column', type=str, required=False, default="tokenized_text", help='Column of corpus_path that was used for training the model.')
    parser_jsonfy.set_defaults(func=jsonfy)

    args = argparser.parse_args()
    if 'func' in args:
        args.func(vars(args))
    else:
        argparser.print_help()


if __name__ == "__main__":
    main()
