import logging

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class Embedder:
    """
    Generates embeddings using sentence-transformers.

    The model only reads max_tokens of its own word pieces and silently drops
    the rest, so the chunker measures chunks with count_tokens() against
    max_tokens instead of using a different tokenizer.
    """

    def __init__(self, model_name="all-MiniLM-L6-v2"):

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

        # 256 for all-MiniLM-L6-v2, 512 for BGE small
        self.max_tokens = self.model.max_seq_length

    def count_tokens(self, text):
        """
        Number of word pieces the model sees, excluding [CLS] / [SEP].
        """

        # verbose=False: measuring a too-long candidate is expected, not an error
        return len(
            self.model.tokenizer.encode(text, add_special_tokens=False, verbose=False)
        )

    def embed_texts(self, texts):
        """
        Takes list of strings and returns embeddings.

        Example:
        [
            "termination clause...",
            "liability clause..."
        ]

        Returns:
        [
            [0.12,0.34,...],
            [0.45,0.11,...]
        ]
        """

        # Two special tokens are added on top of the text
        too_long = [t for t in texts if self.count_tokens(t) + 2 > self.max_tokens]

        if too_long:
            logger.warning(
                "%d text(s) exceed %d tokens and will be truncated by %s",
                len(too_long),
                self.max_tokens,
                self.model_name,
            )

        embeddings = self.model.encode(
            texts, show_progress_bar=False, normalize_embeddings=True
        )

        return embeddings.tolist()

    def embed_query(self, query):
        """
        Embeds a single search query.
        """

        embedding = self.model.encode(query, normalize_embeddings=True)

        return embedding.tolist()
