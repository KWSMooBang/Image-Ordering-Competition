"""Compatibility wrapper for src.pairwise_compare.reconstruct."""

from src.pairwise_compare.reconstruct import *  # noqa: F401,F403
from src.pairwise_compare.reconstruct import main, order_to_submission_answer

order_to_answer = order_to_submission_answer


if __name__ == "__main__":
    main()
