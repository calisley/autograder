from collections import defaultdict
import tiktoken 

class TokenTracker:
    def __init__(self, encoding="o200k_base"):
        self.tokenizer = tiktoken.get_encoding(encoding)
        self.process_totals = defaultdict(int)
        self.grand_total = 0

    def encode(self, text):
        """Encode text into tokens using the specified encoding."""
        return self.tokenizer.encode(text)
    
    def count_tokens(self, text):
        """Count the number of tokens in the given text."""
        return len(self.encode(text))

    def add(self, process_name, value):
        """
        Add tokens to the tracker.
        If value is an int, treat as token count.
        If value is a str, count tokens in the string.
        """
        if isinstance(value, int):
            tokens = value
        elif isinstance(value, str):
            tokens = self.count_tokens(value)
        else:
            raise ValueError("add() expects an int (token count) or str (text to count tokens)")
        self.process_totals[process_name] += tokens
        self.grand_total += tokens

    def print_process(self, process_name):
        print(f"Total tokens for {process_name}: {self.process_totals[process_name]}")

    def print_grand_total(self):
        print(f"GRAND TOTAL tokens for autograder run: {self.grand_total}")

# Singleton instance
token_tracker = TokenTracker() 