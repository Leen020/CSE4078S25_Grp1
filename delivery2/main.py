import pandas as pd
import tweepy
import time
import os
from tqdm import tqdm
from datetime import datetime


# Configuration
BEARER_TOKEN = "AAAAAAAAAAAAAAAAAAAAALiU0wEAAAAA6mKGwNxTbzIuQK19lMzTMectRUM%3D95ormOv0VTg7KVJbz9cNKKYCdV88IG4yHt39qvxM0wC47E2ACJ"
DATASET_PATH = "../../../train.txt"
OUTPUT_FILE = "hydrated_tweets.csv"
MISSING_FILE = "missing_tweets.txt"
BATCH_SIZE = 100
COMPLETED_BATCHES_FILE = "completed_batches.txt"
LOG_FILE = "hydration_log.txt"




def log_message(message):
    """Log messages with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}\n"
    print(log_entry.strip())
    with open(LOG_FILE, 'a') as f:
        f.write(log_entry)


# Initialize Twitter client
client = tweepy.Client(
    bearer_token=BEARER_TOKEN,
    wait_on_rate_limit=True
)


def load_data():
    """Load and prepare tweet IDs"""
    log_message("Reading tweet IDs...")
    df = pd.read_csv(DATASET_PATH, sep="\t")
    tweet_ids = df['tweet_id'].unique().tolist()
    log_message(f"Found {len(tweet_ids)} unique tweet IDs.")

    if os.path.exists(OUTPUT_FILE):
        hydrated_df = pd.read_csv(OUTPUT_FILE)
        hydrated_ids = hydrated_df['tweet_id'].tolist()
        log_message(f"{len(hydrated_ids)} tweets already hydrated.")
    else:
        hydrated_df = pd.DataFrame(columns=['tweet_id', 'text', 'created_at', 'author_id'])
        hydrated_ids = []

    remaining_ids = [i for i in tweet_ids if i not in hydrated_ids]
    log_message(f"{len(remaining_ids)} tweet IDs remaining to process.")

    return remaining_ids, hydrated_df


def process_batch(batch, hydrated_df):
    """Process a single batch of tweet IDs"""
    missing_in_batch = []
    new_rows = []

    try:
        response = client.get_tweets(
            ids=batch,
            tweet_fields=["created_at", "author_id"]
        )

        if response.data:
            for tweet in response.data:
                new_rows.append({
                    'tweet_id': tweet.id,
                    'text': tweet.text,
                    'created_at': tweet.created_at,
                    'author_id': tweet.author_id
                })

        found_ids = [tweet.id for tweet in response.data] if response.data else []
        missing_in_batch = list(set(batch) - set(found_ids))

    except tweepy.TweepyException as e:
        log_message(f"Error in batch processing: {str(e)}")
        if "429" in str(e):  # Rate limit error
            wait_time = 900  # 15 minutes
            log_message(f"Rate limit hit. Sleeping for {wait_time} seconds...")
            time.sleep(wait_time)
            return process_batch(batch, hydrated_df)  # Retry after waiting
        missing_in_batch = batch  # Consider entire batch missing on error

    return new_rows, missing_in_batch


def main():
    remaining_ids, hydrated_df = load_data()
    missing_ids = []

    if not remaining_ids:
        log_message("No tweets left to hydrate. Exiting.")
        return

    log_message("Starting hydration process...")

    for i in tqdm(range(0, len(remaining_ids), BATCH_SIZE),
                  desc="Hydrating tweets",
                  unit="batch",
                  ncols=80):
        batch = remaining_ids[i:i + BATCH_SIZE]

        new_rows, missing_in_batch = process_batch(batch, hydrated_df)
        missing_ids.extend(missing_in_batch)

        if new_rows:
            hydrated_df = pd.concat([hydrated_df, pd.DataFrame(new_rows)], ignore_index=True)
            hydrated_df.to_csv(OUTPUT_FILE, index=False)

        # Log completed batch
        with open(COMPLETED_BATCHES_FILE, 'a') as f:
            for tid in batch:
                f.write(f"{tid}\n")

        # Small sleep between batches to avoid hitting limits
        time.sleep(1)

    if missing_ids:
        log_message(f"Saving {len(missing_ids)} missing tweet IDs...")
        with open(MISSING_FILE, 'w') as f:
            for tid in missing_ids:
                f.write(f"{tid}\n")

    log_message(f"Process completed. Results saved to {OUTPUT_FILE}")
    if missing_ids:
        log_message(f"Missing tweets saved to {MISSING_FILE}")


if __name__ == "__main__":
    main()