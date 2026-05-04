# tweet-harvest/harvest.py

import os
import tweepy
import pandas as pd

class TwitterHarvester:
    def __init__(self, username, credentials, output_dir="tweets-data", max_tweets=100):
        self.username = username
        self.bearer_token = credentials.get("bearer_token")
        self.output_dir = output_dir
        self.max_tweets = max_tweets

        # Setup tweepy client (API v2)
        self.client = tweepy.Client(bearer_token=self.bearer_token, wait_on_rate_limit=True)

        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

    def harvest(self):
        try:
            user = self.client.get_user(username=self.username)
            user_id = user.data.id

            tweets = tweepy.Paginator(
                self.client.get_users_tweets,
                id=user_id,
                tweet_fields=["created_at", "lang", "public_metrics"],
                max_results=100
            ).flatten(limit=self.max_tweets)

            tweet_data = []
            for tweet in tweets:
                tweet_data.append({
                    "id": tweet.id,
                    "created_at": tweet.created_at,
                    "full_text": tweet.text,
                    "like_count": tweet.public_metrics["like_count"],
                    "retweet_count": tweet.public_metrics["retweet_count"]
                })

            if not tweet_data:
                print("No tweets found.")
                return

            df = pd.DataFrame(tweet_data)
            output_path = os.path.join(self.output_dir, f"{self.username}.csv")
            df.to_csv(output_path, index=False, encoding="utf-8-sig")
            print(f"Saved {len(df)} tweets to {output_path}")

        except Exception as e:
            print(f"Error while harvesting tweets: {e}")
            raise
