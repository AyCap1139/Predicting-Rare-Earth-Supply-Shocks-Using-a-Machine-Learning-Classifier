import feedparser
import pandas as pd
from datetime import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import time

analyzer = SentimentIntensityAnalyzer()

KEYWORDS = [
    "rare earth", "rare-earth", "export license",
    "export control", "strategic minerals", "export quota",
    "gallium", "germanium", "dysprosium", "neodymium",
    "critical minerals", "REE"
]

RSS_FEEDS = {
    "Xinhua_World":     "https://english.news.cn/rss/world.xml",
    "Xinhua_Business":  "https://english.news.cn/rss/business.xml",
    "GlobalTimes":      "https://www.globaltimes.cn/rss/outbrain.xml",
    "PeoplesDaily":     "https://en.people.cn/rss/90777.xml",
    "Reuters_Commodities": "https://feeds.reuters.com/reuters/companyNews",
}

def contains_keyword(text):
    text_lower = text.lower()
    return any(k.lower() in text_lower for k in KEYWORDS)

def get_sentiment(text):
    return analyzer.polarity_scores(text)['compound']

results = []

for source_name, feed_url in RSS_FEEDS.items():
    print(f"Fetching {source_name}...")
    try:
        feed = feedparser.parse(feed_url)
        count = 0
        for entry in feed.entries:
            title   = entry.get("title", "")
            summary = entry.get("summary", "")
            text    = title + " " + summary

            if contains_keyword(text):
                # parse date
                published = entry.get("published", "")
                try:
                    date = entry.published_parsed
                    date_str = f"{date.tm_year}{date.tm_mon:02d}{date.tm_mday:02d}"
                except:
                    date_str = datetime.today().strftime("%Y%m%d")

                results.append({
                    "date":      date_str,
                    "source":    source_name,
                    "headline":  title,
                    "summary":   summary[:200],
                    "sentiment": get_sentiment(text),
                    "url":       entry.get("link", "")
                })
                count += 1

        print(f"  {source_name}: {count} REE hits out of {len(feed.entries)} articles")
        time.sleep(1)

    except Exception as e:
        print(f"  {source_name} error: {e}")

# ── SAVE ──────────────────────────────────────────────────────────
if results:
    df = pd.DataFrame(results)
    df = df.sort_values("date", ascending=False)
    print(f"\nTotal REE headlines found: {len(df)}")
    print(df[["date","source","headline","sentiment"]].head(15))
    df.to_csv("chinese_media_raw.csv", index=False)
    print("\nSaved to chinese_media_raw.csv")
else:
    print("\nNo REE headlines found in current RSS feeds")
    print("This is normal — RSS only shows recent articles (last 20-50)")
    print("We will handle historical data in Step 5 using GDELT as proxy")