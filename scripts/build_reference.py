"""Save the training reference profile used by live drift monitoring."""

from fraud import features, monitoring

df = features.load()
train = df[df.split == "train"]
monitoring.save_profile(train)
print(f"wrote {monitoring.PROFILE} from {len(train):,} training rows")
