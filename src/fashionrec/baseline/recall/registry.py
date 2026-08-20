"""The immutable recall set used by the stable baseline."""

BASELINE_RULE_CHANNELS = ("popular", "category_popular", "item2item")
BASELINE_CHANNELS = (*BASELINE_RULE_CHANNELS, "sasrecf")
