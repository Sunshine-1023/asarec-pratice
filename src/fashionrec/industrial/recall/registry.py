"""The immutable recall set used by the industrial application."""

INDUSTRIAL_RULE_CHANNELS = (
    "popular",
    "category_popular",
    "item2item",
    "repurchase",
    "style",
    "content",
)
INDUSTRIAL_CHANNELS = (*INDUSTRIAL_RULE_CHANNELS, "sasrecf")
