"""
分析相关的配置 / 常量。

目前只暴露 FEATURE_COLS，供 app.py 使用。
"""

# 你的 10 个特征，供模型和接口引用
FEATURE_COLS = [
    "avg_comment_scraped",
    "avg_danmaku",
    "avg_length",
    "avg_play",
    "comment_repetition",
    "danmaku_missing_rate",
    "med_danmaku",
    "med_play",
    "std_length",
    "upload_freq",
]
