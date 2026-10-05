# Data availability

Three daily OHLCV panels were used by all five historical studies.

| Panel | Source | Recorded scope | Redistribution |
|---|---|---|---|
| SP500 | https://www.kaggle.com/datasets/andrewmvd/sp-500-stocks | 502 current constituents, 2010-01-04 to 2024-12-20 | Source metadata reports CC0; the panel has survivorship bias |
| KDD17 | https://github.com/fulifeng/Adv-ALSTM | 50 symbols, 2007-01-03 to 2016-12-30 | Obtain from the author release; underlying market-data rights remain with the source |
| ACL18 | https://github.com/yumoxu/stocknet-dataset | 88 symbols, 2012-09-04 to 2017-09-01 | Obtain from the author release; underlying market-data rights remain with the source |

Expected processed-file hashes are recorded in `DATASETS.json`. The SP500
panel uses current constituents and must not be described as a
survivorship-free historical index universe. Per-date eligibility is smaller
than the listed symbol count because sufficient past observations are required.
