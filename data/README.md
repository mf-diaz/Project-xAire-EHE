# data/

Place the Data in Brief deposit here (CORA.RDR, DOI [10.34810/DATA3672](https://doi.org/10.34810/DATA3672)).
Unzip the downloaded archive inside this folder; it unpacks to `Readme.md` and `xAire_CRD_dataset/`, so that
these two files exist:

```
data/xAire_CRD_dataset/processed/participants.csv
data/xAire_CRD_dataset/processed/games.csv
```

The contents of `xAire_CRD_dataset/` placed directly in `data/` (without the folder) also work. Nothing else
from the deposit is read. Everything in this folder except this file is ignored by git; the data must stay in
the deposit and must not be copied into the repository. To keep the deposit elsewhere, set the `XAIRE_DATA`
environment variable to its root folder.
