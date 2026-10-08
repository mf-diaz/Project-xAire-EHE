# data/

Place the Data in Brief deposit here (CORA.RDR; DOI: [[TO CONFIRM: DOI of the final deposit]]).
Unzip the archive and copy its contents into this folder, so that these two files exist:

```
data/processed/participants.csv
data/processed/games.csv
```

Nothing else from the deposit is read. Everything in this folder except this file is ignored by git;
the data must stay in the deposit and must not be copied into the repository. To keep the deposit
elsewhere, set the `XAIRE_DATA` environment variable to its root folder.
