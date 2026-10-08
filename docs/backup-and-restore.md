# Backup and restore

This page explains what is backed up, how to back it up, and how to rebuild everything on a new, empty machine.

## What is backed up and where

| What | How | Where |
|---|---|---|
| Seven DVC-tracked files (the enriched PaySim parquet, model files, calibrator, two small config files) | `dvc push` | a DVC remote named `backup`, which is a folder outside the project |
| MLflow data (`mlflow.db` and `mlruns/`) | `python scripts/backup_mlflow.py --backup-dir <folder>` | a new folder `mlflow-backup-<UTC time>` inside the folder you give |

The remote's address is stored in `.dvc/config.local`, which Git ignores. It is never committed, because it is specific to your machine.

## How to back up

```
dvc remote add --local -d backup <path to an existing empty folder>   # once
dvc push                                                              # whenever DVC files change
dvc status --cloud                                                    # should say cache and remote are in sync
python scripts/backup_mlflow.py --backup-dir <folder>                 # whenever you want a new MLflow backup
```

The MLflow backup copies the database with SQLite's own backup function, copies `mlruns/`, and writes `manifest.json` with the size and SHA-256 of every file. It never overwrites an earlier backup.

## How to restore on a new, empty machine

1. `git clone https://github.com/adithya-313/PROJECT--ashFALL` and `cd` into it.
2. Make the backup folder available, then point DVC at it: `dvc remote add --local -d backup <path to dvc-remote>`.
3. `dvc pull` rebuilds the seven tracked files and checks their hashes.
4. `cp .env.example .env` (and change the settings if you need to). `.env` is not in Git or DVC.
5. Restore MLflow into a new empty folder: `python scripts/restore_mlflow.py --backup-folder <one mlflow-backup-... folder> --target-dir <new folder>`. It checks every file against the manifest first, then rewrites the absolute folder paths stored in the database to the new location (MLflow stores full paths such as `file:C:/Users/...`). Put `mlflow.db` and `mlruns/` where the scripts expect them (the repo root), or set `MLFLOW_TRACKING_URI`.
6. Rebuild the training splits (they are not in DVC): `python scripts/data_split.py`.
7. `data/raw/paysim.csv` is not in Git or DVC. Download the PaySim dataset yourself and put it at that path (nothing in the repo records where it came from).

To prove steps 1-3 work, run `python scripts/verify_dvc_restore.py --dvc-remote-dir <path to dvc-remote>`. It clones the repo into a temporary folder, runs `dvc pull` there, and compares every file with the one in your working copy.

## Not backed up

`data/raw/paysim.csv`, `data/processed/*`, `data/enriched/*.bak`, the copy of the model in `serving/model_repository/`, and `.env`.

## The limit: one disk

The default remote is a folder on the same disk as the project. It protects against deleting or damaging files in the project. It does **not** protect against the disk failing or the laptop being lost. For that, copy the backup folders to another disk, or use an online remote (next section).

## Switching the remote to AWS S3 later (not tested)

DVC remotes are chosen by their address. The commands below come from `dvc remote add --help`, `dvc remote modify --help` and `dvc push --help`:

```
dvc remote add -d <name> <url>        # the url decides the storage type
dvc remote modify <name> <option> <value>
dvc push -r <name>                    # push to a named remote
```

What the S3 address and options look like, and how to keep credentials out of Git (use `--local`), should be taken from the DVC documentation. DVC also needs its S3 support package installed (here `pip show dvc-s3` says it is not installed). None of this has been tested in this project. Never put real keys in a file that Git tracks.
