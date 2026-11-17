import json
from contextlib import closing


def history(ledger, key):
    with closing(ledger.connect()) as db:
        rows = db.execute('SELECT * FROM attempts WHERE key=? ORDER BY id', (key,)).fetchall()
    return [dict(row) for row in rows]


def completed(rows):
    cached = next((row for row in rows if row['status'] in ('valid', 'abstain')), None)
    return json.loads(cached['result']) if cached else None
