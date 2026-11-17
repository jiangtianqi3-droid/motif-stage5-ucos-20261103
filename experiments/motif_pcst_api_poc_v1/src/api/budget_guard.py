"""Durable reservations. An interrupted send remains charged until reconciled."""
import sqlite3
from contextlib import closing

LIMITS = (300, 1_000_000, 60_000)


class BudgetExhausted(RuntimeError):
    pass


class Ledger:
    def __init__(self, path, limits=LIMITS):
        self.path = path
        self.limits = limits
        path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self.connect()) as db, db:
            db.execute("""CREATE TABLE IF NOT EXISTS attempts
                (id INTEGER PRIMARY KEY, key TEXT, input INTEGER, output INTEGER, in_cap INTEGER,
                 out_cap INTEGER, status TEXT, result TEXT, latency REAL)""")
            db.execute('CREATE INDEX IF NOT EXISTS attempts_key ON attempts(key)')

    def connect(self):
        db = sqlite3.connect(self.path)
        db.row_factory = sqlite3.Row
        return db

    @staticmethod
    def totals(db):
        return tuple(db.execute('SELECT count(*),coalesce(sum(input),0),coalesce(sum(output),0) FROM attempts').fetchone())

    def usage(self):
        if not self.path.is_file():
            return (0, 0, 0)
        with closing(self.connect()) as db:
            return self.totals(db)

    def reserve(self, key, input_cap, output_cap):
        with closing(self.connect()) as db, db:
            db.execute('BEGIN IMMEDIATE')
            after = tuple(a+b for a,b in zip(self.totals(db), (1,input_cap,output_cap)))
            if any(x > y for x,y in zip(after,self.limits)):
                raise BudgetExhausted(f'API hard limit: next reserved usage {after} exceeds {self.limits}')
            cur = db.execute('INSERT INTO attempts VALUES(NULL,?,?,?,?,?,?,NULL,NULL)',
                             (key,input_cap,output_cap,input_cap,output_cap,'pending'))
            return cur.lastrowid

    def finish(self, attempt, status, input_used, output_used, result, latency):
        with closing(self.connect()) as db, db:
            row = db.execute('SELECT * FROM attempts WHERE id=?', (attempt,)).fetchone()
            if row is None or row['status'] != 'pending':
                raise RuntimeError('unknown or finalized API attempt')
            a = row['in_cap'] if input_used is None else input_used
            b = row['out_cap'] if output_used is None else output_used
            db.execute('UPDATE attempts SET input=?,output=?,status=?,result=?,latency=? WHERE id=?',
                       (a,b,status,result,latency,attempt))
        if a > row['in_cap'] or b > row['out_cap']:
            raise BudgetExhausted('Provider usage exceeded the request reservation')
