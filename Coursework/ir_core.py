"""Retrieval engine for the SmartEdU academic IR prototype.

All search logic lives here: text preprocessing, the inverted index, Boolean and BM25
retrieval, the join between documents and student records, and the structured SQL queries.
There is no user-interface code in this file, so the Streamlit app, a terminal script or
a notebook can all import the same functions.
"""
import math
import re
import sqlite3
from collections import Counter
from contextlib import closing
from pathlib import Path

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Paths are anchored to this file, so the app works from any working directory
BASE = Path(__file__).resolve().parent
DOCS_DIR = BASE / "data" / "raw_documents"
DB_PATH = BASE / "data" / "structured" / "smartedu.db"

# Created once at import, not on every call
try:
    STOPWORDS = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords", quiet=True)
    STOPWORDS = set(stopwords.words("english"))
STEMMER = PorterStemmer()


# ---------------------------------------------------------------- text side
def preprocess(text):
    """Lowercase, strip punctuation, drop stopwords, Porter-stem. Returns a list of terms."""
    text = re.sub(r"[^\w\s]", "", text.lower())
    return [STEMMER.stem(w) for w in text.split() if w not in STOPWORDS]


def load_corpus(root=DOCS_DIR):
    """Read every .txt under root into {doc_id: text}, where doc_id is the DOCnnn number."""
    corpus = {}
    for path in sorted(Path(root).rglob("*.txt")):
        m = re.search(r"DOC\d+", path.name)
        doc_id = m.group() if m else path.stem
        if doc_id in corpus:
            raise ValueError(f"Two files share the document id {doc_id}")
        corpus[doc_id] = path.read_text(encoding="utf-8", errors="replace")
    return corpus


class SearchEngine:
    """Inverted index plus BM25 statistics, built once from {doc_id: text}."""

    def __init__(self, docs, k1=1.5, b=0.75):
        if not docs:
            raise ValueError("No documents were loaded; check the data/raw_documents folder")
        self.docs = docs
        self.k1, self.b = k1, b
        self.tokens = {d: preprocess(t) for d, t in docs.items()}
        self.n = len(self.tokens)
        self.avgdl = sum(len(t) for t in self.tokens.values()) / self.n
        self.tf = {d: Counter(t) for d, t in self.tokens.items()}
        self.df = Counter(term for t in self.tokens.values() for term in set(t))
        self.index = {}                                   # inverted index: term -> {doc ids}
        for d, t in self.tokens.items():
            for term in set(t):
                self.index.setdefault(term, set()).add(d)

    def idf(self, term):
        # "+ 1" keeps the value positive even for terms found in most documents
        return math.log((self.n - self.df[term] + 0.5) / (self.df[term] + 0.5) + 1)

    def boolean_search(self, query):
        """Documents containing every query term (AND), sorted by id. Unranked."""
        terms = preprocess(query)
        if not terms:
            return []
        return sorted(set.intersection(*(self.index.get(t, set()) for t in terms)))

    def bm25_search(self, query, top_k=None):
        """Documents containing any query term, as (doc_id, score), best first."""
        terms = preprocess(query)
        scores = {}
        for d, tokens in self.tokens.items():
            dl = len(tokens)
            score = 0.0
            for t in terms:
                f = self.tf[d].get(t, 0)
                if f:
                    score += self.idf(t) * f * (self.k1 + 1) / (
                        f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            if score > 0:
                scores[d] = score
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        return ranked[:top_k] if top_k else ranked


# ----------------------------------------------------------- structured side
META_QUERY = """
SELECT m.doc_id, m.title, m.doc_type, m.department AS doc_department,
       s.student_id, s.first_name, s.last_name, s.gpa,
       c.course_id, c.course_code, c.course_name, c.instructor,
       e.semester, e.final_grade, e.letter_grade,
       a.attendance_rate
FROM documents_metadata m
LEFT JOIN students    s ON m.student_id = s.student_id
LEFT JOIN courses     c ON m.course_id  = c.course_id
LEFT JOIN enrollments e ON m.student_id = e.student_id AND m.course_id = e.course_id
LEFT JOIN attendance  a ON m.student_id = a.student_id AND m.course_id = a.course_id
"""


def run_sql(query, params=(), db_path=DB_PATH):
    # A new connection per call keeps this safe across Streamlit's threads
    with closing(sqlite3.connect(db_path)) as con:
        return pd.read_sql(query, con, params=params)


def load_metadata(db_path=DB_PATH):
    """One row per document, joined to its student, course, enrolment and attendance."""
    return run_sql(META_QUERY, db_path=db_path).set_index("doc_id")


def low_attendance(course_code, threshold=75):
    return run_sql("""
        SELECT s.student_id, s.first_name, s.last_name, a.attendance_rate
        FROM attendance a
        JOIN students s ON a.student_id = s.student_id
        JOIN courses  c ON a.course_id  = c.course_id
        WHERE c.course_code = ? AND a.attendance_rate < ?
        ORDER BY a.attendance_rate
    """, (course_code, threshold))


def course_grades(course_code, min_grade=0):
    return run_sql("""
        SELECT s.student_id, s.first_name, s.last_name, e.semester, e.final_grade, e.letter_grade
        FROM enrollments e
        JOIN students s ON e.student_id = s.student_id
        JOIN courses  c ON e.course_id  = c.course_id
        WHERE c.course_code = ? AND e.final_grade >= ?
        ORDER BY e.final_grade DESC
    """, (course_code, min_grade))


def students_in_department(department, min_gpa=0):
    return run_sql("""
        SELECT student_id, first_name, last_name, enrollment_year, gpa
        FROM students
        WHERE department = ? AND gpa >= ?
        ORDER BY gpa DESC
    """, (department, min_gpa))


def list_course_codes():
    return run_sql("SELECT course_code FROM courses ORDER BY course_code")["course_code"].tolist()


def list_student_departments():
    return run_sql("SELECT DISTINCT department FROM students ORDER BY department")["department"].tolist()


# ------------------------------------------------------ both sides together
RESULT_COLUMNS = ["doc_id", "score", "title", "doc_type", "course",
                  "student", "grade", "attendance"]


def search_documents(engine, meta, query, method="bm25", course_code=None,
                     doc_type=None, department=None, top_k=50):
    """Text search (BM25 or Boolean AND), then filter on the linked metadata.

    Scoring happens before filtering, so a document's score never depends on the filter.
    """
    if method == "bm25":
        ranked = engine.bm25_search(query)
    else:
        ranked = [(d, None) for d in engine.boolean_search(query)]

    rows = []
    for doc_id, score in ranked:
        r = meta.loc[doc_id] if doc_id in meta.index else pd.Series(dtype=object)
        if course_code and r.get("course_code") != course_code:
            continue
        if doc_type and r.get("doc_type") != doc_type:
            continue
        if department and r.get("doc_department") != department:
            continue
        no_student = pd.isna(r.get("first_name"))
        rows.append({
            "doc_id": doc_id,
            "score": None if score is None else round(score, 3),
            "title": r.get("title"),
            "doc_type": r.get("doc_type"),
            "course": r.get("course_code"),
            "student": "(course document)" if no_student else f'{r["first_name"]} {r["last_name"]}',
            "grade": r.get("letter_grade"),
            "attendance": r.get("attendance_rate"),
        })
    return pd.DataFrame(rows, columns=RESULT_COLUMNS).head(top_k)
