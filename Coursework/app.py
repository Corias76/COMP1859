"""Streamlit front end for the SmartEdU academic IR prototype.

Run locally with:  streamlit run app.py
All retrieval logic is in ir_core.py; this file only handles the interface.
"""
import time

import streamlit as st

import ir_core as ir

st.set_page_config(page_title="SmartEdU Search", layout="wide")


@st.cache_resource(show_spinner="Building the search index...")
def load_engine():
    return ir.SearchEngine(ir.load_corpus())


@st.cache_data(show_spinner=False)
def load_meta():
    return ir.load_metadata()


@st.cache_data(show_spinner=False)
def load_choices():
    meta = ir.load_metadata()
    return {
        "courses": ir.list_course_codes(),
        "doc_types": sorted(meta["doc_type"].dropna().unique()),
        "doc_departments": sorted(meta["doc_department"].dropna().unique()),
        "student_departments": ir.list_student_departments(),
    }


engine = load_engine()
meta = load_meta()
choices = load_choices()

st.title("SmartEdU Academic Search")
st.caption("Search unstructured documents (assignments, feedback, emails, syllabi) and "
           "query structured student records in one place.")

tab_docs, tab_records = st.tabs(["Document search", "Student records (SQL)"])

# ------------------------------------------------------------ document search
with tab_docs:
    query = st.text_input("Search documents", placeholder="e.g. inverted index")
    c1, c2, c3, c4 = st.columns(4)
    method_label = c1.selectbox("Method", ["BM25 (ranked)", "Boolean AND (unranked)"])
    course = c2.selectbox("Course", ["All"] + choices["courses"])
    doc_type = c3.selectbox("Document type", ["All"] + choices["doc_types"])
    department = c4.selectbox("Department", ["All"] + choices["doc_departments"])

    if not query.strip():
        st.info("Enter a search query to begin.")
    else:
        start = time.perf_counter()
        results = ir.search_documents(
            engine, meta, query,
            method="bm25" if method_label.startswith("BM25") else "boolean",
            course_code=None if course == "All" else course,
            doc_type=None if doc_type == "All" else doc_type,
            department=None if department == "All" else department,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        if results.empty:
            st.warning("No matching documents. Try fewer words, or remove a filter.")
        else:
            st.write(f"**{len(results)} result(s)** in {elapsed_ms:.1f} ms")
            st.dataframe(results, hide_index=True)
            st.caption("Grade and attendance are empty when the document has no student, "
                       "or the student has no enrolment record for that course.")
            st.subheader("Full text")
            for _, row in results.iterrows():
                with st.expander(f'{row["doc_id"]}: {row["title"]}'):
                    st.text(engine.docs[row["doc_id"]])

# ------------------------------------------------------------ structured data
with tab_records:
    st.caption("Exact-match SQL queries over the student database. "
               "Results are rows that meet the condition, with no relevance score.")
    kind = st.selectbox("Query", ["Low attendance in a course",
                                  "Grades in a course",
                                  "Students by department and GPA"])
    start = time.perf_counter()
    if kind == "Low attendance in a course":
        code = st.selectbox("Course", choices["courses"], key="att_course")
        threshold = st.slider("Attendance below (%)", 0, 100, 75, step=5)
        table = ir.low_attendance(code, threshold)
    elif kind == "Grades in a course":
        code = st.selectbox("Course", choices["courses"], key="grade_course")
        min_grade = st.slider("Minimum grade", 0, 100, 0, step=5)
        table = ir.course_grades(code, min_grade)
    else:
        dept = st.selectbox("Department", choices["student_departments"])
        min_gpa = st.slider("Minimum GPA", 0.0, 4.0, 3.5, step=0.1)
        table = ir.students_in_department(dept, min_gpa)
    elapsed_ms = (time.perf_counter() - start) * 1000

    if table.empty:
        st.warning("No students match these conditions.")
    else:
        st.write(f"**{len(table)} row(s)** in {elapsed_ms:.1f} ms")
        st.dataframe(table, hide_index=True)
