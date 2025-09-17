"""
Streamlit dashboard landing page.
"""
import os
import threading
import streamlit as st
import time

from pipeline import run_pipeline


def _kickoff_background_pipeline_once():
    done_flag = os.path.join("/app/data/outputs", ".pipeline_done")
    if os.path.exists(done_flag):
        st.session_state["pipeline_done"] = True
        return
    if "pipeline_started" in st.session_state:
        return
    st.session_state["pipeline_started"] = True

    def _run():
        try:
            run_pipeline(
                data_yaml="/app/data/traffic-detection-project/data.yaml",
                project="/app/data/outputs",
                train_name="yolo_train",
                eval_name="yolo_eval",
                imgsz=640,
            )
            st.session_state["pipeline_done"] = True
            # persist completion across page reloads
            try:
                with open(done_flag, "w") as f:
                    f.write("done")
            except Exception:
                pass
        except Exception as e:
            st.session_state["pipeline_error"] = str(e)

    th = threading.Thread(target=_run, daemon=True)
    th.start()


def main():
    st.set_page_config(page_title="Traffic Detection Dashboard", layout="centered")
    st.title("Traffic Detection Project")

    _kickoff_background_pipeline_once()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Stage", "Training/Eval")
    with col2:
        status = "Done" if st.session_state.get("pipeline_done") else ("Error" if st.session_state.get("pipeline_error") else "Running")
        st.metric("Status", status)
    with col3:
        st.metric("Epochs", "1")

    if st.session_state.get("pipeline_error"):
        st.error(f"Auto training/evaluation failed: {st.session_state['pipeline_error']}")
    elif st.session_state.get("pipeline_done"):
        st.success("Training and evaluation completed. See the Evaluation page for reports.")
        if st.button("Go to Evaluation", type="primary"):
            st.switch_page("pages/3_Evaluation.py")
    else:
        with st.spinner("Training/evaluation in progress... this is a minimal 1-epoch CPU run"):
            time.sleep(0.1)
        st.info("Reports will appear on the Evaluation page when ready.")

    st.divider()
    st.write("Use the sidebar pages to explore: About Data, Training, Evaluation, etc.")
    st.page_link("pages/1_About_Data.py", label="About Data", icon="📊")
    st.page_link("pages/3_Evaluation.py", label="Evaluation", icon="✅")


if __name__ == "__main__":
    main()
