import streamlit as st

done = False
        
def progress(p):
    st.progress(p)

def analyze():
    with st.spinner("Analyzing..."):
        from src.analyzer import Analyzer
        a = Analyzer(detector=detector.lower())
        done = a.analyze(file, outfile, False, save_video, skip, True, confidence)

st.title("Audience emotion analyzer")

st.header("What do I do?")
st.write("Upload a video file for analysis. Change any settings you want changed, and press the Analyze button.")

file = st.file_uploader('Select file to upload...')
col1, col2 = st.columns(2)
save_video = col1.toggle("Save analyzed video")
if save_video:
    outfile = col2.text_input('Filename:', value='output.mp4')
else:
    outfile = None
col3, col4 = st.columns(2)
detector = col3.selectbox('Face detector:', ("CV", "MTCNN"))
col4.info("CV is faster but less accurate.  \nMTCNN is considerably slower, but more accurate when detecting faces.")
col5, col6 = st.columns(2)
skip = col5.number_input('Frames to skip', 0, None, 100)
col6.info("Skip frames to speed up analysis.  \n0 means all frames will be analyzed.")
col7, col8 = st.columns(2)
confidence = col5.number_input('Emotion analyzer confidence', 0., 1., 0.5)
col6.info("Return predictions above this probability.")

st.button("Analyze", on_click=analyze)

if done:
    st.download_button("Download report", f'report/report_{file.name}.pdf')