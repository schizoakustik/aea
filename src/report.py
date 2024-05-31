from datetime import timedelta
import subprocess
import numpy as np
import pandas as pd
import skimage as ski
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import string
import os
import os.path
from jinja2 import Environment, FileSystemLoader

env = Environment(loader=FileSystemLoader('src/templates'))

# Emotion labels
emotion_labels = list(map(str.lower, ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']))

# Custom colormaps
cmaps = {
    'angry': ListedColormap([
        [1., 0., 0., .01], [1., 0., 0., .25], [1., 0., 0., .5]]),
    'happy': ListedColormap([
        [0., 1., 0., .01], [0., 1., 0., .25], [0., 1., 0., .5]]),
    'sad': ListedColormap([
        [0., 0., 1., .01], [0., 0., 1., .25], [0., 0., 1., .5]]),
    'neutral': ListedColormap([
        [0., 0., 0., .01], [0., 0., 0., .25], [0., 0., 0., .5]]),
    'disgust': ListedColormap([
        [1., 0.5, 0.5, .01], [1., 0.5, 0.5, .25], [1., 0.5, 0.5, .5]]),
    'fear': ListedColormap([
        [1., 0., 1., .01], [1., 0., 1., .25], [1., 0., 1., .5]]),
    'surprise': ListedColormap([
        [1., 1., 0., .01], [1., 1., 0., .25], [1., 1., 0., .5]])
}

hues = {e:val(2) for e, val in cmaps.items()}

# Create custom legend for emotions plot
handles = []
for emotion in emotion_labels:
    handles.append(Patch(color=cmaps[emotion](2), label=emotion))

# Plot faces
def plot_face_positions(df, px):
    for _, row in df.iterrows():
        px[row.y:row.y+row.height, row.x:row.x+row.width] += 1
    return px

# Create empty array for face plotting
px = np.zeros(shape=(360, 640))

class Report:
    """ Report template """
    def __init__(self, file, outfile, detector, confidence, number_of_frames, skip, _time, results, frame_cap) -> None:
        self.file = file
        self.outfile = outfile
        self.detector = detector
        self.confidence = confidence
        self.number_of_frames = number_of_frames
        self.skip = skip
        self._time = _time
        self.results = results
        self.frame_cap = frame_cap

    def generate_report(self):
        print('Generating report...')
        print("Creating temporary directory...")
        os.mkdir('tmp')
        print('Preparing data...')
        file = os.path.splitext(os.path.basename(self.file))[0]
        frame = self.frame_cap
        outfile = self.outfile
        melted = pd.melt(self.results, id_vars=["frame", "x", "y", "width", "height"], value_vars=emotion_labels, var_name="emotion", value_name="_count")
        melted = melted[melted._count > 0].reset_index()
        
        # Process screenshot of random frame
        frame = ski.color.rgb2gray(self.frame_cap)
        edges = ski.filters.sobel(frame)
        thresh = ski.filters.threshold_otsu(edges)
        img = edges < thresh

        img = img.astype(float)
        img[img== True] = np.nan
        
        table = """|Emotion|Count|
|:--|--:|
"""
        for emotion, _count in melted.groupby('emotion')._count.apply(np.sum).items():
            table += f"|{emotion}|{_count}|\n" 
        
        print('Generating plots...')
        # Plot emotions on frame
        fig = plt.figure(figsize=(8, 4), dpi=96)
        ax = fig.gca()

        for emotion, row in melted.groupby('emotion'):
            sns.heatmap(plot_face_positions(row, px.copy()), xticklabels=False, yticklabels=False, cbar=False, ax = ax, cmap=cmaps[emotion])
        sns.heatmap(img, xticklabels=False, yticklabels=False, cmap='gray', cbar=False, ax = ax)
        fig.legend(handles=handles, loc='outside lower center', ncols=2, borderaxespad=0.)
        ax.set(title="Position and magnitude of emotions")
        plt.savefig("tmp/plot_01.png")

        # Emotion timelines
        frame_counts = melted.groupby(['frame', 'emotion'])._count.apply(np.sum).to_frame().reset_index()
        fg = sns.relplot(kind='line', data=frame_counts, x="frame", y="_count", hue = "emotion", palette=hues, height=3, aspect=3).set_axis_labels(x_var="Frame", y_var="Count")
        fg.figure.suptitle("Emotion timeline")
        plt.savefig("tmp/plot_02.png")

        fg = sns.relplot(kind='line', data=frame_counts, x="frame", y="_count", hue='emotion', row = "emotion", palette=hues, facet_kws={'sharey': False, 'sharex': False}, height=2, aspect=5, legend=False).set_axis_labels(x_var="Frame", y_var="Count")
        plt.savefig("tmp/plot_03.png")

        print('Generating output...')
        template = env.get_template('report.qmd')
        output = template.render(file=file, outfile=outfile, detector=self.detector, confidence=self.confidence, number_of_frames=self.number_of_frames, skip=self.skip, time=timedelta(seconds=self._time), table=table)
        with open(f"report_{file}.qmd", "w") as f: f.write(output)
        print('Rendering PDF...')
        os.mkdir('report')
        subprocess.run(['quarto', 'render', f'report_{file}.qmd', '--to', 'pdf', '--output-dir', 'report', '--quiet'])
        print('Removing temporary directory...')
        for f in os.scandir('tmp'):
            os.remove(f)
        os.rmdir('tmp')
        os.remove(f"report_{file}.qmd")
        print(f'Done! Report saved as ./report_{file}.pdf.')
        return False

if __name__ == "__main__":
    results = pd.read_csv("results.csv")
    frame = ski.io.imread("screenshot.png")
    r = Report("some_dir/video1.mp4", "video_out.mp4", "cv", .5, 703, False, 830, results, frame)
    r.generate_report()