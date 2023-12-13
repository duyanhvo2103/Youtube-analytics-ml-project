from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
import pickle
import re
from typing import List
import os

# NLP libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
# nltk.download('stopwords')
# nltk.download('punkt')

# ------------------------------ Prepare ------------------------------

# Load the saved KMeans model
with open(f"./data/models/my_best_model.pkl", "rb") as model_file:
    my_best_model = pickle.load(model_file)

# Preprocess the data: convert to lowercase, remove special characters, and stopwords
stop_words = set(stopwords.words('english'))


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = [token for token in text.split() if token not in stop_words]
    return ' '.join(tokens)


# Read the data
clustered_df = pd.read_csv("./data/processed/video_data_clustered.csv")
clustered_df['clean_title'] = clustered_df['title'].apply(preprocess_text)

# Create a TF-IDF vectorizer to vectorize the text
# Reference: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
vectorizer = TfidfVectorizer(lowercase=True, stop_words='english',
                             max_features=20_000,  # !reduce number of features
                             dtype=np.float32)
vectorizer.fit(clustered_df['clean_title'])


def _recommend_top_n_videos(cluster_id: int, top_n: int = 10) -> pd.DataFrame:
    # Return top `n` video titles in the same cluster

    metric_to_sort: str = 'score'
    threshold_for_shuffle: int = 10

    # Define a dataframe to store the recommended videos
    recommended_videos = pd.DataFrame(
        columns=['channelTitle', 'title', 'video_id']
    )

    # Define a set to store the channels that have been recommended
    set_of_channels = set()

    # Get all videos in the same cluster
    #   Then sort them by their views
    video_in_cluster = clustered_df.query(f"cluster_id == {cluster_id}")\
        .sort_values(by=metric_to_sort, ascending=False)

    # Iterate through each video in the cluster
    def _get_n_videos_from_different_channel(video_df: pd.DataFrame,
                                             n_videos: int) -> None:
        n_completed = 0
        for i, video in video_df.iterrows():
            channelTitle = video['channelTitle']
            videoTitle = video['title']
            video_id = video['video_id']

            # If the channel has already been recommended, skip it
            if channelTitle in set_of_channels:
                continue

            # Otherwise, add it to the list of recommended channels
            recommended_videos.loc[len(recommended_videos)] \
                = [channelTitle, videoTitle, video_id]

            # Add the channel to the set
            set_of_channels.add(channelTitle)

            # If we have enough channels, stop
            n_completed += 1
            if n_completed >= n_videos:
                return

    if top_n >= threshold_for_shuffle:
        # Get first half of the videos
        first_half = top_n // 2
        _get_n_videos_from_different_channel(video_df=video_in_cluster,
                                             n_videos=first_half)
        # Get second half of the videos -> randomly shuffle the videos
        video_in_cluster = video_in_cluster.sample(frac=1)
        second_half = top_n - first_half
        _get_n_videos_from_different_channel(video_df=video_in_cluster,
                                             n_videos=second_half)
    else:
        _get_n_videos_from_different_channel(video_df=video_in_cluster,
                                             n_videos=top_n)

    # Convert the `video_id` to a `URL`
    recommended_videos['URL'] = recommended_videos['video_id']\
        .apply(lambda x: f"https://www.youtube.com/watch?v={x}")
    recommended_videos.drop(columns=['video_id'], inplace=True)

    return recommended_videos


def _recommend_top_n_tags(cluster_id: int, top_n: int = 10) -> np.ndarray:
    print(f"Getting top {top_n} tags of cluster {cluster_id}...")
    tags_in_cluster = clustered_df.query(f"cluster_id == {cluster_id}")
    tags_in_cluster = tags_in_cluster.sort_values(
        by='viewCount', ascending=False
    )['tags'].head(4*top_n)

    # Create a dataframe to store the one-hot encoding of tags
    one_hot_df = tags_in_cluster.str.get_dummies(sep='|')
    if "(notag)" in one_hot_df.columns:
        one_hot_df = one_hot_df.drop("(notag)", axis=1)
    # Sum the one-hot encoding of tags
    all_tags = one_hot_df.sum().sort_values(ascending=False).head(4*top_n)

    # Preprocess all tags
    preprocessed_tags = pd.Series(all_tags.index).apply(preprocess_text)
    preprocessed_tags = preprocessed_tags[preprocessed_tags != ""]

    # Get only unique tags
    unique_tags = pd.unique(preprocessed_tags)
    # Get some random tags
    np.random.shuffle(unique_tags)
    return unique_tags[:top_n]


def _add_link(html_string: str) -> str:
    # Reference: https://www.semrush.com/blog/html-link-code/
    pattern = r'<td>(https://www.youtube.com/watch\?v=[a-zA-Z0-9_-]+)</td>'
    replacement = r'<td><a href="\1">\1</a></td>'
    return re.sub(pattern=pattern, repl=replacement,
                  string=html_string)


def _modify_html_template(html_string: str) -> str:
    # Add links to the video URLs
    html_string = _add_link(html_string)

    # Reference: https://www.educative.io/answers/what-is-the-resub-function-in-python
    pattern = r'<tr style="text-align: right;">'
    replacement = r'<tr style="text-align: center;">'
    return re.sub(pattern=pattern, repl=replacement,
                  string=html_string)


# ------------------------------ APP ------------------------------
# Reference: https://stackoverflow.com/questions/52644035/how-to-show-a-pandas-dataframe-into-a-existing-flask-html-table
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/search', methods=['POST'])
def search():
    if request.method == 'POST':
        text = request.form['text']

        # Preprocess the input text
        text_data = pd.Series(text)
        text_data = text_data.apply(preprocess_text)
        text_transformed = vectorizer.transform(text_data)

        # Use the trained model to predict cluster for the input text
        cluster = my_best_model.predict(text_transformed)

        # Additional logic to recommend videos based on the cluster
        recommended_videos = _recommend_top_n_videos(cluster, top_n=10)
        recommended_videos.index += 1
        recommended_videos.columns = ['Channel', 'Video title', 'URL']

        # Additional logic to recommended tags based on the cluster
        recommended_tags = _recommend_top_n_tags(cluster, top_n=5)
        recommended_tags = [
            "\"" + "\", \"".join(recommended_tags.tolist()) + "\""
        ]

        # Set come variables for rendering the HTML template
        page_name = 'index.html'
        html_string_table = _modify_html_template(
            # Reference: https://stackoverflow.com/questions/50807744/apply-css-class-to-pandas-dataframe-using-to-html
            recommended_videos.to_html(classes='mystyle')
        )

        return render_template(
            page_name,
            tables=[html_string_table],
            tags=recommended_tags,
            index=False,
            index_names=False,
            justify="center",
            bold_rows=True,
            render_links=True,
            encoding="utf-8"
        )


if __name__ == '__main__':
    print(">> Starting the Flask app...")
    app.run(debug=True)
