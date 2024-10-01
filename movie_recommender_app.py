import streamlit as st

# Set page config (this must be the first Streamlit command)
st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")

import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import pickle
from datetime import datetime

def parse_date_and_extract_year(date_string):
    if pd.isna(date_string):
        return None
    try:
        # Try parsing as full date
        date = pd.to_datetime(date_string)
        return date.year
    except:
        # If full date parsing fails, try extracting year directly
        import re
        year_match = re.search(r'\d{4}', str(date_string))
        if year_match:
            return int(year_match.group())
        else:
            return None

def create_movie_title_with_year(row):
    year = parse_date_and_extract_year(row['release_date'])
    if year:
        return f"{row['movie_name']} ({year})"
    else:
        return row['movie_name']

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('final_movie_dataset.csv')
    
    # Add a new column with movie title and year
    df['movie_title_with_year'] = df.apply(create_movie_title_with_year, axis=1)
    
    embeddings = np.load('reduced_movie_embeddings.npy')
    with open('movie_dict.pkl', 'rb') as f:
        combined_dict = pickle.load(f)
    movie_dict = combined_dict['movie_data']
    
    # Create a mapping from wikipedia_movie_id to movie_dict index
    wiki_id_to_index = {row['wikipedia_movie_id']: str(i) for i, row in df.iterrows()}
    
    # Update movie_name_to_id to use the new movie_title_with_year
    movie_name_to_id = df.set_index('movie_title_with_year')['wikipedia_movie_id'].to_dict()
    
    return df, embeddings, movie_dict, movie_name_to_id, wiki_id_to_index

df, embeddings, movie_dict, movie_name_to_id, wiki_id_to_index = load_data()

def create_movie_path_graph(df, similarity_matrix, selected_movie_row, percentile=70, max_depth=2, max_connections=5):
    G = nx.Graph()
    start_index = selected_movie_row.name
    start_title = selected_movie_row['movie_title_with_year']
    G.add_node(start_index, title=start_title, depth=0)
    
    def add_connections(node, current_depth):
        if current_depth >= max_depth:
            return
        similarities = similarity_matrix[node]
        threshold = np.percentile(similarities, percentile)
        similar_indices = np.argsort(similarities)[::-1][1:max_connections+1]
        for idx in similar_indices:
            similarity = similarities[idx]
            if similarity > threshold:
                if idx not in G.nodes():
                    movie_title = df.loc[idx, 'movie_title_with_year']
                    G.add_node(idx, title=movie_title, depth=current_depth+1)
                G.add_edge(node, idx, weight=similarity)
                add_connections(idx, current_depth+1)
    
    add_connections(start_index, 0)
    return G

def plot_movie_graph(G):
    pos = nx.spring_layout(G)
    for node in G.nodes():
        G.nodes[node]['pos'] = pos[node]

    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = G.nodes[edge[0]]['pos']
        x1, y1 = G.nodes[edge[1]]['pos']
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )

    node_x, node_y = [], []
    for node in G.nodes():
        x, y = G.nodes[node]['pos']
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='Viridis',
            reversescale=True,
            color=[],
            size=10,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line_width=2
        )
    )

    node_adjacencies = []
    node_text = []
    for node, adjacencies in G.adjacency():
        node_adjacencies.append(len(adjacencies))
        node_text.append(f"{G.nodes[node]['title']}<br># of connections: {len(adjacencies)}")

    node_trace.marker.color = node_adjacencies
    node_trace.text = node_text

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='Movie Similarity Network',
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40),
                        annotations=[dict(
                            text="Movie Recommendation System",
                            showarrow=False,
                            xref="paper", yref="paper",
                            x=0.005, y=-0.002
                        )],
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)'
    ))

    # Add animation
    fig.update_layout(
        updatemenus=[dict(
            type='buttons',
            showactive=False,
            buttons=[dict(
                label='Play',
                method='animate',
                args=[None, dict(frame=dict(duration=500, redraw=True), fromcurrent=True, mode='immediate')]
            )]
        )]
    )

    # Create frames for animation
    frames = [go.Frame(data=[go.Scatter(x=[node_x[i]], y=[node_y[i]], mode='markers', marker=dict(size=20, color='red'))]) for i in range(len(node_x))]
    fig.frames = frames

    return fig

# Load data
df, embeddings, movie_dict, movie_name_to_id, wiki_id_to_index = load_data()

# Load pre-computed similarity matrix
similarity_matrix = np.load('movie_similarity_matrix.npy')

# Custom CSS to improve the look
st.markdown("""
    <style>
    .stApp {
        background-color: #f0f0f5;
    }
    .stButton>button {
        color: #ffffff;
        background-color: #4CAF50;
        border-radius: 5px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    </style>
    """, unsafe_allow_html=True)

# App title
st.title('🎬 Movie Similarity Explorer')

# User input
start_movie = st.selectbox('Select a movie:', df['movie_title_with_year'].tolist())

# Find the correct movie in the dataframe
selected_movie_row = df[df['movie_title_with_year'] == start_movie]

if selected_movie_row.empty:
    st.error(f"No movie found with the title '{start_movie}'")
else:
    selected_movie_row = selected_movie_row.iloc[0]
    
    # Use this row for further processing
    movie_id = selected_movie_row['wikipedia_movie_id']
    
    # Use the mapping to get the correct key for movie_dict
    movie_dict_key = wiki_id_to_index.get(movie_id)
    
    if movie_dict_key and movie_dict_key in movie_dict:
        movie_data = movie_dict[movie_dict_key]
    else:
        # If not, use the data from the DataFrame
        movie_data = selected_movie_row.to_dict()
        st.warning(f"Detailed data for '{start_movie}' not found. Using available information.")

    # Sidebar for parameters
    st.sidebar.header('Graph Parameters')
    percentile = st.sidebar.slider('Similarity Percentile', 0, 100, 70, 5)
    max_depth = st.sidebar.slider('Max Depth', 1, 5, 2)
    max_connections = st.sidebar.slider('Max Connections per Movie', 1, 10, 5)

    if st.button('Generate Graph'):
        with st.spinner('Generating graph...'):
            G = create_movie_path_graph(df, similarity_matrix, selected_movie_row, percentile, max_depth, max_connections)
            fig = plot_movie_graph(G)
            st.plotly_chart(fig, use_container_width=True)

        st.success(f"Graph generated with {G.number_of_nodes()} movies and {G.number_of_edges()} connections!")

        # Display movie details
        st.subheader('Movie Details')
        st.write(f"**Title:** {start_movie}")
        
        # Use a more flexible approach to display movie details
        details_to_display = [
            ('Release Date', 'release_date'),
            ('Box Office Revenue', 'box_office_revenue'),
            ('Runtime', 'runtime'),
            ('Languages', 'languages'),
            ('Countries', 'countries'),
            ('Genres', 'genres'),
            ('Plot Summary', 'plot_summary')
        ]

        for display_name, key in details_to_display:
            if key in movie_data:
                value = movie_data[key]
                if key == 'box_office_revenue':
                    value = f"${value:,}" if isinstance(value, (int, float)) else value
                elif key == 'runtime':
                    value = f"{value} minutes" if isinstance(value, (int, float)) else value
                st.write(f"**{display_name}:** {value}")
            else:
                st.write(f"**{display_name}:** Information not available")

# Add some instructions
st.sidebar.markdown("""
## How to use:
1. Select a movie from the dropdown menu.
2. Adjust the graph parameters in the sidebar.
3. Click 'Generate Graph' to visualize the movie similarity network.
4. Hover over nodes to see movie titles and connection counts.
5. Use the 'Play' button to animate the graph.
6. Zoom, pan, and interact with the graph to explore relationships.
""")