from graphviz import Digraph
import os

# Create the results folder if it doesn't exist
results_dir = "results"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Create a directed graph with vertical orientation (top-to-bottom)
dot = Digraph(comment='Enhanced Phishing URL Detection Workflow', format='png')
dot.attr(rankdir='TB', size='8,10', splines='true')
dot.attr('node', shape='box', style='filled', fontsize='10')

# Define nodes with plain text and simple colors
dot.node('A', 'Data Collection & Acquisition', color='lightblue')
dot.node('B', 'Data Preprocessing\n(Cleaning, Normalization)', color='lightgreen')
dot.node('C', 'Feature Extraction\n(Lexical, Host-based, Temporal)', color='orange')
dot.node('D', 'Model Training\n(RandomForest, Ensemble)', color='yellow')
dot.node('E', 'Interpretability Analysis\n(SHAP, LIME)', color='pink')
dot.node('F', 'Adversarial Robustness Testing', color='violet')
dot.node('G', 'Visualization & Evaluation\n(t-SNE, ROC, Confusion Matrices)', color='lightgrey')
dot.node('H', 'Result Synthesis & Reporting', color='lightcoral')

# Connect the nodes with labeled edges
dot.edge('A', 'B', label='Step 1')
dot.edge('B', 'C', label='Step 2')
dot.edge('C', 'D', label='Step 3')
dot.edge('D', 'E', label='Step 4')
dot.edge('D', 'F', label='Adversarial Testing')
dot.edge('E', 'G', label='Visualization')
dot.edge('F', 'G', label='Evaluation')
dot.edge('G', 'H', label='Final Reporting')

# Render and save the diagram as a PNG file in the results folder
output_path = os.path.join(results_dir, "vertical_overview_workflow")
dot.render(output_path, view=True)
