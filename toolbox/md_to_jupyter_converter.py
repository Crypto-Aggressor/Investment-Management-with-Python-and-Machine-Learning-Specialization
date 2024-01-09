import nbformat as nbf

def md_to_notebook(md_filename, notebook_filename):
    with open(md_filename, 'r') as md_file:
        md_content = md_file.readlines()
    
    nb = nbf.v4.new_notebook()
    md_cell = nbf.v4.new_markdown_cell

    nb['cells'] = [md_cell(''.join(md_content))]

    with open(notebook_filename, 'w') as nb_file:
        nbf.write(nb, nb_file)

# Replace with your specific file paths
md_filename = "C:/Users/amine/bin/github-projects/Investment-Management-with-Python-and-Machine-Learning-Specialization/1-Introduction-to-Portfolio-Construction-and-Analysis-with-Python/1.1-Understanding-Returns-and-Assessing-Risks-with-Value-at-Risk.md"
notebook_filename = "C:/Users/amine/bin/github-projects/Investment-Management-with-Python-and-Machine-Learning-Specialization/1-Introduction-to-Portfolio-Construction-and-Analysis-with-Python/notebooks/1.1-Understanding-Returns-and-Assessing-Risks-with-Value-at-Risk.ipynb"

md_to_notebook(md_filename, notebook_filename)
