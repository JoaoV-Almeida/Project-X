---
description: Edit Jupyter Notebook (.ipynb) files by modifying their JSON structure
---

# Editing Jupyter Notebooks

Since `.ipynb` files cannot be edited directly, use the `edit_notebook.py` utility script to make changes.

## Workflow Steps

### 1. Create the edit script
Create a file at the project root called `edit_notebook.py` with the required modifications.

### 2. Script Template
```python
"""
Notebook Editor Script
Usage: python edit_notebook.py
"""
import json

# Path to the notebook
notebook_path = r'PATH_TO_NOTEBOOK.ipynb'

# Load the notebook
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Access cells by index: nb['cells'][0], nb['cells'][1], etc.
# Each cell has:
#   - 'cell_type': 'code' or 'markdown'
#   - 'source': list of strings (each line is a string)
#   - 'outputs': list of output objects (for code cells)
#   - 'execution_count': int or None

# Example: Modify cell 0's source
# nb['cells'][0]['source'] = [
#     "# New content\n",
#     "import pandas as pd\n",
# ]

# Example: Clear outputs from all cells
# for cell in nb['cells']:
#     if cell['cell_type'] == 'code':
#         cell['outputs'] = []
#         cell['execution_count'] = None

# Save the notebook
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("[OK] Notebook updated successfully!")
```

// turbo
### 3. Run the script
```powershell
python edit_notebook.py
```

// turbo
### 4. Clean up
```powershell
del edit_notebook.py
```

## Tips

- **Cell indexing**: Cells are 0-indexed (first cell is `nb['cells'][0]`)
- **Source format**: Each line in source must be a separate string, ending with `\n` except the last line
- **Unicode**: Use `ensure_ascii=False` when saving to preserve special characters
- **View structure**: To see the current notebook structure, use `print(json.dumps(nb['cells'][0], indent=2))`

## Common Operations

### Add a new cell at the end
```python
new_cell = {
    "cell_type": "code",
    "execution_count": None,
    "id": "new_cell_id",
    "metadata": {},
    "outputs": [],
    "source": ["# New cell content\n"]
}
nb['cells'].append(new_cell)
```

### Insert a cell at specific position
```python
nb['cells'].insert(2, new_cell)  # Insert at index 2
```

### Delete a cell
```python
del nb['cells'][3]  # Delete cell at index 3
```

### Replace text in all cells
```python
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        cell['source'] = [line.replace('old_text', 'new_text') for line in cell['source']]
```
