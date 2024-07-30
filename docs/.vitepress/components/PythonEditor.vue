<template>
    <div class="python-editor">
      <textarea v-model="code" class="code-input"></textarea>
      <button @click="runCode">Run Code</button>
      <pre class="output">{{ output }}</pre>
    </div>
  </template>
  
  <script setup>
  import { ref, onMounted, reactive } from 'vue';
  
  // Importing required CodeMirror packages
  import { EditorState } from '@codemirror/state';
  import { EditorView, basicSetup } from '@codemirror/view'; // Adjusted import paths
  import { python } from '@codemirror/lang-python';
  
  // Define where Pyodide will be loaded from
  const PYODIDE_CDN = 'https://cdn.jsdelivr.net/pyodide/v0.18.1/full/pyodide.js';
  
  const code = ref('print("Hello, Python in the browser!")'); // Bind the textarea with v-model
  const output = ref('');
  let pyodide = null;
  let editorView = null;
  
  onMounted(async () => {
    // Load Pyodide
    pyodide = await loadPyodide();
  
    // Load additional Python packages if required
    await pyodide.loadPackage(['numpy', 'pandas']); // Example packages, adjust as needed
  
    // Initialize CodeMirror editor
    editorView = new EditorView({
      state: EditorState.create({
        doc: code.value,
        extensions: [basicSetup, python()],
      }),
      parent: document.querySelector('.code-input'), // Bind CodeMirror to the textarea
    });
  
    // Load a custom Python module from a URL
    await pyodide.runPythonAsync(`
      import micropip
      await micropip.install('https://raw.githubusercontent.com/Crypto-Aggressor/Investment-Management-with-Python-and-Machine-Learning-Specialization/production/PortfolioOptimizationKit.py')
    import PortfolioOptimizationKit')
      import portfolioOptimizationKit
    `);
  });
  
  async function runCode() {
    try {
      // Run the Python code in the editor
      const results = await pyodide.runPythonAsync(editorView.state.doc.toString());
      output.value = results.toString();
    } catch (error) {
      output.value = `Error: ${error.message}`;
    }
  }
  
  // Load Pyodide function
  async function loadPyodide() {
    if (!pyodide) {
      await import(PYODIDE_CDN);
      pyodide = await globalThis.loadPyodide();
    }
    return pyodide;
  }
  </script>
  
  <style>
  .python-editor {
    display: flex;
    flex-direction: column;
    align-items: stretch;
    padding: 10px;
  }
  
  .code-input {
    height: 150px; /* Adjust height of the textarea */
    margin-bottom: 10px;
    font-family: monospace;
  }
  
  .output {
    background-color: #f4f4f4;
    border: 1px solid #ccc;
    padding: 10px;
    white-space: pre-wrap;
  }
  </style>
  