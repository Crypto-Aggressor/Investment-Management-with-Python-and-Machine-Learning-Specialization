// vite.config.js
import { defineConfig } from 'vite';
import vue from '@vitejs/plugin-vue';
import path from 'path';

export default defineConfig({
  plugins: [vue()],
  resolve: {
    alias: {
      '@codemirror/state': path.resolve(__dirname, 'node_modules/@codemirror/state/dist/index.js'),
      '@codemirror/view': path.resolve(__dirname, 'node_modules/@codemirror/view/dist/index.js'),
      '@codemirror/lang-python': path.resolve(__dirname, 'node_modules/@codemirror/lang-python/dist/index.js')
    }
  }
});
