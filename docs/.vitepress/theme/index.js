import DefaultTheme from 'vitepress/theme'
import PythonEditor from '../components/PythonEditor.vue'

export default {
  ...DefaultTheme,
  enhanceApp({ app }) {
    app.component('PythonEditor', PythonEditor);
  }
}
