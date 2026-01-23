// MathJax configuration for mkdocs-material (instant navigation).
//
// - We support both markdown pages and notebook-rendered pages.
//   (Notebooks may not wrap math in arithmatex spans.)
// - mkdocs-material uses client-side navigation; we re-typeset after page changes.

window.MathJax = {
  tex: {
    inlineMath: [['\\(', '\\)'], ['$', '$']],
    displayMath: [['\\[', '\\]'], ['$$', '$$']],
    processEscapes: true,
  },
  options: {
    // Process math in all HTML (including notebook-rendered pages).
    // Avoid processing inside code/pre blocks.
    skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code'],
  },
};

// Re-render math on each page load / navigation.
// document$ is injected by mkdocs-material.
document$.subscribe(() => {
  if (window.MathJax && window.MathJax.typesetPromise) {
    window.MathJax.typesetPromise();
  }
});
