document.addEventListener('copy', function(e) {
  // Check if the selection is within a code block
  if (window.getSelection().anchorNode && window.getSelection().anchorNode.parentNode.tagName === 'CODE') {
    let text = window.getSelection().toString();
    // Create a new text string without leading $ signs and new lines that follow them
    let modifiedText = text.replace(/^\$ /gm, '');
    e.clipboardData.setData('text/plain', modifiedText);
    // Prevent the default copy operation
    e.preventDefault();
  }
});
