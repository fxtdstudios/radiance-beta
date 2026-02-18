
import os
import re

DOCS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "docs")
TUTORIALS_PATH = os.path.join(DOCS_DIR, "tutorials.html")

CSS = """
    <style>
        /* Interactive Enhancements */
        .step-check {
            margin-right: 0.75rem;
            width: 1.3rem;
            height: 1.3rem;
            accent-color: var(--accent-gold);
            cursor: pointer;
            transition: transform 0.2s;
        }
        .step-check:hover {
            transform: scale(1.1);
        }
        
        .step h3 {
            display: flex;
            align-items: center;
        }
        
        .step.completed h3, 
        .step.completed p {
            opacity: 0.6;
            text-decoration: none; /* simple dimming */
        }
        
        .workflow-container {
            position: relative;
            margin: 1rem 0;
        }
        
        .copy-btn {
            position: absolute;
            top: 0.5rem;
            right: 0.5rem;
            background: rgba(0,0,0,0.3);
            border: 1px solid var(--border-default);
            color: var(--text-secondary);
            padding: 0.3rem 0.6rem;
            border-radius: 6px;
            font-size: 0.75rem;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 0.4rem;
            transition: all 0.2s;
            backdrop-filter: blur(4px);
        }
        
        .copy-btn:hover {
            background: var(--bg-card-hover);
            color: var(--accent-gold);
            border-color: var(--accent-gold);
        }
        
        .tutorial-nav {
            position: sticky;
            top: 5rem; /* Below main nav */
            z-index: 90;
        }
    </style>
"""

JS = """
    <script>
        document.addEventListener('DOMContentLoaded', () => {
            // 1. Checkbox Progress
            const steps = document.querySelectorAll('.step');
            steps.forEach(step => {
                const checkbox = step.querySelector('.step-check');
                if (!checkbox) return;
                
                // Unique ID based on tutorial section + step number
                const tutorialId = step.closest('.tutorial').id;
                const stepNum = step.getAttribute('data-step');
                const storageId = `radiance-tutorial-${tutorialId}-step-${stepNum}`;
                
                // Restore state
                const isChecked = localStorage.getItem(storageId) === 'true';
                checkbox.checked = isChecked;
                if (isChecked) step.classList.add('completed');
                
                // Handle change
                checkbox.addEventListener('change', (e) => {
                    const checked = e.target.checked;
                    localStorage.setItem(storageId, checked);
                    step.classList.toggle('completed', checked);
                    
                    // Celebration if all steps in tutorial are done? 
                    // (Optional enhancement)
                });
            });

            // 2. Copy Workflow Buttons
            document.querySelectorAll('.copy-btn').forEach(btn => {
                btn.addEventListener('click', () => {
                    const workflowDiv = btn.nextElementSibling; // The .workflow div
                    if (!workflowDiv) return;
                    
                    const text = workflowDiv.innerText;
                    navigator.clipboard.writeText(text).then(() => {
                        const originalHTML = btn.innerHTML;
                        btn.innerHTML = '✅ Copied!';
                        btn.style.color = 'var(--accent-teal)';
                        setTimeout(() => {
                            btn.innerHTML = originalHTML;
                            btn.style.color = '';
                        }, 2000);
                    });
                });
            });
        });
    </script>
"""

def upgrade_tutorials():
    if not os.path.exists(TUTORIALS_PATH):
        print("tutorials.html not found!")
        return

    with open(TUTORIALS_PATH, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. Inject CSS (before </head>)
    if "/* Interactive Enhancements */" not in content:
        content = content.replace("</head>", CSS + "\n</head>")

    # 2. Inject JS (before </body>)
    if "DOMContentLoaded" not in content:
        content = content.replace("</body>", JS + "\n</body>")

    # 3. Inject Checkboxes
    # Find all <div class="step" ...> ... <h3> and insert checkbox
    # Using regex to match specific structure
    # Match: <div class="step" data-step="1"> \n <h3> -> <div...> \n <h3><input...>
    
    # We'll use a reliable replacement pattern
    # Replace '<h3>' with '<h3><input type="checkbox" class="step-check">' BUT only if it's inside a step?
    # Actually, the file structure is consistent. All <h3> inside the body content are either headers (already handled?) or step headers.
    # The tutorial links used <h3> too! We don't want checkboxes there.
    # The links are <a href... class="tutorial-link"> <h3>...</h3> </a>
    # The Step headers are inside <div class="step"...>
    
    # Let's iterate line by line or use specific regex for the step container
    # Complex regex: (<div class="step"[^>]*>\s*<h3>)
    content = re.sub(
        r'(<div class="step"[^>]*>\s*<h3>)', 
        r'\1<input type="checkbox" class="step-check">', 
        content
    )

    # 4. Inject Copy Buttons
    # Wrap .workflow in a container? Or just prepend button if we make it absolute?
    # Current: <div class="workflow">...</div>
    # New: <div class="workflow-container"><button...>Copy</button><div class="workflow">...</div></div>
    
    # Regex for workflow div
    # We match the opening tag and wrap it
    # note: We need to close the container too. This is hard with regex if we don't know where it ends.
    # BUT! My CSS allows `.copy-btn` to be absolute if the parent is relative.
    # If I just inject the button *inside* the `.workflow` div (at the start), and make `.workflow` relative?
    # That changes the DOM.
    # Better: Wrap it.
    
    # Let's try injecting the button *immediately before* `<div class="workflow">` and wrap both in a `div`?
    # No, simple is best.
    # Let's add `position: relative` to `.workflow` styles (I can allow that in CSS).
    # And inject the button *inside* `<div class="workflow">`.
    # Replace `<div class="workflow">` with `<div class="workflow" style="position:relative"><button class="copy-btn">📋 Copy</button>`
    content = re.sub(
        r'(<div class="workflow">)', 
        r'<div class="workflow" style="position: relative; padding-top: 2.5rem;"><button class="copy-btn">📋 Copy</button>', 
        content
    )
    # Added padding-top to make room for the button so it doesn't overlap text.

    with open(TUTORIALS_PATH, "w", encoding="utf-8") as f:
        f.write(content)
    
    print("tutorials.html upgraded successfully!")

if __name__ == "__main__":
    upgrade_tutorials()
