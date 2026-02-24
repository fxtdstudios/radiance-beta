import os
import re

DOCS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "docs")
FAQ_PATH = os.path.join(DOCS_DIR, "faq.html")

NUKE_SECTION = """
        <!-- Nuke Bridge Section -->
        <section class="faq-section" id="nuke">
            <h2><svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feature-icon"><polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/></svg> Nuke Bridge</h2>

            <div class="faq-item">
                <div class="faq-question">
                    <h3>"Connection Refused" error</h3>
                    <span class="icon">+</span>
                </div>
                <div class="faq-answer">
                    <p>The bridge cannot connect to Nuke if the server isn't running.</p>
                    <div class="solution">
                        <h4> Fix</h4>
                        <ol>
                            <li>Open Nuke</li>
                            <li>Run the script: <code>File > Script Editor</code></li>
                            <li>Load/Run <code>start_nuke_server.py</code> from the Radiance/tools folder</li>
                            <li>Check the Nuke console for "Server listening on port 5555"</li>
                        </ol>
                    </div>
                </div>
            </div>

            <div class="faq-item">
                <div class="faq-question">
                    <h3>Images not appearing in Nuke</h3>
                    <span class="icon">+</span>
                </div>
                <div class="faq-answer">
                    <p>If the connection succeeds but no image appears:</p>
                    <ul>
                        <li>Check if the Write node in Nuke is created but empty</li>
                        <li>Ensure firewall isn't blocking port 5555</li>
                        <li>Verify you are sending a valid image (not None) from ComfyUI</li>
                    </ul>
                </div>
            </div>
        </section>
"""

VIEWER_FIX = """
            <div class="faq-item">
                <div class="faq-question">
                    <h3>Viewer shows black screen</h3>
                    <span class="icon">+</span>
                </div>
                <div class="faq-answer">
                    <p>This usually happens with corrupted HDR data or NaN (Not a Number) values.</p>
                    <div class="solution">
                        <h4> Fixes</h4>
                        <ul>
                            <li><strong>Update Radiance:</strong> v2.1.0+ automatically handles bad values</li>
                            <li><strong>Check VAE:</strong> Some VAEs produce NaNs with fp16. Try using <code>Use fp32 VAE</code> option</li>
                            <li><strong>Fallback:</strong> The viewer now attempts to load a PNG fallback if HDR fails</li>
                        </ul>
                    </div>
                </div>
            </div>
"""

JS = """
    <script>
        document.addEventListener('DOMContentLoaded', () => {
            // FAQ Accordion
            const questions = document.querySelectorAll('.faq-question');
            questions.forEach(q => {
                q.addEventListener('click', () => {
                    const item = q.parentElement;
                    const isOpen = item.classList.contains('open');
                    
                    // Close others? Optional. Let's keep multiple open allowed.
                    
                    item.classList.toggle('open');
                    
                    // Rotate icon
                    const icon = q.querySelector('.icon');
                    if(icon) icon.style.transform = isOpen ? 'rotate(0deg)' : 'rotate(45deg)';
                });
            });
            
            // Search Functionality
            const searchInput = document.querySelector('.search-box input');
            if(searchInput) {
                searchInput.addEventListener('input', (e) => {
                    const term = e.target.value.toLowerCase();
                    document.querySelectorAll('.faq-item').forEach(item => {
                        const text = item.innerText.toLowerCase();
                        item.style.display = text.includes(term) ? 'block' : 'none';
                    });
                });
            }
        });
    </script>
"""


def upgrade_faq():
    if not os.path.exists(FAQ_PATH):
        print("faq.html not found!")
        return

    with open(FAQ_PATH, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. Inject JS (before </body>)
    if "FAQ Accordion" not in content:
        content = content.replace("</body>", JS + "\n</body>")

    # 2. Inject Nuke Section (after Viewer section)
    # Find end of Viewer section
    # Viewer section id="viewer"
    # We look for </section> after id="viewer"
    # This is tricky with simple replace if there are multiple sections.
    # We can perform a regex split or find.

    # Pattern: (<section[^>]*id="viewer"[^>]*>.*?</section>)
    # We want to insert AFTER this.
    if 'id="nuke"' not in content:
        content = re.sub(
            r'(<section[^>]*id="viewer"[^>]*>.*?</section>)',
            r"\1\n" + NUKE_SECTION,
            content,
            flags=re.DOTALL,
        )

    # 3. Inject Viewer Fix (inside Viewer section)
    # We want to append this new item to the viewer section.
    # We can match `</section>` of the viewer section and insert before it.
    # But wait, step 2 might have changed the string.
    # Actually, step 2 appended Nuke AFTER Viewer.
    # So Viewer section is still accessible.

    # Let's find the closing `</section>` of the viewer section specifically.
    # We can match `(<section[^>]*id="viewer"[^>]*>.*?)(\s*</section>)`
    # And insert `VIEWER_FIX` before `</section>`.
    if "Viewer shows black screen" not in content:
        content = re.sub(
            r'(<section[^>]*id="viewer"[^>]*>.*?)(\s*</section>)',
            r"\1" + VIEWER_FIX + r"\2",
            content,
            flags=re.DOTALL,
        )

    # 4. Update Category Nav to include Nuke
    # <div class="category-nav"> ... </div>
    # Append <a href="#nuke" class="category-btn">Nuke Bridge</a>
    if '<a href="#nuke"' not in content:
        content = content.replace(
            '<a href="#viewer" class="category-btn">Viewer</a>',
            '<a href="#viewer" class="category-btn">Viewer</a>\n            <a href="#nuke" class="category-btn">Nuke Bridge</a>',
        )

    # 5. Update Quick Links
    if '<a href="#nuke"' not in content:  # Check again if not added to quick links
        # Find Viewer link in quick links and append Nuke
        # <a href="#viewer" class="quick-link">... Radiance Pro Viewer</a>
        # This is a bit looser, but safe enough.
        # We need an SVG for Nuke. I used one in NUKE_SECTION.
        nuke_link = '                <a href="#nuke" class="quick-link"><svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feature-icon"><polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/></svg> Nuke Bridge</a>'

        # Regex to find the viewer link line
        content = re.sub(
            r'(<a href="#viewer" class="quick-link">.*?</a>)',
            r"\1\n" + nuke_link,
            content,
        )

    with open(FAQ_PATH, "w", encoding="utf-8") as f:
        f.write(content)

    print("faq.html upgraded successfully!")


if __name__ == "__main__":
    upgrade_faq()
